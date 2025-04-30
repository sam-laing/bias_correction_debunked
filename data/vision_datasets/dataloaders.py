from torch.utils.data import DataLoader, Subset, SequentialSampler, RandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import autoaugment
import torchvision
from typing import Any, Tuple, Callable, Optional
import os
import torch
import numpy as np
import random
from sklearn.model_selection import StratifiedShuffleSplit
from data.vision_datasets.datasamplers import StatefulSequentialSampler, StatefulDistributedSampler
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from datasets import load_from_disk



def get_loaders(cfg) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Return a train, val and test loader defined under cfg.dataset attribute
    If cfg.val_size is 0, val_loader will be None
    """
    # make CUDA operations deterministic if requested
    if hasattr(cfg, 'deterministic') and cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    #set the random seed for reproducibility with cfg.sampler_seed
    if cfg.sampler_seed is not None:
        sampler_seed = cfg.seed if cfg.one_seeded else cfg.sampler_seed
        torch.manual_seed(sampler_seed)
        np.random.seed(sampler_seed)
        random.seed(sampler_seed)
        
    
    if cfg.dataset == "cifar10":
        return _get_cifar10_loaders(cfg)
    elif cfg.dataset == "cifar100":
        return _get_cifar100_loaders(cfg)
    elif cfg.dataset == "svhn":
        return _get_svhn_loaders(cfg)
    elif cfg.dataset == "tiny_imagenet":
        return _get_tinyimagenet_loaders(cfg)
    elif cfg.dataset == "imagenet":
        return _get_imagenet_loaders(cfg)
    elif cfg.dataset == "cub":
        return _get_cub_loaders(cfg)
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset} not implemented or misspelled")
    




# Base directory for all datasets
dataset_names = {
    "cifar10": "cifar10", 
    "cifar100": "cifar100",
    "svhn": "svhn",
    "tiny_imagenet": "tiny-imagenet-200",
    "cub": "CUB_200_2011"
}

def seed_worker(worker_id: int) -> None:
    """
    Worker initialization function to set seeds for worker processes.
    Makes DataLoader workers deterministic.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataset_path(cfg):
    """Get the full path to the dataset directory"""
    data_dir = "/fast/slaing/data/vision/"
    if cfg.dataset == "imagenet":
        return "/fast/najroldi/data/imagenet/"
    if hasattr(cfg, 'data_dir') and cfg.data_dir:
        # Use provided data_dir if it exists
        return cfg.data_dir
    else:
        # Otherwise construct from base data_dir + dataset name
        return os.path.join(data_dir, dataset_names[cfg.dataset])

def stratified_split(dataset, val_ratio=0.2, random_state=None):
    """
    Create a stratified split of dataset ensuring equal class distribution
    If val_ratio is 0, returns all indices for training and empty list for validation
    
    Args:
        dataset: PyTorch dataset with targets attribute or similar
        val_ratio: Proportion of validation samples (default 0.2)
        random_state: Random seed for reproducibility
        
    Returns:
        train_indices, val_indices
    """
    # If validation ratio is 0, return all indices for training and empty array for validation
    if val_ratio == 0:
        return np.arange(len(dataset)), np.array([], dtype=int)
        
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'labels'):
        targets = dataset.labels
    elif hasattr(dataset, 'label'):
        targets = dataset.label
    else:
        # For ImageFolder datasets that don't have a direct targets attribute
        targets = [sample[1] for sample in dataset.samples] if hasattr(dataset, 'samples') else None
    
    if targets is None:
        raise ValueError("Could not find labels in the dataset")
    
    # Convert targets to numpy array if it's not already
    if isinstance(targets, list) or isinstance(targets, tuple):
        targets = np.array(targets)
    elif isinstance(targets, torch.Tensor):
        targets = targets.numpy()
        
    # Use scikit-learn's stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_indices, val_indices = next(splitter.split(np.zeros(len(targets)), targets))
    
    return train_indices, val_indices

def create_deterministic_loader(
    dataset, 
    cfg, 
    batch_size=None, 
    shuffle=False, 
    drop_last=False,
    generator=None
):
    """Create a dataloader with deterministic settings if requested"""
    batch_size = batch_size or cfg.batch_size
    ddp = dist.is_initialized()
    
    # Create a deterministic random generator if needed
    if cfg.deterministic and generator is None:
        generator = torch.Generator()
        if cfg.one_seeded:
            generator.manual_seed(cfg.seed)
        else:
            generator.manual_seed(cfg.sampler_seed)
    
    if ddp:
        # Distributed data parallel mode
        if cfg.deterministic:

            sampler_seed = cfg.seed if cfg.one_seeded else cfg.sampler_seed
            sampler = DistributedSampler(
                dataset, 
                shuffle=shuffle,
                seed=sampler_seed,
                drop_last=drop_last
            )
        else:
            sampler = DistributedSampler(
                dataset, 
                shuffle=shuffle,
                drop_last=drop_last
            )
        shuffle = False  # Don't use shuffle with DistributedSampler
    elif cfg.deterministic and shuffle:
        # Use deterministic sampler for non-distributed training
        sampler = RandomSampler(dataset, generator=generator)
        shuffle = False  # Don't use shuffle with RandomSampler
    else:
        sampler = None
    
    # Set worker init function for deterministic data loading
    worker_init_fn = seed_worker if cfg.deterministic else None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') else 4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator if cfg.deterministic else None,
        drop_last=drop_last,
        persistent_workers=cfg.num_workers > 0 if hasattr(cfg, 'num_workers') else True
    )

        
def _get_cifar10_loaders(cfg):
    """
    Return a train, val and test loader for CIFAR-10 dataset with stratified splits
    and data augmentation for training. If cfg.val_size is 0, val_loader will be None.
    """
    dataset_path = get_dataset_path(cfg)
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # No augmentation for validation and test sets
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets with no transform initially to perform stratified split
    dataset_no_transform = torchvision.datasets.CIFAR10(
        root=dataset_path, train=True, download=False, transform=None
    )
    
    # Get validation size from config, default to 0.2 if not specified
    val_ratio = getattr(cfg, 'val_size', 0.2)
    
    # Use stratified split
    train_indices, val_indices = stratified_split(
        dataset_no_transform, val_ratio=val_ratio, random_state=cfg.seed if cfg.one_seeded else cfg.sampler_seed
    )
    
    # Create datasets with proper transforms
    train_dataset = Subset(
        torchvision.datasets.CIFAR10(
            root=dataset_path, train=True, download=False, transform=transform_train
        ),
        train_indices
    )
    
    # Create test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root=dataset_path, train=False, download=False, transform=transform_test
    )
    
    # Create deterministic data loaders
    train_loader = create_deterministic_loader(
        train_dataset, cfg, shuffle=True, drop_last=True
    )
    
    # Create validation loader only if validation size > 0
    val_loader = None
    if val_ratio > 0 and len(val_indices) > 0:
        val_dataset = Subset(
            torchvision.datasets.CIFAR10(
                root=dataset_path, train=True, download=False, transform=transform_test
            ),
            val_indices
        )
        
        val_loader = create_deterministic_loader(
            val_dataset, cfg, shuffle=False, drop_last=False
        )
    
    test_loader = create_deterministic_loader(
        test_dataset, cfg, shuffle=False, drop_last=False
    )
    
    return train_loader, val_loader, test_loader

def _get_cifar100_loaders(cfg):
    """
    Return a train, val and test loader for CIFAR-100 dataset with stratified splits
    and data augmentation for training. If cfg.val_size is 0, val_loader will be None.
    """
    dataset_path = get_dataset_path(cfg)
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # No augmentation for validation and test sets
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Load dataset with no transform initially to perform stratified split
    dataset_no_transform = torchvision.datasets.CIFAR100(
        root=dataset_path, train=True, download=False, transform=None
    )
    
    # Get validation size from config, default to 0.2 if not specified
    val_ratio = getattr(cfg, 'val_size', 0.2)
    
    # Use stratified split
    train_indices, val_indices = stratified_split(
        dataset_no_transform, val_ratio=val_ratio, random_state=cfg.seed if cfg.one_seeded else cfg.sampler_seed
    )
    
    # Create datasets with proper transforms
    train_dataset = Subset(
        torchvision.datasets.CIFAR100(
            root=dataset_path, train=True, download=False, transform=transform_train
        ),
        train_indices
    )
    
    # Create test dataset
    test_dataset = torchvision.datasets.CIFAR100(
        root=dataset_path, train=False, download=False, transform=transform_test
    )
    
    # Create deterministic data loaders
    train_loader = create_deterministic_loader(
        train_dataset, cfg, shuffle=True, drop_last=True
    )
    
    # Create validation loader only if validation size > 0
    val_loader = None
    if val_ratio > 0 and len(val_indices) > 0:
        val_dataset = Subset(
            torchvision.datasets.CIFAR100(
                root=dataset_path, train=True, download=False, transform=transform_test
            ),
            val_indices
        )
        
        val_loader = create_deterministic_loader(
            val_dataset, cfg, shuffle=False, drop_last=False
        )
    
    test_loader = create_deterministic_loader(
        test_dataset, cfg, shuffle=False, drop_last=False
    )
    
    return train_loader, val_loader, test_loader

def _get_svhn_loaders(cfg):
    """
    Return a train, val and test loader for SVHN dataset with stratified splits
    and data augmentation for training. If cfg.val_size is 0, val_loader will be None.
    """
    dataset_path = get_dataset_path(cfg)
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
    
    # No augmentation for validation and test sets
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
    
    # Load dataset with no transform initially to perform stratified split
    dataset_no_transform = datasets.SVHN(
        root=dataset_path, split='train', download=False, transform=None
    )
    
    # Create test dataset
    test_dataset = datasets.SVHN(
        root=dataset_path, split='test', download=False, transform=transform_test
    )
    
    # Get validation size from config, default to match test size if not specified
    default_val_ratio = len(test_dataset) / len(dataset_no_transform)
    val_ratio = getattr(cfg, 'val_size', default_val_ratio)
    
    # Use stratified split
    train_indices, val_indices = stratified_split(
        dataset_no_transform, val_ratio=val_ratio, random_state=cfg.seed if cfg.one_seeded else cfg.sampler_seed
    )
    
    # Create datasets with proper transforms
    train_dataset = Subset(
        datasets.SVHN(root=dataset_path, split='train', download=False, transform=transform_train),
        train_indices
    )
    
    # Create deterministic data loaders
    train_loader = create_deterministic_loader(
        train_dataset, cfg, shuffle=True, drop_last=True
    )
    
    # Create validation loader only if validation size > 0
    val_loader = None
    if val_ratio > 0 and len(val_indices) > 0:
        val_dataset = Subset(
            datasets.SVHN(root=dataset_path, split='train', download=False, transform=transform_test),
            val_indices
        )
        
        val_loader = create_deterministic_loader(
            val_dataset, cfg, shuffle=False, drop_last=False
        )
    
    test_loader = create_deterministic_loader(
        test_dataset, cfg, shuffle=False, drop_last=False
    )
    
    return train_loader, val_loader, test_loader

def _get_tinyimagenet_loaders(cfg):
    """
    Return train, validation, and test loaders for TinyImageNet dataset
    with proper directory structure handling for the validation set.
    If cfg.val_size is 0, val_loader will be None.
    """
    import os
    import shutil
    
    dataset_path = get_dataset_path(cfg)
    
    # Check if validation directory needs restructuring
    val_dir = os.path.join(dataset_path, 'val')
    val_img_dir = os.path.join(val_dir, 'images')
    annotations_file = os.path.join(val_dir, 'val_annotations.txt')
    
    # If validation directory has the original structure (not already restructured)
    if os.path.exists(val_img_dir) and os.path.exists(annotations_file):
        print("Restructuring TinyImageNet validation directory...")
        
        # Read validation annotations file to get image-to-class mapping
        val_img_dict = {}
        with open(annotations_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                val_img_dict[parts[0]] = parts[1]
        
        # Create class directories
        for class_name in set(val_img_dict.values()):
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        
        # Move images to corresponding class directories
        for img_name, class_name in val_img_dict.items():
            source = os.path.join(val_img_dir, img_name)
            target = os.path.join(val_dir, class_name, img_name)
            if os.path.exists(source):
                shutil.move(source, target)
        
        # Remove now-empty images directory
        if os.path.exists(val_img_dir) and len(os.listdir(val_img_dir)) == 0:
            os.rmdir(val_img_dir)
            
        print("Validation directory restructured successfully.")
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy.IMAGENET),  # Use ImageNet policy
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    ])
    
    # No augmentation for validation and test sets
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    ])
    
    try:
        # Load datasets from their directories
        train_dataset = datasets.ImageFolder(
            os.path.join(dataset_path, 'train'), 
            transform=transform_train
        )
        
        test_dataset = datasets.ImageFolder(
            os.path.join(dataset_path, 'val'), 
            transform=transform_test
        )
        
        # Check if we should use a separate validation set
        val_ratio = getattr(cfg, 'val_size', 0.0)
        
        # Create deterministic data loaders
        train_loader = create_deterministic_loader(
            train_dataset, cfg, shuffle=True, drop_last=True
        )
        
        # Create validation loader only if validation size > 0
        val_loader = None
        if val_ratio > 0:
            # For TinyImageNet, create validation set from training data
            dataset_no_transform = datasets.ImageFolder(
                os.path.join(dataset_path, 'train'), 
                transform=None
            )
            
            train_indices, val_indices = stratified_split(
                dataset_no_transform, val_ratio=val_ratio, random_state=cfg.seed if cfg.one_seeded else cfg.sampler_seed
            )
            
            # Update train dataset to use only train indices
            train_dataset = Subset(train_dataset, train_indices)
            train_loader = create_deterministic_loader(
                train_dataset, cfg, shuffle=True, drop_last=True
            )
            
            # Create validation dataset
            val_dataset = Subset(
                datasets.ImageFolder(
                    os.path.join(dataset_path, 'train'), 
                    transform=transform_test
                ),
                val_indices
            )
            
            val_loader = create_deterministic_loader(
                val_dataset, cfg, shuffle=False, drop_last=False
            )
        else:
            # If no validation set is requested, use the test set as validation
            val_loader = None
        
        test_loader = create_deterministic_loader(
            test_dataset, cfg, shuffle=False, drop_last=False
        )
        
    except Exception as e:
        print(f"Error loading TinyImageNet datasets: {e}")
        raise
    
    return train_loader, val_loader, test_loader

def _get_imagenet_loaders(cfg):
    """
    Return train, validation, and test loaders for ImageNet with deterministic data loading
    and standard ImageNet preprocessing. If cfg.val_size is 0, val_loader will be None.
    """
    dataset_path = get_dataset_path(cfg)
    
    # Verify the dataset paths exist
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"ImageNet directories not found at {dataset_path}. "
                                f"Expected 'train' and 'val' subdirectories.")
    
    print(f"Loading ImageNet dataset from {dataset_path}...")
    
    # Standard ImageNet preprocessing for training
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Standard ImageNet preprocessing for evaluation
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(val_dir, transform=transform_test)
    
    # Check if we should create a validation set
    val_ratio = getattr(cfg, 'val_size', 0.0)
    
    # Create train loader
    train_loader = create_deterministic_loader(
        train_dataset, cfg, shuffle=True, drop_last=True
    )
    
    # Create validation loader only if validation size > 0
    val_loader = None
    if val_ratio > 0:
        # For ImageNet, create validation set from training data or subset of validation data
        dataset_no_transform = datasets.ImageFolder(train_dir, transform=None)
        train_indices, val_indices = stratified_split(
            dataset_no_transform, val_ratio=val_ratio, random_state=cfg.seed if cfg.one_seeded else cfg.sampler_seed
        )
        
        # Update train dataset
        train_dataset = Subset(train_dataset, train_indices)
        train_loader = create_deterministic_loader(
            train_dataset, cfg, shuffle=True, drop_last=True
        )
        
        # Create validation dataset
        val_dataset = Subset(
            datasets.ImageFolder(train_dir, transform=transform_test),
            val_indices
        )
        
        val_loader = create_deterministic_loader(
            val_dataset, cfg, shuffle=False, drop_last=False
        )
    
    test_loader = create_deterministic_loader(
        test_dataset, cfg, shuffle=False, drop_last=False
    )
    
    return train_loader, val_loader, test_loader



def _get_cub_loaders(cfg):
    """
    Return a train, val and test loader for CUB-200-2011 dataset with stratified splits
    and data augmentation for training. If cfg.val_size is 0, val_loader will be None.
    """
    dataset_path = get_dataset_path(cfg)
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # No augmentation for validation and test sets
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load dataset with no transform initially for stratified split
    dataset_no_transform = datasets.ImageFolder(
        os.path.join(dataset_path, 'train'), transform=None
    )
    
    # Test set
    test_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, 'test'), transform=transform_test
    )
    
    # Get validation size from config or use default based on test set size
    default_val_ratio = len(test_dataset) / len(dataset_no_transform)
    val_ratio = getattr(cfg, 'val_size', default_val_ratio)
    
    # Use stratified split
    train_indices, val_indices = stratified_split(
        dataset_no_transform, val_ratio=val_ratio, random_state=cfg.seed if cfg.one_seeded else cfg.sampler_seed
    )
    
    # Create datasets with proper transforms
    train_dataset = Subset(
        datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=transform_train),
        train_indices
    )
    
    # Create deterministic data loaders
    train_loader = create_deterministic_loader(
        train_dataset, cfg, shuffle=True, drop_last=True
    )
    
    # Create validation loader only if validation size > 0
    val_loader = None
    if val_ratio > 0 and len(val_indices) > 0:
        val_dataset = Subset(
            datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=transform_test),
            val_indices
        )
        
        val_loader = create_deterministic_loader(
            val_dataset, cfg, shuffle=False, drop_last=False
        )
    
    test_loader = create_deterministic_loader(
        test_dataset, cfg, shuffle=False, drop_last=False
    )
    
    return train_loader, val_loader, test_loader

def _get_sampler(dataset, cfg, start_step=0):
    """
    Return a sampler for dataset based on cfg.sampler
    Not used in the main code path, but kept for compatibility
    """
    # DDP check
    ddp = dist.is_initialized()
    
    # This was the default in the original code
    if not hasattr(cfg, 'sampler'):
        if ddp:
            if hasattr(cfg, 'resume') and cfg.resume:
                return StatefulDistributedSampler(
                    dataset, shuffle=True, seed=cfg.seed if cfg.one_seeded else cfg.sampler_seed, start_step=start_step)
            else:
                return DistributedSampler(dataset, shuffle=True, seed=cfg.seed if cfg.one_seeded else cfg.sampler_seed)
        else:
            # No specified sampler, single GPU
            if hasattr(cfg, 'resume') and cfg.resume:
                return StatefulSequentialSampler(dataset, start_step=start_step)
            else:
                return None  # Use the default PyTorch batch sampler
    
    # Specific sampler types
    if cfg.sampler == 'sequential':
        if hasattr(cfg, 'resume') and cfg.resume:
            return StatefulSequentialSampler(dataset, start_step=start_step)
        else:
            return SequentialSampler(dataset)
    elif cfg.sampler == 'random':
        if hasattr(cfg, 'resume') and cfg.resume:
            # Use stateful random sampler here if needed
            return RandomSampler(dataset)
        else:
            return RandomSampler(dataset)
    elif cfg.sampler == 'distributed':
        if hasattr(cfg, 'resume') and cfg.resume:
            return StatefulDistributedSampler(dataset, shuffle=True, seed=cfg.seed if cfg.one_seeded else cfg.sampler_seed, start_step=start_step)
        else:
            return DistributedSampler(dataset, shuffle=True, seed=cfg.seed if cfg.one_seeded else cfg.sampler_seed)
    
    return None  # Default to no sampler if none of the conditions are met