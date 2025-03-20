from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision
from typing import Any, Tuple
import os
import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Base directory for all datasets
data_dir = "/fast/slaing/data/vision/"

# Dataset directory name mappings
dataset_names = {
    "cifar10": "cifar10", 
    "cifar100": "cifar100",
    "svhn": "svhn",
    "tiny_imagenet": "tiny-imagenet-200",
    "cub": "CUB_200_2011"
}

def get_dataset_path(cfg):
    """Get the full path to the dataset directory"""
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
    
    Args:
        dataset: PyTorch dataset with targets attribute or similar
        val_ratio: Proportion of validation samples (default 0.2)
        random_state: Random seed for reproducibility
        
    Returns:
        train_indices, val_indices
    """
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

def get_loaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return a train, val and test loader defined under cfg.dataset attribute
    """
    # Set the random seed for reproducibility with cfg.seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    if cfg.dataset == "cifar10":
        return _get_cifar10_loaders(cfg)
    elif cfg.dataset == "cifar100":
        return _get_cifar100_loaders(cfg)
    elif cfg.dataset == "svhn":
        return _get_svhn_loaders(cfg)
    elif cfg.dataset == "tiny_imagenet":
        return _get_tinyimagenet_loaders(cfg)
    elif cfg.dataset == "cub":
        return _get_cub_loaders(cfg)
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset} not implemented or misspelled")
        
def _get_cifar10_loaders(cfg):
    """
    Return a train, val and test loader for CIFAR-10 dataset with stratified splits
    and data augmentation for training
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
    
    # Use stratified split with validation ratio of 0.2 (10,000 out of 50,000)
    train_indices, val_indices = stratified_split(
        dataset_no_transform, val_ratio=0.2, random_state=cfg.seed
    )
    
    # Create datasets with proper transforms
    train_dataset = Subset(
        torchvision.datasets.CIFAR10(
            root=dataset_path, train=True, download=False, transform=transform_train
        ),
        train_indices
    )
    
    val_dataset = Subset(
        torchvision.datasets.CIFAR10(
            root=dataset_path, train=True, download=False, transform=transform_test
        ),
        val_indices
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=dataset_path, train=False, download=False, transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') else 4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') else 4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') else 4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def _get_cifar100_loaders(cfg):
    """
    Return a train, val and test loader for CIFAR-100 dataset with stratified splits
    and data augmentation for training
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
    
    # Use stratified split with validation ratio of 0.2 (10,000 out of 50,000)
    train_indices, val_indices = stratified_split(
        dataset_no_transform, val_ratio=0.2, random_state=cfg.seed
    )
    
    # Create datasets with proper transforms
    train_dataset = Subset(
        torchvision.datasets.CIFAR100(
            root=dataset_path, train=True, download=False, transform=transform_train
        ),
        train_indices
    )
    
    val_dataset = Subset(
        torchvision.datasets.CIFAR100(
            root=dataset_path, train=True, download=False, transform=transform_test
        ),
        val_indices
    )
    
    test_dataset = torchvision.datasets.CIFAR100(
        root=dataset_path, train=False, download=False, transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') else 4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') else 4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') else 4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def _get_svhn_loaders(cfg):
    """
    Return a train, val and test loader for SVHN dataset with stratified splits
    and data augmentation for training
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
    
    # Calculate validation size to match test set size
    test_dataset = datasets.SVHN(
        root=dataset_path, split='test', download=False, transform=transform_test
    )
    
    # Calculate validation ratio to match test set size
    val_ratio = len(test_dataset) / len(dataset_no_transform)
    
    # Use stratified split
    train_indices, val_indices = stratified_split(
        dataset_no_transform, val_ratio=val_ratio, random_state=cfg.seed
    )
    
    # Create datasets with proper transforms
    train_dataset = Subset(
        datasets.SVHN(root=dataset_path, split='train', download=False, transform=transform_train),
        train_indices
    )
    
    val_dataset = Subset(
        datasets.SVHN(root=dataset_path, split='train', download=False, transform=transform_test),
        val_indices
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') else 4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') else 4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') else 4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader



def _get_tinyimagenet_loaders(cfg):
    """
    Return train, validation, and test loaders for TinyImageNet dataset
    with proper directory structure handling for the validation set.
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
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    ])
    
    # No augmentation for validation and test sets
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    ])
    
    # Load datasets from their directories
    try:
        train_dataset = datasets.ImageFolder(
            os.path.join(dataset_path, 'train'), 
            transform=transform_train
        )
        
        val_dataset = datasets.ImageFolder(
            os.path.join(dataset_path, 'val'), 
            transform=transform_test
        )
        
        # Debug info
        print(f"TinyImageNet train set: {len(train_dataset)} images")
        print(f"TinyImageNet val set: {len(val_dataset)} images")
        
        # Verify class distribution in validation set
        val_targets = [label for _, label in val_dataset.samples]
        class_counts = {}
        for class_idx in range(200):
            count = val_targets.count(class_idx)
            class_counts[class_idx] = count
            if count != 50:  # Each class should have 50 samples
                print(f"Warning: Class {class_idx} has {count} samples (expected 50)")
        
        # TinyImageNet typically uses the validation set as test set
        test_dataset = val_dataset
        
    except Exception as e:
        print(f"Error loading TinyImageNet datasets: {e}")
        raise
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') else 4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') else 4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') else 4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def _get_imagenet_loaders(cfg):
    pass


def _get_cub_loaders(cfg):
    """
    Return a train, val and test loader for CUB-200-2011 dataset with stratified splits
    and data augmentation for training
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
    
    # Calculate validation ratio to match test set size
    val_ratio = len(test_dataset) / len(dataset_no_transform)
    
    # Use stratified split
    train_indices, val_indices = stratified_split(
        dataset_no_transform, val_ratio=val_ratio, random_state=cfg.seed
    )
    
    # Create datasets with proper transforms
    train_dataset = Subset(
        datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=transform_train),
        train_indices
    )
    
    val_dataset = Subset(
        datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=transform_test),
        val_indices
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') else 4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') else 4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') else 4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

    