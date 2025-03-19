from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision
from typing import Any, Tuple
import os
import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

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
        root=cfg.data_dir if hasattr(cfg, 'data_dir') else "/fast/slaing/data/vision/cifar10", 
        train=True, download=False, transform=None
    )
    
    # Use stratified split with validation ratio of 0.2 (10,000 out of 50,000)
    train_indices, val_indices = stratified_split(
        dataset_no_transform, val_ratio=0.2, random_state=cfg.seed
    )
    
    # Create datasets with proper transforms
    train_dataset = Subset(
        torchvision.datasets.CIFAR10(
            root=cfg.data_dir if hasattr(cfg, 'data_dir') else "/fast/slaing/data/vision/cifar10", 
            train=True, download=False, transform=transform_train
        ),
        train_indices
    )
    
    val_dataset = Subset(
        torchvision.datasets.CIFAR10(
            root=cfg.data_dir if hasattr(cfg, 'data_dir') else "/fast/slaing/data/vision/cifar10", 
            train=True, download=False, transform=transform_test
        ),
        val_indices
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=cfg.data_dir if hasattr(cfg, 'data_dir') else "/fast/slaing/data/vision/cifar10", 
        train=False, download=False, transform=transform_test
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
        root=cfg.data_dir, train=True, download=False, transform=None
    )
    
    # Use stratified split with validation ratio of 0.2 (10,000 out of 50,000)
    train_indices, val_indices = stratified_split(
        dataset_no_transform, val_ratio=0.2, random_state=cfg.seed
    )
    
    # Create datasets with proper transforms
    train_dataset = Subset(
        torchvision.datasets.CIFAR100(
            root=cfg.data_dir, train=True, download=False, transform=transform_train
        ),
        train_indices
    )
    
    val_dataset = Subset(
        torchvision.datasets.CIFAR100(
            root=cfg.data_dir, train=True, download=False, transform=transform_test
        ),
        val_indices
    )
    
    test_dataset = torchvision.datasets.CIFAR100(
        root=cfg.data_dir, train=False, download=False, transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def _get_svhn_loaders(cfg):
    """
    Return a train, val and test loader for SVHN dataset with stratified splits
    and data augmentation for training
    """
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
        root=cfg.data_dir, split='train', download=False, transform=None
    )
    
    # Calculate validation size to match test set size
    test_dataset = datasets.SVHN(
        root=cfg.data_dir, split='test', download=False, transform=transform_test
    )
    
    # Calculate validation ratio to match test set size
    val_ratio = len(test_dataset) / len(dataset_no_transform)
    
    # Use stratified split
    train_indices, val_indices = stratified_split(
        dataset_no_transform, val_ratio=val_ratio, random_state=cfg.seed
    )
    
    # Create datasets with proper transforms
    train_dataset = Subset(
        datasets.SVHN(root=cfg.data_dir, split='train', download=False, transform=transform_train),
        train_indices
    )
    
    val_dataset = Subset(
        datasets.SVHN(root=cfg.data_dir, split='train', download=False, transform=transform_test),
        val_indices
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def _get_tinyimagenet_loaders(cfg):
    """
    Return a train, val and test loader for TinyImageNet dataset with stratified splits
    and data augmentation for training
    """
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
    
    # Load dataset with no transform initially for stratified split
    dataset_no_transform = datasets.ImageFolder(
        os.path.join(cfg.data_dir, 'train'), transform=None
    )
    
    # Test set
    test_dataset = datasets.ImageFolder(
        os.path.join(cfg.data_dir, 'val'), transform=transform_test
    )
    
    # Calculate validation ratio to match test set size
    val_ratio = len(test_dataset) / len(dataset_no_transform)
    
    # Use stratified split
    train_indices, val_indices = stratified_split(
        dataset_no_transform, val_ratio=val_ratio, random_state=cfg.seed
    )
    
    # Create datasets with proper transforms
    train_dataset = Subset(
        datasets.ImageFolder(os.path.join(cfg.data_dir, 'train'), transform=transform_train),
        train_indices
    )
    
    val_dataset = Subset(
        datasets.ImageFolder(os.path.join(cfg.data_dir, 'train'), transform=transform_test),
        val_indices
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def _get_cub_loaders(cfg):
    """
    Return a train, val and test loader for CUB-200-2011 dataset with stratified splits
    and data augmentation for training
    """
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
        os.path.join(cfg.data_dir, 'train'), transform=None
    )
    
    # Test set
    test_dataset = datasets.ImageFolder(
        os.path.join(cfg.data_dir, 'test'), transform=transform_test
    )
    
    # Calculate validation ratio to match test set size
    val_ratio = len(test_dataset) / len(dataset_no_transform)
    
    # Use stratified split
    train_indices, val_indices = stratified_split(
        dataset_no_transform, val_ratio=val_ratio, random_state=cfg.seed
    )
    
    # Create datasets with proper transforms
    train_dataset = Subset(
        datasets.ImageFolder(os.path.join(cfg.data_dir, 'train'), transform=transform_train),
        train_indices
    )
    
    val_dataset = Subset(
        datasets.ImageFolder(os.path.join(cfg.data_dir, 'train'), transform=transform_test),
        val_indices
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader








