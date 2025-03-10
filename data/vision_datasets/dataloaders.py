from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchvision
from typing import Any, Tuple
import os
import torch
import numpy as np

def get_loaders(cfg)-> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    return a train, val and test loader defined under cfg.dataset attribute
    """
    # set the random seed for reproducibility with cfg.seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.dataset == "cifar10":
        return _get_cifar10_loaders(cfg)
    elif cfg.dataset == "svhn":
        return _get_svhn_loaders(cfg)
    elif cfg.dataset == "tiny_imagenet":
        return _get_tinyimagenet_loaders(cfg)
    else:
        raise NotImplementedError(f"dataset {cfg.dataset} not implemented or misspelled")
        

def _get_cifar10_loaders(cfg):

    N_train, N_val = 40_000, 10_000

    
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # make the trainset and valset according to N_train and N_val
    trainset = torchvision.datasets.CIFAR10(
        root="/fast/slaing/data/vision/cifar10", train=True, download=True, transform=transform_train)
    trainset, valset = torch.utils.data.random_split(trainset, [N_train, N_val])

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=4
        )
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=cfg.batch_size, shuffle=False, num_workers=4
        )
    

    testset = torchvision.datasets.CIFAR10(
        root="/fast/slaing/data/vision/cifar10", train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

def _get_cifar100_loaders(cfg):
    """
    Return a train, val and test loader for CIFAR-100 dataset with proper splits
    and data augmentation for training
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Load datasets
    full_train_dataset = torchvision.datasets.CIFAR100(
        root=cfg.data_dir, train=True, download=False, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=cfg.data_dir, train=False, download=False, transform=transform_test
    )
    
    # Calculate proper sizes for train/val split - use 80/20 split of training data
    total_train_size = len(full_train_dataset)  
    N_val = len(test_dataset)                   
    N_train = total_train_size - N_val    
    
    # Split training set into train and validation
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [N_train, N_val], 
        generator=torch.Generator().manual_seed(cfg.seed)
    )
    
    # For validation set, we need to replace the transform
    val_dataset = torch.utils.data.Subset(
        torchvision.datasets.CIFAR100(root=cfg.data_dir, train=True, download=False, transform=transform_test),
        val_dataset.indices
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, 
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=cfg.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=cfg.num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def _get_svhn_loaders(cfg):
    """
    Return a train, val and test loader for SVHN dataset with proper splits
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
    # Load the full training set
    full_train_dataset = datasets.SVHN(
        root=cfg.data_dir, split='train', download=False, transform=transform_train
    )
    # Test set
    test_dataset = datasets.SVHN(
        root=cfg.data_dir, split='test', download=False, transform=transform_test
    )
    # Split training set into train and validation
    N = len(full_train_dataset)
    N_val = len(test_dataset)
    N_train = N - N_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [N_train, N_val], 
        generator=torch.Generator().manual_seed(cfg.seed)
    )
    # For validation set, we need to replace the transform
    val_dataset = torch.utils.data.Subset(
        datasets.SVHN(root=cfg.data_dir, split='train', download=False, transform=transform_test),
        val_dataset.indices
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, 
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=cfg.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=cfg.num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

    
def _get_tinyimagenet_loaders(cfg):
    """
    Return a train, val and test loader for TinyImageNet dataset with proper splits
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
    # Load the full training set
    full_train_dataset = datasets.ImageFolder(
        os.path.join(cfg.data_dir, 'train'), transform=transform_train
    )
    # Test set
    test_dataset = datasets.ImageFolder(
        os.path.join(cfg.data_dir, 'test'), transform=transform_test
    )
    # Split training set into train and validation
    N = len(full_train_dataset)
    N_val = len(test_dataset)
    N_train = N - N_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [N_train, N_val], 
        generator=torch.Generator().manual_seed(cfg.seed)
    )
    # For validation set, we need to replace the transform
    val_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(os.path.join(cfg.data_dir, 'train'), transform=transform_test),
        val_dataset.indices
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, 
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=cfg.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=cfg.num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def _get_cub_loaders(cfg):
    """
    Return a train, val and test loader for CUB-200-2011 dataset with proper splits
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
    # Load the full training set
    full_train_dataset = datasets.ImageFolder(
        os.path.join(cfg.data_dir, 'train'), transform=transform_train
    )
    # Test set
    test_dataset = datasets.ImageFolder(
        os.path.join(cfg.data_dir, 'test'), transform=transform_test
    )
    # Split training set into train and validation
    N = len(full_train_dataset)
    N_val = len(test_dataset)
    N_train = N - N_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [N_train, N_val], 
        generator=torch.Generator().manual_seed(cfg.seed)
    )
    # For validation set, we need to replace the transform
    val_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(os.path.join(cfg.data_dir, 'train'), transform=transform_test),
        val_dataset.indices
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, 
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=cfg.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=cfg.num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader










