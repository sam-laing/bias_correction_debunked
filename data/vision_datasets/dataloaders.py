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
    if cfg.dataset=="cifar10":
        # maybe better 
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root="/fast/slaing/data/vision/cifar10", train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(
        root="/fast/slaing/data/vision/cifar10", train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader
    




def get_cifar10_loaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    return a train, val and test loader for CIFAR10 dataset
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root="/fast/slaing/data/vision/cifar10", train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(
        root="/fast/slaing/data/vision/cifar10", train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader

def get_svhn_loaders(cfg):
    """
    return a train, val and test loader for SVHN dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.SVHN(
        root=cfg.data_dir, split='train', download=True, transform=transform
    )
    val_dataset = datasets.SVHN(
        root=cfg.data_dir, split='test', download=True, transform=transform
    )
    test_dataset = datasets.SVHN(
        root=cfg.data_dir, split='test', download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    
    return train_loader, val_loader, test_loader