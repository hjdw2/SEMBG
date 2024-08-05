from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

crop_size = 32
padding = 4

def prepare_cifar100_train_dataset(data_dir, dataset='cifar100', batch_size=128,
                                    shuffle=True, num_workers=4):

    train_transform = transforms.Compose([
        transforms.RandomCrop(crop_size, padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761]),
    ])
    train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True,
                                                 download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                pin_memory=True)
    return train_loader

def prepare_cifar100_test_dataset(data_dir, dataset='cifar100', batch_size=128,
                                    shuffle=False, num_workers=4):
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761]),
        ])
    testset = torchvision.datasets.CIFAR100(root=data_dir,
                                               train=False,
                                               download=True,
                                               transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                pin_memory=True)
    return test_loader

def prepare_cifar10_train_dataset(data_dir, dataset='cifar10', batch_size=128,
                                    shuffle=True, num_workers=4):

    train_transform = transforms.Compose([
        transforms.RandomCrop(crop_size, padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2471, 0.2435, 0.2616]),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                 download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                pin_memory=True)
    return train_loader

def prepare_cifar10_test_dataset(data_dir, dataset='cifar10', batch_size=128,
                                    shuffle=False, num_workers=4):
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2471, 0.2435, 0.2616]),
        ])
    testset = torchvision.datasets.CIFAR10(root=data_dir,
                                               train=False,
                                               download=True,
                                               transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                pin_memory=True)
    return test_loader
