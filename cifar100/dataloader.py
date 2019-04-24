# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:56:08 2019

@author: 1
"""
import os
import sys
import pickle
import torch
import torchvision
import random
import numpy as np
from torchvision import transforms
from PIL import Image

# load data
def load_data():
    transform_train = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    root = './data'
        
    trainset = torchvision.datasets.CIFAR100(root, train=True, download=True,
                                            transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=125,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root, train=False, download=True, 
                                          transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=125, 
                                             shuffle=False, num_workers=2)
    
    return trainloader, testloader

class CIFAR100_sym(torchvision.datasets.CIFAR100):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, noise_rate=0.2):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['fine_labels'])
        for idx in range(len(self.targets)):
            if random.random() < noise_rate:
                self.targets[idx] = random.randint(0, 99)

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.ids = list(range(len(self.targets)))

        self._load_meta()
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, idx = self.data[index], self.targets[index], self.ids[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, idx

class CIFAR100_asym(CIFAR100_sym):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, noise_rate=0.2):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['fine_labels'])
        for idx in range(len(self.targets)):
            if random.random() < noise_rate:
                self.targets[idx] = (self.targets[idx] + 1) % 100

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.ids = list(range(len(self.targets)))

        self._load_meta()

def load_data_sym(noise_rate=0.2):
    transform_train = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    root = './data'
        
    trainset = CIFAR100_sym(root, train=True, download=True,
                            transform=transform_train, noise_rate=noise_rate)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=125,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root, train=False, download=True, 
                                          transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=125, 
                                             shuffle=False, num_workers=2)
    
    return trainloader, testloader

def load_data_asym(noise_rate=0.2):
    transform_train = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    root = './data'
        
    trainset = CIFAR100_asym(root, train=True, download=True,
                            transform=transform_train, noise_rate=noise_rate)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=125,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root, train=False, download=True, 
                                          transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=125, 
                                             shuffle=False, num_workers=2)
    
    return trainloader, testloader