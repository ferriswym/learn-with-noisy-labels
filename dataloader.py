# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:56:08 2019

@author: 1
"""
import torch
import torchvision
from torchvision import transforms

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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=75,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root, train=False, download=True, 
                                          transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=75, 
                                             shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes