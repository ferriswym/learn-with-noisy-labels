# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:56:08 2019

@author: 1
"""

import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# load data
def load_data(train_source, test_source):
    transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.480, 0.457, 0.406), (1, 1, 1))])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.480, 0.457, 0.406), (1, 1, 1))])
    trainset = image_qa_data(train_source, transform_train)
    testset = image_qa_data(test_source, transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=8) 
    
    return trainloader, testloader

class image_qa_data(torch.utils.data.Dataset):
    def __init__(self, source, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        with open(source) as f:
            num = 0
            for l in f.readlines():
                num += 1
                full_path, label = l.split(' ')
                print(num)
                self.data.append(full_path)
                self.labels.append(int(label))
        self.data = np.stack(self.data)#.transpose((0, 3, 1, 2))
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        try:
            name, label = self.data[index], self.labels[index]
        except:
            print(index, self.data.shape)
            
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(name)
        
        if self.transform is not None:
            img = self.transform(img)

        return img, label, name
    
    def __len__(self):
        return len(self.data)