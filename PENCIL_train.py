#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:28:52 2019

@author: yuming
"""
import sys
sys.path.append('/home/yuming/projects/learn-with-noisy-labels/nets')
import copy

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from squeezenet import squeezenet1_1

ALPHA = 0.1
BETA = 0.4
LAMBDA = 10000
EPSILON = 1e-6

num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def SumLog(factor, exp):
    return torch.sum(factor * torch.log(exp + EPSILON))

def total_loss(outputs_s, label_contrib, noisy_labels):
    # competibility loss
    Lo_loss = F.cross_entropy(label_contrib, noisy_labels)
    
    # classification loss
    outputs = F.softmax(outputs_s)
    label_distrib = F.softmax(label_contrib)
    Lc_loss = SumLog(outputs, outputs / (label_distrib + EPSILON))
    
    # cross entropy loss
    Le_loss = -SumLog(outputs, outputs)
    
    num_classes = label_contrib.shape[1]
    loss = 1/num_classes * Lc_loss + ALPHA * Lo_loss + BETA/num_classes * Le_loss
    if np.isnan(loss.cpu().detach().numpy()):
        print(outputs, label_distrib, noisy_labels)
        sys.exit()
    return 1/num_classes * Lc_loss + ALPHA * Lo_loss + BETA/num_classes * Le_loss

def train_model(model, trainloader, testloader, num_epochs=10, num_classes=3):
        
    # initialize labels distribution
    init_fact = 10
    label_dict = {}
    for data in trainloader:
        names, images, noisy_labels = data
        for idx in range(len(names)):
            if names[idx] not in label_dict:
                name = names[idx]
                noisy_label = noisy_labels[idx]
                noisy_onehot = torch.zeros(num_classes).scatter_(0, noisy_label, 1)
                label_dict[name] = init_fact * noisy_onehot
                
    # training on each epoch
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-5)
    val_acc_history = []
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        model.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            names, inputs, noisy_labels = data
            labels_contrib = []
            for idx in range(len(names)):
                name = names[idx]
                labels_contrib.append(label_dict[name])
            labels_contrib = torch.stack(labels_contrib)
            inputs, labels_contrib, noisy_labels = inputs.to(device), \
                                labels_contrib.to(device), noisy_labels.to(device)
            
            labels_contrib.requires_grad = True
            optimizer.zero_grad()
            outputs = model(inputs)            
            loss = total_loss(outputs, labels_contrib, noisy_labels)            
            loss.backward()
            optimizer.step()
            
            if labels_contrib.grad is not None:
                for idx in range(len(names)):
                    name = names[idx]
                    label_dict[name] -= LAMBDA * labels_contrib.grad[idx].cpu()
                labels_contrib.grad.zero_()
            else:
                exit()
                    
        model.eval()
        correct = 0
        total = 0
        
        for data in testloader:
            _, images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels)
            
        # statistics
        eval_acc = correct.double() / total
        val_acc_history.append(eval_acc)
        
        # deep copy the best model
        if eval_acc > best_acc:
            best_acc = eval_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "models/model_squeezenet_best.pth")
                    
        print(eval_acc)
            
#        print('loss: %.5f' % (running_loss / (i+1)))

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

        return name, img, label
    
    def __len__(self):
        return len(self.data)
    
if __name__ == "__main__":
    # load model
    model = squeezenet1_1(num_classes=3)
    state_dict = torch.load('/home/yuming/projects/learn-with-noisy-labels/models/model_squeezenet_20190409.pth')
    model.load_state_dict(state_dict)
    
    # load data
    transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.480, 0.457, 0.406), (1, 1, 1))])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.480, 0.457, 0.406), (1, 1, 1))])
    train_source = '/data/yuming/image_qa/data/fuzzy_train_crop.txt'
    test_source = '/data/yuming/image_qa/data/fuzzy_test_crop.txt'
    trainset = image_qa_data(train_source, transform_train)
    testset = image_qa_data(test_source, transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=8)
    
    
    train_model(model, trainloader, testloader)