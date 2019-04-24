#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:11:31 2019

@author: yuming
"""

import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("/home/yuming/projects/learn-with-noisy-labels")
from nets.resnet import ResNet34
from torchvision import models
from dataloader import load_data
from dataloader import load_data_sym
from dataloader import load_data_asym

# Detect if we have a GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_name = "../models/raw_resnet.pth"

# Number of epochs to train for
num_epochs = 200

def train_model(model, trainloader, testloader, num_epochs=num_epochs):
    since = time.time()
    model.to(device)
    
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 120, 160], gamma=0.2)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        
        scheduler.step()
        running_loss = 0.0
        model.train()

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels, _ = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print('loss: %.5f' % (running_loss / (i+1)))
        
        # evaluation
        running_loss = 0.0
        model.eval()
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
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
            torch.save(best_model_wts, save_name)
                
        print(eval_acc)
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

# Plot the training curves of validation accuracy
def plot(hist):
    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, num_epochs+1), hist)
    plt.ylim(0, 1.)
    plt.xticks(np.arange(0, len(hist) + 1, 50))
    plt.show()
    
if __name__ == '__main__':
    # data
#    trainloader, testloader = load_data()
#    trainloader, testloader = load_data_asym(noise_rate=0.3)
    noise_rate=0.1
    trainloader, testloader = load_data_asym(noise_rate=noise_rate)
    
    # model
    model = ResNet34()
#    model.load_state_dict(models.resn, classeset34().state_dict(), strict=False)
    print(model)
    print("noise rate: %.1f"%noise_rate)
        
    model, hist = train_model(model, trainloader, testloader, num_epochs)
    plot(hist)
    print("noise rate: %.1f"%noise_rate)