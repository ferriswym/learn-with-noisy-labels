#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:39:24 2019

@author: yuming
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("/home/yuming/projects/learn-with-noisy-labels")
from tensorboardX import SummaryWriter
from nets.resnet import ResNet34
from torchvision import models
from dataloader import load_data
from dataloader import load_data_sym
from dataloader import load_data_asym

# Detect if we have a GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_name = "../models/PENCIL.pth"
log_file = 'log.txt'
writer = SummaryWriter()

def backbond_train(model, trainloader, testloader, num_epochs):
    since = time.time()
    model.to(device)
    with open(log_file, 'w') as f:
        f.write('')

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.35, momentum=0.9, weight_decay=1e-4)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        
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
        with open(log_file, 'a') as f:
            f.write('loss: %.5f\n' % (running_loss / (i+1)))
        
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
        print(eval_acc)
    
    time_elapsed = time.time() - since
    torch.save(model, "../models/Backbond.pth")
    print('Backbond training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return model

ALPHA = 0.1
BETA = 0.4
LAMBDA = 10000
#EPSILON = 1e-20

def SumLogSoftmax(factor, inputs):
    return torch.sum(F.softmax(factor, 1) * F.log_softmax(inputs, 1)) / factor.shape[0]

def total_loss(outputs, label_contrib, noisy_labels):
    # competibility loss
    Lo_loss = F.cross_entropy(label_contrib, noisy_labels)
    
    # classification loss
    Lc_loss = SumLogSoftmax(outputs, outputs) - SumLogSoftmax(outputs, label_contrib)
    
    # cross entropy loss
    Le_loss = -SumLogSoftmax(outputs, outputs)
    
    num_classes = label_contrib.shape[1]
#    noisy_onehot = torch.zeros(label_contrib.shape).scatter_(1, noisy_labels.cpu().resize_(noisy_labels.shape[0], 1), 1)
#    noisy_onehot = noisy_onehot.to(device)
##    print(noisy_onehot[0], outputs[0])
#    cls_loss = F.cross_entropy(outputs, noisy_labels)
#    cls_loss2 = -SumLogSoftmax(noisy_onehot, outputs)
#    print(cls_loss, cls_loss2)
    loss = 1/num_classes * Lc_loss + ALPHA * Lo_loss + BETA/num_classes * Le_loss
#    if np.isnan(loss.cpu().detach().numpy()):
#        print(outputs, label_distrib, noisy_labels)
#        sys.exit()
#    print(1/num_classes * Lc_loss, ALPHA * Lo_loss, BETA/num_classes * Le_loss)
    return loss

def PENCIL_train(model, trainloader, testloader, num_epochs, num_classes=100):
    since = time.time()
    model.to(device)
    with open(log_file, 'w') as f:
        f.write('')

    # initialize labels distribution
    init_fact = 10
    label_dict = {}
    for data in trainloader:
        images, noisy_labels, ids = data
        for idx in range(len(ids)):
            if int(ids[idx]) not in label_dict:
                num = int(ids[idx])
                noisy_label = noisy_labels[idx]
                noisy_onehot = torch.zeros(num_classes).scatter_(0, noisy_label, 1)
                label_dict[num] = init_fact * noisy_onehot

    # training on each epoch
    optimizer = optim.SGD(model.parameters(), lr=0.035, momentum=0.9, weight_decay=1e-4)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))        
        running_loss = 0.0
        running_Lo = 0.0
        running_Le = 0.0
        running_Lc = 0.0
        model.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, noisy_labels, ids = data
            labels_contrib = []
            for idx in range(len(ids)):
                num = int(ids[idx])
                labels_contrib.append(label_dict[num])
            labels_contrib = torch.stack(labels_contrib)
            inputs, labels_contrib, noisy_labels = inputs.to(device), \
                                labels_contrib.to(device), noisy_labels.to(device)
            
            labels_contrib.requires_grad = True
            optimizer.zero_grad()
            outputs = model(inputs)
            Lo_loss = ALPHA * F.cross_entropy(labels_contrib, noisy_labels)
            Le_loss = -BETA/num_classes * SumLogSoftmax(outputs, outputs)
            Lc_loss = 1/num_classes * (SumLogSoftmax(outputs, outputs) - SumLogSoftmax(outputs, labels_contrib))
            loss = Lo_loss + Le_loss + Lc_loss
#            loss = total_loss(outputs, labels_contrib, noisy_labels)            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_Lo += Lo_loss.item()
            running_Le += Le_loss.item()
            running_Lc += Lc_loss.item()
            
            if labels_contrib.grad is not None:
                for idx in range(len(ids)):
                    num = int(ids[idx])
                    label_dict[num] -= LAMBDA * labels_contrib.grad[idx].cpu()
                labels_contrib.grad.zero_()
            else:
                sys.exit()
        print('loss: %.5f' % (running_loss / (i+1)))
        writer.add_scalar('loss/total_loss', running_loss / (i+1), epoch)
        writer.add_scalar('loss/Lo_loss', running_Lo / (i+1), epoch)
        writer.add_scalar('loss/Lc_loss', running_Lc / (i+1), epoch)
        writer.add_scalar('loss/Le_loss', running_Le / (i+1), epoch)
        with open(log_file, 'a') as f:
            f.write('loss: %.5f\n' % (running_loss / (i+1)))
            
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
        print(eval_acc)
    
    with open('soft_label.txt', 'w') as f:    
        for num in label_dict:
            try:
                f.write("%s: %s\n"%(str(num), str(F.softmax(label_dict[num]))))
            except:
                pass
            
    time_elapsed = time.time() - since
    torch.save(model.state_dict(), save_name)
    print('PENCIL training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return model, label_dict

def fine_tune(model, label_distrib, trainloader, testloader, num_epochs, num_classes=100):
    since = time.time()
    model.to(device)

    # training on each epoch
    optimizer = optim.SGD(model.parameters(), lr=2e-2, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [40, 80])
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        
        scheduler.step()
        running_loss = 0.0
        model.train()
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, _, ids = data
            labels = []
            for idx in range(len(ids)):
                num = int(ids[idx])
                labels.append(label_distrib[num])
            labels = torch.stack(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            
            labels.requires_grad = False
            optimizer.zero_grad()
            outputs = model(inputs) 
            loss = 1/num_classes * (SumLogSoftmax(outputs, outputs) - SumLogSoftmax(outputs, labels))   
            loss.backward()
            optimizer.step()
            running_loss += loss.item()            
        
        print('loss: %.5f' % (running_loss / (i+1)))
        with open(log_file, 'a') as f:
            f.write('loss: %.5f\n' % (running_loss / (i+1)))
            
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
        print(eval_acc) 
        
    time_elapsed = time.time() - since
    print('Fine-tune complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    torch.save(model, "../models/finetune.pth")
    return model

# Plot the training curves of validation accuracy
def plot(hist, num_epochs):
    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, num_epochs+1), hist)
    plt.ylim(0, 1.)
    plt.xticks(np.arange(0, len(hist) + 1, 50))
    plt.show()

def draw_loss(log_file):
    loss = []
    with open(log_file, 'r') as f:
        for l in f.readlines():
            if l.split(' ')[0] == 'loss:':
                loss.append(float(l.split(' ')[1]))
    plt.title("Loss vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1, len(loss) + 1), loss)
    plt.yticks(np.arange(min(loss), max(loss)))
    plt.xticks(np.arange(0, len(loss) + 1, 40))
    plt.show()
    
if __name__ == '__main__':
    # data
#    trainloader, testloader = load_data()
#    trainloader, testloader = load_data_asym(noise_rate=0.3)
    noise_rate = 0.3
    trainloader, testloader = load_data_asym(noise_rate=noise_rate)
    
    # model
    model = ResNet34()
#    state_dict = torch.load("../models/Backbond.pth")
#    model.load_state_dict(state_dict, strict=False)
    print(model)
    print("noise rate: %.1f"%noise_rate)
        
    model = backbond_train(model, trainloader, testloader, num_epochs=70)
    model, label_distrib = PENCIL_train(model, trainloader, testloader, num_epochs=130)
    model = fine_tune(model, label_distrib, trainloader, testloader, num_epochs=120)
    draw_loss(log_file)
    print("noise rate: %.1f"%noise_rate)