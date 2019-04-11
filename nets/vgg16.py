#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 14:18:19 2018

@author: zs
"""
import torch.nn as nn
import torch.nn.functional as F

class vgg16(nn.Module):
#    def __init__(self):
#        super(vgg16, self).__init__()
#        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
#        self.bn1 = nn.BatchNorm2d(64)
#        self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
#        self.bn2 = nn.BatchNorm2d(64)
#        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
#        self.bn3 = nn.BatchNorm2d(128)
#        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
#        self.bn4 = nn.BatchNorm2d(128)
#        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
#        self.bn5 = nn.BatchNorm2d(256)
#        self.conv6 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
#        self.bn6 = nn.BatchNorm2d(256)
#        self.conv7 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
#        self.bn7 = nn.BatchNorm2d(256)
#        self.conv8 = nn.Conv2d(256, 512, 3, padding=1, bias=False)
#        self.bn8 = nn.BatchNorm2d(512)
#        self.conv9 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
#        self.bn9 = nn.BatchNorm2d(512)
#        self.conv10 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
#        self.bn10 = nn.BatchNorm2d(512)
#        self.conv11 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
#        self.bn11 = nn.BatchNorm2d(512)
#        self.conv12 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
#        self.bn12 = nn.BatchNorm2d(512)
#        self.conv13 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
#        self.bn13 = nn.BatchNorm2d(512)
#        self.pool = nn.MaxPool2d(2, 2)
#        self.fc = nn.Linear(512, 100)

    def __init__(self):
        super(vgg16, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(512, 3)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.pool(x)
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = self.pool(x)
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.relu(self.bn13(self.conv13(x)))
        x = self.pool(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x
    
class vgg16_pruned(nn.Module):
#    def __init__(self):
#        super(vgg16_pruned, self).__init__()
#        self.base = 64
#        self.conv1 = nn.Conv2d(3, self.base, 3, padding=1, bias=False)
#        self.bn1 = nn.BatchNorm2d(self.base)
#        self.conv2 = nn.Conv2d(self.base, self.base, 3, padding=1, bias=False)
#        self.bn2 = nn.BatchNorm2d(self.base)
#        self.conv3 = nn.Conv2d(self.base, self.base*2, 3, padding=1, bias=False)
#        self.bn3 = nn.BatchNorm2d(self.base*2)
#        self.conv4 = nn.Conv2d(self.base*2, self.base*2, 3, padding=1, bias=False)
#        self.bn4 = nn.BatchNorm2d(self.base*2)
#        self.conv5 = nn.Conv2d(self.base*2, self.base*4, 3, padding=1, bias=False)
#        self.bn5 = nn.BatchNorm2d(self.base*4)
#        self.conv6 = nn.Conv2d(self.base*4, self.base*4, 3, padding=1, bias=False)
#        self.bn6 = nn.BatchNorm2d(self.base*4)
#        self.conv7 = nn.Conv2d(self.base*4, self.base*4, 3, padding=1, bias=False)
#        self.bn7 = nn.BatchNorm2d(self.base*4)
#        self.conv8 = nn.Conv2d(self.base*4, self.base*8, 3, padding=1, bias=False)
#        self.bn8 = nn.BatchNorm2d(self.base*8)
#        self.conv9 = nn.Conv2d(self.base*8, self.base*8, 3, padding=1, bias=False)
#        self.bn9 = nn.BatchNorm2d(self.base*8)
#        self.conv10 = nn.Conv2d(self.base*8, self.base*8, 3, padding=1, bias=False)
#        self.bn10 = nn.BatchNorm2d(self.base*8)
#        self.conv11 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
#        self.bn11 = nn.BatchNorm2d(512)
#        self.conv12 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
#        self.bn12 = nn.BatchNorm2d(512)
#        self.conv13 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
#        self.bn13 = nn.BatchNorm2d(512)
#        self.pool = nn.MaxPool2d(2, 2)
#        self.fc = nn.Linear(512, 100)
#        self.fores = []
#        self.feats = []
        
    def __init__(self):
        super(vgg16_pruned, self).__init__()
        self.base = 64
        self.conv1 = nn.Conv2d(3, self.base, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.base)
        self.conv2 = nn.Conv2d(self.base, self.base, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.base)
        self.conv3 = nn.Conv2d(self.base, self.base*2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(self.base*2)
        self.conv4 = nn.Conv2d(self.base*2, self.base*2, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(self.base*2)
        self.conv5 = nn.Conv2d(self.base*2, self.base*4, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(self.base*4)
        self.conv6 = nn.Conv2d(self.base*4, self.base*4, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(self.base*4)
        self.conv7 = nn.Conv2d(self.base*4, self.base*4, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(self.base*4)
        self.conv8 = nn.Conv2d(self.base*4, self.base*8, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(self.base*8)
        self.conv9 = nn.Conv2d(self.base*8, self.base*8, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(self.base*8)
        self.conv10 = nn.Conv2d(self.base*8, self.base*8, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(self.base*8)
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(512, 100)
        self.fores = []
        self.feats = []
        
    def forward(self, x):
        self.fores.append(x)
        x = self.conv1(x)
        self.feats.append(x)
        x = F.relu(self.bn1(x))
        
        self.fores.append(x)
        x = self.conv2(x)
        self.feats.append(x)
        x = F.relu(self.bn2(x))
        x = self.pool(x)
        
        self.fores.append(x)
        x = self.conv3(x)
        self.feats.append(x)
        x = F.relu(self.bn3(x))
        
        self.fores.append(x)
        x = self.conv4(x)
        self.feats.append(x)
        x = F.relu(self.bn4(x))
        x = self.pool(x)
        
        self.fores.append(x)
        x = self.conv5(x)
        self.feats.append(x)
        x = F.relu(self.bn5(x))
        
        self.fores.append(x)
        x = self.conv6(x)
        self.feats.append(x)
        x = F.relu(self.bn6(x))
        
        self.fores.append(x)
        x = self.conv7(x)
        self.feats.append(x)
        x = F.relu(self.bn7(x))
        x = self.pool(x)
        
        self.fores.append(x)
        x = self.conv8(x)
        self.feats.append(x)
        x = F.relu(self.bn8(x))
        
        self.fores.append(x)
        x = self.conv9(x)
        self.feats.append(x)
        x = F.relu(self.bn9(x))
        
        self.fores.append(x)
        x = self.conv10(x)
        self.feats.append(x)
        x = F.relu(self.bn10(x))
        x = self.pool(x)

        self.fores.append(x)
        x = self.conv11(x)
        self.feats.append(x)        
        x = F.relu(self.bn11(x))

        self.fores.append(x)
        x = self.conv12(x)
        self.feats.append(x)        
        x = F.relu(self.bn12(x))
        
        self.fores.append(x)
        x = self.conv13(x)
        self.feats.append(x)        
        x = F.relu(self.bn13(x))
        
        x = self.pool(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x
    
    def reset(self):
        self.fores = []
        self.feats = []