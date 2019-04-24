#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:02:53 2019

@author: yuming
"""
import sys
sys.path.append('/home/yuming/projects/learn-with-noisy-labels/nets')

import os
import torch
import numpy as np
import pandas as pd
from squeezenet import squeezenet1_1
from PIL import Image

def test_crop(model, test_file, output_path):
    model.eval()
    confusion_matrix = np.zeros((3, 3), dtype=int)
    label_dic = {'repeat': 0, 'fuzzy': 1, 'clear': 2}
    clear = ['normal', 'second_tiled', 'tiled', 'background']
    counter = 0
    with open(test_file) as f:
        mean = torch.tensor([123, 117, 104], dtype=torch.float32)
        files = f.readlines()
        total = len(files)
        for l in files:
            full_path = l.split(' ')[0]
            img = torch.tensor(np.expand_dims(np.array(Image.open(
                    full_path)), 0).transpose((0, 3, 1, 2)), dtype=torch.float32)            
            img = img.sub_(mean[:, None, None]).div_(torch.tensor(256, dtype=torch.float32))
            label = full_path.split('/')[-2]
            if label in clear:
                label = 'clear'
            output = model(img)
            _, predicted = torch.max(output.data, 1)
            confusion_matrix[label_dic[label]][int(predicted)] += 1
            if counter % (total // 50) == 0 or counter == total:
                sys.stdout.write('\rcomplete {:.2f}%'.format(100. * counter / total))
                sys.stdout.flush()
            counter += 1
    print('\n')
    print(confusion_matrix)    
    sum_recall = np.sum(confusion_matrix, 1)
    sum_precision = np.sum(confusion_matrix, 0)
    truth_positive = np.diag(confusion_matrix)  # TP for each class corresponding to its index
    recall = 1. * truth_positive / sum_recall
    precision = 1. * truth_positive / sum_precision
    precision_ = np.hstack((precision, np.array([np.nan])))  # precision with one more empty element
    # confusion_matrix_with_recall
    confusion_matrix_ = np.hstack((confusion_matrix, recall.reshape((recall.shape[0], 1))))
    # confusion_matrix_with_recall_and_precision
    _confusion_matrix_ = np.vstack((confusion_matrix_, precision_.reshape(1, precision_.shape[0])))
    cols = ['repeat', 'fuzzy', 'clear']
    rows = ['repeat', 'fuzzy', 'clear']
    cols.append('Recall')
    rows.append('Precision')
    df_confusion_matrix = pd.DataFrame(_confusion_matrix_, index=rows, columns=cols)
    df_confusion_matrix.to_csv(os.path.join(output_path, "crop.csv"), encoding='utf-8')

def test_full(model, test_file, output_path):
    model.eval()
    confusion_matrix = np.zeros((3, 3), dtype=int)
    label_dic = {'repeat': 0, 'fuzzy': 1, 'clear': 2}
    clear = ['normal', 'second_tilted', 'tilted', 'background']
    counter = 0
    with open(test_file) as f:
        mean = torch.tensor([123, 117, 104], dtype=torch.float32)
        files = f.readlines()
        total = len(files)
        for l in files:
            l = l.strip()
            full_path = l.split(' ')[0]
            try:
                img_source = Image.open(full_path)
                img = torch.tensor(np.expand_dims(np.array(img_source
                            ), 0).transpose((0, 3, 1, 2)), dtype=torch.float32)
                img = img.sub_(mean[:, None, None]).div_(torch.tensor(256, dtype=torch.float32))
            except:
                print('\n' + full_path)
                continue
            label = full_path.split('/')[-2]
            if label in clear:
                label = 'clear'
            
            # top left
            try:
                img_crop = img[:, :, img.shape[2]*2//5 - 112:img.shape[2]*2//5 + 112, img.shape[3]*2//5 - 112:img.shape[3]*2//5 + 112]
                output = model(img_crop)
            except:
                img_crop = img[:, :, 0:224, 0:224]
                output = model(img_crop)
            
            # top right
            try:
                img_crop = img[:, :, img.shape[2]*3//5 - 112:img.shape[2]*3//5 + 112, img.shape[3]*2//5 - 112:img.shape[3]*2//5 + 112]
                output += model(img_crop)
            except:
                img_crop = img[:, :, img.shape[2] - 224:img.shape[2], 0:224]
                output += model(img_crop)
                
            # bottom left
            try:
                img_crop = img[:, :, img.shape[2]*2//5 - 112:img.shape[2]*2//5 + 112, img.shape[3]*3//5 - 112:img.shape[3]*3//5 + 112]
                output += model(img_crop)
            except:
                img_crop = img[:, :, 0:224, img.shape[3] - 224:img.shape[3]]
                output += model(img_crop)
                
            # bottom right
            try:
                img_crop = img[:, :, img.shape[2]*3//5 - 112:img.shape[2]*3//5 + 112, img.shape[3]*3//5 - 112:img.shape[3]*3//5 + 112]
                output += model(img_crop)
            except:
                img_crop = img[:, :, img.shape[2] - 224:img.shape[2], img.shape[3] - 224:img.shape[3]]
                output += model(img_crop)
            
            img_crop = img[:, :, img.shape[2]//2 - 112:img.shape[2]//2 + 112, img.shape[3]//2 - 112:img.shape[3]//2 + 112]
            output += model(img_crop)
            
            _, predicted = torch.max(output.data, 1)
#            print(int(predicted))
            confusion_matrix[label_dic[label]][int(predicted)] += 1
            if counter % (total // 50) == 0 or counter == total:
                sys.stdout.write('\rcomplete {:.2f}%'.format(100. * counter / total))
                sys.stdout.flush()
            counter += 1
    print('\n')
    print(confusion_matrix)    
    sum_recall = np.sum(confusion_matrix, 1)
    sum_precision = np.sum(confusion_matrix, 0)
    truth_positive = np.diag(confusion_matrix)  # TP for each class corresponding to its index
    recall = 1. * truth_positive / sum_recall
    precision = 1. * truth_positive / sum_precision
    precision_ = np.hstack((precision, np.array([np.nan])))  # precision with one more empty element
    # confusion_matrix_with_recall
    confusion_matrix_ = np.hstack((confusion_matrix, recall.reshape((recall.shape[0], 1))))
    # confusion_matrix_with_recall_and_precision
    _confusion_matrix_ = np.vstack((confusion_matrix_, precision_.reshape(1, precision_.shape[0])))
    cols = ['repeat', 'fuzzy', 'clear']
    rows = ['repeat', 'fuzzy', 'clear']
    cols.append('Recall')
    rows.append('Precision')
    df_confusion_matrix = pd.DataFrame(_confusion_matrix_, index=rows, columns=cols)
    df_confusion_matrix.to_csv(os.path.join(output_path, "full.csv"), encoding='utf-8')

if __name__ == '__main__':
    model = squeezenet1_1(num_classes=4)
    state_dict = torch.load('/home/yuming/projects/learn-with-noisy-labels/models/finetune_0418.pth')
    model.load_state_dict(state_dict)
    test_crop_file = '/data/yuming/image_qa/data/fuzzy_test_crop.txt'
    test_full_file = '/data/yuming/image_qa/data/files.txt'
    output_path = '/home/yuming/projects/learn-with-noisy-labels/results'
    test_crop(model, test_crop_file, output_path)
    test_full(model, test_full_file, output_path)