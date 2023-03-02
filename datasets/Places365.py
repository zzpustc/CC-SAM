#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
import csv
import numpy as np
from PIL import Image


# Dataset
class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None, mode = 'train'):
        self.img_path = []
        self.labels = []
        self.real_label = []
        self.transform = transform
        if mode != 'semi':
            with open(txt) as f:
                for line in f:
                    self.img_path.append(os.path.join(root, line.split()[0]))
                    self.labels.append(int(line.split()[1]))
        else:
            with open(txt) as f:
                reader = csv.reader(f)
                for row in reader:
                    self.img_path.append(os.path.join(root, row[0]))
                    a = row[1][1:-2].split()
                    a = np.array(list(map(float, a)))
                    self.labels.append(a)
                    self.real_label.append(int(row[2]))

        self.mode = mode

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        if self.mode == 'semi':
            y_real = self.real_label[index]
            return path, sample, label, y_real

        return sample, label, path