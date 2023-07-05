# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 20:16:49 2020

@author: linyi
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import cv2
import os
import numpy as np
from model import SRCNN
import time



class SRDataset(Dataset):
    def __init__(self, input_npy, label_npy):
        inputs = np.load(input_npy)
        labels = np.load(label_npy)

        #converting from N x H x W x C to N x C x H x W so it can be used in convolutional layers
        inputs = inputs.transpose((0, 3, 1, 2))
        labels = labels.transpose((0, 3, 1, 2))

        self.inputs = torch.from_numpy(inputs).double()
        self.labels = torch.from_numpy(labels).double()
    
    def __getitem__(self, idx):
        return(self.inputs[idx], self.labels[idx])
    
    def __len__(self):
        return len(self.inputs)

