# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 21:38:52 2020

@author: linyi
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self,csize=1):
        super(SRCNN,self).__init__()
        self.lrelu=F.leaky_relu
        self.conv1=nn.Conv2d(csize,64,kernel_size=9,padding=0)
        self.conv2=nn.Conv2d(64,32,kernel_size=1,padding=0)
        self.conv3=nn.Conv2d(32,csize,kernel_size=5,padding=0)
        
        
        
    def forward(self,x):
        x=self.lrelu(self.conv1(x))
        x=self.lrelu(self.conv2(x))
        x=self.conv3(x)
        
        return x