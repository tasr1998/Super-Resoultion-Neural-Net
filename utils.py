import torch
from torch import nn
from torch.utils.data import DataLoader

from math import log10
import numpy as np
from model import SRCNN
from data import SRDataset
from predict import predict
import os
from torch import nn
from torchvision.transforms import ToTensor
import cv2

#Takes in a Y channel of an image in YCbCr format, means len(img.shape) = len(gt.shape) = 2
def psnr(gt, img, maximum):
    gt=ToTensor()(gt)
    img=ToTensor()(img)
    criterion = nn.MSELoss()
    loss=criterion(gt,img)
    psnr = 20*log10(maximum) - 10*log10(loss.data)
    return psnr


#Takes in a bgr img normalized to 0,1
def getNormalizedY(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[0]
    img = cv2.normalize(img.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return img

def center_crop(img, new_width, new_height):        

    width = img.shape[1]
    height = img.shape[0]


    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    center_cropped_img = img[top:bottom, left:right]

    return center_cropped_img