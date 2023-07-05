# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:25:16 2020

@author: linyi
"""

#FOR TESTING INDIVIDUAL PREDICTED IMAGES

import cv2

from torch import nn

from torchvision.transforms import ToTensor
import numpy as np
from math import log10

from skimage.metrics import structural_similarity as ssim


import argparse
parser = argparse.ArgumentParser(description="Image-wise PSNR")
parser.add_argument('--input-path', '-i', required=True, help="GT image path" )
parser.add_argument('--output-path',  '-o',required=True, help="predicted image path")
args = parser.parse_args()

if __name__ == "__main__":

    image=cv2.imread(args.input_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:,:,0]
    image = cv2.normalize(image.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
    output=cv2.imread(args.output_path)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2YCrCb)[:,:,0]
    output = cv2.normalize(output.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
    
    ssim_value=ssim(image, output, datarange=output.max()-output.min())

    output=ToTensor()(output)
    image=ToTensor()(image)
    
    criterion = nn.MSELoss()
    loss=criterion(output,image)
    psnr = 20*log10(1) - 10*log10(loss.data)
    print("PSNR VALUE:"+str(psnr))
    print("SSIM VALUE:"+str(ssim_value))
    

