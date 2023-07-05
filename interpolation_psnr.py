# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 21:18:09 2020

@author: linyi
"""

#FOR TESTING INDIVIDUAL INTERPOLATED IMAGES

import cv2

from torch import nn

from torchvision.transforms import ToTensor
import numpy as np
from math import log10
from skimage.metrics import structural_similarity as ssim

import argparse
parser = argparse.ArgumentParser(description="Interpolation PSNR")
parser.add_argument('--image-path', '-i', required=True, help="GT image path" )
parser.add_argument('--interpolation',  '-n',required=True, help="Interpolation to be used, \"BICUBIC\" or \"BILINEAR\" of \"NEAREST_NEIGHBOR\"")
parser.add_argument('--scale',  '-s',default=3, help="Scale Factor")
args = parser.parse_args()

if __name__ == "__main__":

    image=cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:,:,0]
    image = cv2.normalize(image.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
    interpolation_format = cv2.INTER_CUBIC
    if(args.interpolation.upper() == "BICUBIC"):
        interpolation_format = cv2.INTER_CUBIC
    elif(args.interpolation.upper() == "BILINEAR"):
        interpolation_format = cv2.INTER_LINEAR                  
    elif(args.interpolation.upper() == "NEAREST_NEIGHBOR"):
        interpolation_format = cv2.INTER_NEAREST         
    #cv2.INTER_CUBIC
    #cv2.INTER_LINEAR
    #cv2.INTER_NEAREST
    #print(image.shape)
    if type(args.scale)==int:
        scale=args.scale
    else:
        scale=int(args.scale)

    interpolated=image.copy()
    interpolated=cv2.resize(interpolated,(image.shape[1]//scale,image.shape[0]//scale))
    interpolated=cv2.resize(interpolated,(image.shape[1],image.shape[0]),interpolation=interpolation_format)

    ssim_value=ssim(image, interpolated, datarange=interpolated.max()-interpolated.min())
    
    
    interpolated=ToTensor()(interpolated)
    image=ToTensor()(image)
    
    criterion = nn.MSELoss()
    loss=criterion(interpolated,image)
    psnr = 20*log10(1) - 10*log10(loss.data)
    
    print("PSNR VALUE:"+str(psnr))
    print("SSIM VALUE:"+str(ssim_value))
    

