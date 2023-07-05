# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 19:05:59 2020

@author: linyi
"""
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
from utils import center_crop
from utils import psnr
from utils import getNormalizedY


import argparse
parser = argparse.ArgumentParser(description="PSNR test")
parser.add_argument('--model-path',  '-m',required=True, help="path of model")
parser.add_argument('--test-folder', '-t', required=True, help="test folder" )
parser.add_argument('--output-path',  '-o',required=True, help="test output folder")
parser.add_argument('--interpolation',  '-i',required=True, help="BILINEAR,BICUBIC,NEAREST_NEIGHBOR")
parser.add_argument('--scale',  '-s',required=True, help="scale factor")

args = parser.parse_args()
print(args)



#runs model on test folder and prints  psnr values for each
def test(model_path, test_path, output_path, sf, interpolation, save=False):
    imgNames = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]
    print("Name\t\t\tBICUBIC\t\t\tPREDICTION")
    amt = 0
    totalPsnrBic = 0
    totalPsnrPred = 0
    for imgName in imgNames:
        baseName = os.path.splitext(imgName)[0] + "_x{}".format(sf)
        img = cv2.imread(os.path.join(test_path, imgName))
        img_out_path = os.path.join(output_path, baseName)
        
        if(save and not os.path.exists(img_out_path)):
            os.mkdir(img_out_path)

        #cropping img to be divisible by scale factor
        height, width, c = img.shape
        height = height - (height % sf)
        width = width - (width % sf)
        img =  center_crop(img, width, height)
        
        if(save):
            cv2.imwrite(os.path.join(img_out_path, "gt.png"), img)

        #reducing and rescaling
        smollImg = cv2.resize(img, (width//sf, height//sf), interpolation)

        #bicubic
        imgBic = cv2.resize(smollImg, (width,height), interpolation)
        if(save):
            cv2.imwrite(os.path.join(img_out_path, "bc.png"), imgBic)

        #calculating psnr for bicubic
        imgBicY = getNormalizedY(imgBic)
        imgY = getNormalizedY(img)
        bicPSNR = psnr(imgY, imgBicY, 1.0)
        totalPsnrBic += bicPSNR

        #predicting
        imgPred = predict(sf, model_path, smollImg, interpolation)
        
        if(save):
            cv2.imwrite(os.path.join(img_out_path, "cnn.png"), imgPred)

        #pred psnr
        imgPredY = getNormalizedY(imgPred)
        predPSNR = psnr(imgY, imgPredY,1.0)
        totalPsnrPred += predPSNR

        amt += 1
        print("{}\t\t\t{:.4f}\t\t\t{:.4f}".format(baseName, bicPSNR, predPSNR))
    print("AVG Pred PSNR: {:.4f}\nAVG Bicbuic PSNR: {:.4f}".format(totalPsnrPred/amt, totalPsnrBic/amt))




if __name__ == "__main__":
    interpolation_format = cv2.INTER_CUBIC
    if(args.interpolation.upper() == "BICUBIC"):
        interpolation_format = cv2.INTER_CUBIC
    elif(args.interpolation.upper() == "BILINEAR"):
        interpolation_format = cv2.INTER_LINEAR                  
    elif(args.interpolation.upper() == "NEAREST_NEIGHBOR"):
        print("NN")
        interpolation_format = cv2.INTER_NEAREST         
    if type(args.scale)==int:
        scale=args.scale
    else:
        scale=int(args.scale)
    test(args.model_path, args.test_folder, args.output_path, scale,interpolation_format, save=True)


