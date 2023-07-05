import numpy as np
import cv2
import os
import argparse
from utils import center_crop
parser = argparse.ArgumentParser(description="Create Image patches from folder and save as npy")
parser.add_argument('--image-folder', '-i', required=True, help="Folder to get images from" )
parser.add_argument('--input-npy', '-o', required=True, help="Path of output npy file containing input patches")
parser.add_argument('--label-npy', '-O', required=True, help="Path of output npy file containing input labels")
parser.add_argument('--patch-size', '-p', type=int, default=33, help="Side length of square patch")
parser.add_argument('--stride', '-s', type=int, default=14, help="Side length of square patch")
parser.add_argument('--label-size', '-l', type=int, default=21, help="label dimensions (look at network structure to set)")
parser.add_argument('--scale-factor', '-f', type=int, default=3, help="scale factor for data")
parser.add_argument('--augment', '-a',  help="If we want data augmentation", action="store_true")
parser.add_argument('--color-mode',  default="Y", help="color mode \"Y\" or \"RGB\"")
parser.add_argument('--interpolation',  default="BICUBIC", help="Interpolation to be used, \"BICUBIC\" or \"BILINEAR\" of \"NEAREST_NEIGHBOR\"")


args = parser.parse_args()



#returns an array of image augmentations
def do_augmentations(img):
    flipUd = cv2.flip(patch, 0)
    flipLr = cv2.flip(patch, 1)
    flipUdLr = cv2.flip(patch, -1)
    rot270 = cv2.rotate(patch, cv2.cv2.ROTATE_90_CLOCKWISE)
    rot90 = cv2.rotate(patch, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    return [flipLr, flipUd, flipUdLr, rot270, rot90]



#splits images in a folder into training patches and labels and then saves the np arrays
def make_patches(image_folder, 
    input_npy, 
    label_npy, 
    patch_size=33, 
    stride=14, 
    label_size=21, 
    scale_factor=3, 
    augment=False, 
    color_mode="Y", 
    interpolation="BICUBIC"):
    files = os.listdir(image_folder)

    inputs, labels = [], []
    patches = []
    for imgName in files:
        img = cv2.imread(image_folder + "/"+ imgName)
        height, width, channels = img.shape

        for y in range(0, height-patch_size +1, stride):
            for x in range(0, width-patch_size+1, stride):

                #extract patch
                patch = img[y: y+patch_size, x: x+patch_size, :]

                #extract Y channel if we choose, otherwise keep rgb
                if(color_mode.upper() == "Y"):
                    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)[:,:,0]

                #normalize to float range 0, 1 (this was done in original matlab code) ref https://stackoverflow.com/questions/29100722/equivalent-im2double-function-in-opencv-python/29104511
                patch = cv2.normalize(patch.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


                #picking interpolation
                interpolation_format = cv2.INTER_CUBIC
                if(interpolation.upper() == "BICUBIC"):
                    interpolation_format = cv2.INTER_CUBIC
                elif(interpolation.upper() == "BILINEAR"):
                      interpolation_format = cv2.INTER_LINEAR                  
                elif(interpolation.upper() == "NEAREST_NEIGHBOR"):
                      interpolation_format = cv2.INTER_NEAREST     

                #Get input by downscaling and upscaling patch
                inputPatch = cv2.resize(patch, (patch_size// scale_factor, patch_size // scale_factor), interpolation=interpolation_format)
                inputPatch = cv2.resize(inputPatch, (patch_size, patch_size), interpolation=interpolation_format)

                labelPatch = center_crop(patch, label_size, label_size)

                inputs.append(inputPatch)
                labels.append(labelPatch)

                if(augment):
                    inputs.extend(do_augmentations(inputPatch))
                    labels.extend(do_augmentations(labelPatch))
                patches.append(patch)

    length = len(inputs)
    inputs = np.array(inputs, dtype=np.float_).reshape((length, patch_size, patch_size, -1))
    labels = np.array(labels, dtype=np.float_).reshape((length, label_size, label_size, -1))



    np.save(input_npy, inputs)
    np.save(label_npy, labels)

    print("Patches created")
    return (inputs, labels)


if __name__ == "__main__":
    print("Creating Patches")
    (inputs, labels) = make_patches(args.image_folder, 
        args.input_npy, 
        args.label_npy, 
        args.patch_size,
        args.stride,  
        args.label_size, 
        args.scale_factor, 
        args.augment, 
        args.color_mode, 
        args.interpolation)
    print("Saved Input with shape {} and Label with shape {} ".format(inputs.shape, labels.shape))