
import torch
import cv2
from model import SRCNN
import numpy as np



#takes in a BGR formatted image attained from cv2 imread
def predict(scale_factor, model_path, img,interpolation,padding=6):
    

    height, width, channels = img.shape

    img = cv2.resize(img, (img.shape[1]*scale_factor, img.shape[0]*scale_factor), interpolation=interpolation)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    imgY = img[:,:,0]
    
    imgY = cv2.normalize(imgY.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)

    imgY_t = torch.from_numpy(imgY)

    imgY_t = imgY_t.view((1,1,height*scale_factor,width*scale_factor))
    imgY_t = imgY_t.to("cuda")


    model = SRCNN()
    model.load_state_dict(torch.load(model_path))
    model = model.double()
    model = model.to("cuda")

    with torch.no_grad():
        pred = model(imgY_t)
        pred = pred.view((pred.size()[2], -1))

    pred = pred.to("cpu")
    pred = cv2.normalize(pred.numpy().astype("float"), None, 0.0, 255.0, cv2.NORM_MINMAX)
    img[padding:height*scale_factor-padding, padding:width*scale_factor-padding, 0] = pred
    img = cv2.cvtColor(img,cv2.COLOR_YCrCb2BGR)
    return img

def predictSave(scale_factor, model_path, img_input_path, img_output_path,interpolation,padding=6):
    img = cv2.imread(img_input_path)
    out = predict(scale_factor, model_path, img,interpolation)
    cv2.imwrite(img_output_path, out)
    return out

#cv2.INTER_CUBIC
#cv2.INTER_LINEAR
#cv2.INTER_NEAREST

