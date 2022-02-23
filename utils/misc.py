import cv2
import os
import numpy as np
from tqdm import tqdm

def load_data(no_of_images, dir_source = '/content/drive/MyDrive/Datasets/bone_supression/xray-bone-shadow-supression/augmented/augmented/source', dir_target = '/content/drive/MyDrive/Datasets/bone_supression/xray-bone-shadow-supression/augmented/augmented/target'):
    
    img_size = (128,128)
    imgs_source = []
    imgs_target = []
       
    i = 0
    for _, _, filenames in tqdm(os.walk(dir_target), total=no_of_images):
        for filename in filenames:
            i = i+1
            if(i > no_of_images):
                break
            img_source = cv2.imread(os.path.join(dir_source,filename),cv2.IMREAD_GRAYSCALE)
            img_target = cv2.imread(os.path.join(dir_target, filename),cv2.IMREAD_GRAYSCALE)
            # resizing images
            img_source = cv2.resize(img_source,img_size)
            img_target = cv2.resize(img_target,img_size)
            # normalizing images
            img_source = np.array(img_source)/255
            img_target = np.array(img_target)/255
            
            imgs_source.append(img_source)
            imgs_target.append(img_target)
    return imgs_source, imgs_target