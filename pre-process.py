import tensorflow as tf
import numpy as np
import cv2
import os
from tqdm import tqdm

TRAIN_DIR='C:/Health-Portal/train-01'
TARGET_DIR='C:/Health-Portal/train-01-processed'

#reducing the size


def preprocess_img(dir_path,target_path):
    dir_list=path_list_maker(dir_path)
    for i in tqdm(dir_list):
        img=cv2.imread(i[1])
        img_resized=cv2.resize(img,(500,400))
        cv2.imwrite(TARGET_DIR+'/'+i[0] + ".jpeg", img_resized)




preprocess_img(TRAIN_DIR,TARGET_DIR)