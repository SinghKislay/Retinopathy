import numpy as np
from tqdm import tqdm
from natsort import natsorted,ns
import os
import csv
import cv2
import tensorflow as tf




#this is where we write what we want to do with each extracted image,label
def parse(serialized):
    features={
        'image':tf.FixedLenFeature([],tf.string),
        'label':tf.FixedLenFeature([],tf.int64)
    }
    parsed_example = tf.parse_single_example(serialized=serialized,features=features)

    image_raw=parsed_example['image']
    image=tf.image.decode_jpeg(image_raw)
    image=tf.cast(image,dtype=tf.float32)
    image=tf.reshape(image,[500,400,3])
    
    
    label_raw=parsed_example['label']
    
    labels=tf.cast(label_raw,tf.int32)
    
    
    return image,labels


#this is where we do some of the hyper-parameter tunings 
def input_func(filenames,batch_size=60):
    dataset=tf.data.TFRecordDataset(filenames=filenames)
    dataset=dataset.map(parse,num_parallel_calls=32)
    dataset=dataset.batch(batch_size)
    dataset=dataset.shuffle(40)
    dataset=dataset.prefetch(32)
    return dataset
