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
        'image1':tf.FixedLenFeature([],tf.string),
        'image2':tf.FixedLenFeature([],tf.string),
        'label':tf.FixedLenFeature([],tf.int64)
    }
    parsed_example = tf.parse_single_example(serialized=serialized,features=features)

    image_raw1=parsed_example['image1']
    image_raw2=parsed_example["image2"]

    image1=tf.image.decode_jpeg(image_raw1)
    image1=tf.cast(image1,dtype=tf.float32)
    image1=tf.reshape(image1,[500,400,3])
    
    image2=tf.image.decode_jpeg(image_raw2)
    image2=tf.cast(image2,dtype=tf.float32)
    image2=tf.reshape(image2,[500,400,3])
    
    combined_image=tf.concat([image1,image2],axis=-1)
    label_raw=parsed_example['label']
    
    labels=tf.cast(label_raw,tf.int32)
    
    
    return combined_image,labels


#this is where we do some of the hyper-parameter tunings 
def input_func(filenames,batch_size=50):
    dataset=tf.data.TFRecordDataset(filenames=filenames)
    dataset=dataset.map(parse,num_parallel_calls=8)
    dataset=dataset.batch(batch_size)
    dataset=dataset.shuffle(20)
    dataset=dataset.prefetch(8)
    return dataset
