import numpy as np
from tqdm import tqdm
from natsort import natsorted,ns
import os
import csv
import cv2
import tensorflow as tf



#You change the file location according to your storage
TRAIN_DIR='C:/Health-Portal/train-01-processed'
LABEL="C:/Health-Portal/trainLabels.csv"
with open(LABEL,'r') as csvfile1:
    csvreader1=csv.reader(csvfile1)
    csvreader1=list(csvreader1)
    csvreader1.pop(0)
    csvreader1=natsorted(csvreader1 ,key=lambda x:x[0])
    length=len(csvreader1)



#This is where we specify the datatype and data structure of the data that is to be converted to tfRecords
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



#this is the function that writes the protocol buffer aka tfrecords :)
def create_train_data(train_dir,csv):
    training_data=[]
    temp=os.listdir(train_dir)
    dir_list=natsorted(temp,key=lambda x:x[0])
    count1=0
    val_filename = 'train.tfrecords'  # address to save the TFRecords file
    writer = tf.python_io.TFRecordWriter(val_filename)
    for img in tqdm(dir_list):
        for i in range(100000):
            r=img.split('.')[0]
            if r==csv[count1][0]:
                path=os.path.join(train_dir,img)
                with open(path, "rb") as image_file:
                    encoded_string = image_file.read()


                label=int(csv[count1][1])
                feature = {
                    'label': _int64_feature(label),
                    'image': _bytes_feature(encoded_string)
                }

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())


                count1=count1+1
                if(count1>=length):
                    count1=0
                break
            count1=count1+1
            if(count1>=length):
                count1=0
    writer.close()





create_train_data(TRAIN_DIR,csvreader1)   


        