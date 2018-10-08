import tensorflow as tf
import numpy as np
import cv2
import os
from tqdm import tqdm
from input_pipeline import input_func


dataset=input_func('./train.tfrecords')


#if you have memory allocation error go to the input_pipeline and reduce the batchsize

def data_pipe_line(data,checkpoint_path,i_data=None,epoch=150):

    place_X=tf.placeholder(tf.float32,[None,500,400,3],name='p1')
    place_Y=tf.placeholder(tf.int32,[None],name='p2')
    infer_data=tf.data.Dataset.from_tensor_slices((place_X,place_Y))
    infer_data=infer_data.batch(100)
    iterator=tf.data.Iterator.from_structure(data.output_types,data.output_shapes)
    next_image,next_label=iterator.get_next()
    Y=tf.one_hot(next_label,5)
    Y=tf.cast(Y,tf.float32)
    logits=model(next_image)
    train_iterator=iterator.make_initializer(data)
    inference_iterator_op=iterator.make_initializer(infer_data,name='inference_op')
    
    
    
    with tf.name_scope("loss"):
        loss=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=logits))
        #the learning rate is so low because the batch-size is very small and has a lot of noise
        optimizer=tf.train.AdamOptimizer(learning_rate=0.000005).minimize(loss)
        #getting the accuracy
        prediction=tf.argmax(logits,1,name='pred')
        equality=tf.equal(prediction,tf.argmax(Y,1))
        accuracy=tf.reduce_mean(tf.cast(equality,tf.float32))
        init_op=tf.global_variables_initializer()
        tf.summary.scalar("loss",loss)
        tf.summary.scalar("accuracy",accuracy)
        merged=tf.summary.merge_all()
        saver=tf.train.Saver()
        
   
     
    j=0
    with tf.Session() as sess:
        writer=tf.summary.FileWriter("./nn_logs",sess.graph)
        sess.run(init_op)
        for _ in range(epoch):
            sess.run(train_iterator)
            while True:
                try:
                    j=j+1 
                    summary = sess.run(merged)
                    _,acc,l=sess.run([optimizer,accuracy,loss]) 
                    if(j%5==0 or j==1):
                        print("iters: {}, loss: {:.10f}, training accuracy: {:.2f}%".format(j, l, acc * 100))
                    writer.add_summary(summary,j)
                except tf.errors.OutOfRangeError:
                    break
        saver.save(sess,checkpoint_path)
        









def model(in_data):
    
    in_data=tf.cast(in_data,tf.float32)
    c1 = tf.layers.conv2d(in_data,filters=128,kernel_size=8,strides=(2,2),padding='same',activation=tf.nn.relu)
    c2=tf.layers.max_pooling2d(c1,pool_size=4,strides=4)
    c3=tf.layers.conv2d(c2,filters=80,kernel_size=4,strides=(1,1),padding='valid',activation=tf.nn.relu)
    c4=tf.layers.conv2d(c3,filters=80,kernel_size=4,strides=(1,1),padding='same',activation=tf.nn.relu)
    c4=tf.layers.max_pooling2d(c4,pool_size=2,strides=2)
    c4=tf.layers.conv2d(c3,filters=80,kernel_size=4,strides=(1,1),padding='same',activation=tf.nn.relu)
    c4=tf.layers.max_pooling2d(c4,pool_size=2,strides=2)
    c4=tf.layers.conv2d(c3,filters=60,kernel_size=4,strides=(1,1),padding='same',activation=tf.nn.relu)
    c4=tf.layers.max_pooling2d(c4,pool_size=2,strides=2)
    c4=tf.layers.conv2d(c3,filters=60,kernel_size=4,strides=(1,1),padding='same',activation=tf.nn.relu)
    c4=tf.layers.conv2d(c4,filters=32,kernel_size=4,strides=(1,1),padding='same',activation=tf.nn.relu)

    
    
    fc0 = tf.contrib.layers.flatten(c4)
    fc1 = tf.layers.dense(fc0, 512,activation=tf.nn.relu)
    fc3 = tf.layers.dense(fc1, 128,activation=tf.nn.relu)
    fc3 = tf.layers.dense(fc1, 64,activation=tf.nn.relu)
    fc3 = tf.layers.dense(fc3, 32,activation=tf.nn.relu)
    fc3 = tf.layers.dense(fc3, 5)
    
    
    return fc3



data_pipe_line(dataset,"./trained_graph/retina_model")
