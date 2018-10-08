import tensorflow as tf
import numpy as np
import os
import cv2



img=np.array([cv2.imread('./infer/16_left.jpeg').reshape(500,400,3)])


garb=np.array([img])



with tf.Session() as sess:
    new_saver=tf.train.import_meta_graph('C:/Health-Portal/trained_graph/retina_model.meta')
    new_saver.restore(sess,tf.train.latest_checkpoint('./trained_graph'))
    graph = tf.get_default_graph()
    dataset_init_op = graph.get_operation_by_name('inference_op')
    x_input=graph.get_tensor_by_name("p1:0")
    y=graph.get_tensor_by_name("p2:0")
    logit=graph.get_tensor_by_name("loss/pred:0")
    sess.run(dataset_init_op,feed_dict={x_input:img,y:garb})
    log=sess.run(logit)
    print(log)