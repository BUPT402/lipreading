from __future__ import absolute_import,division,print_function

import os 
import tensorflow as tf
import numpy as np
import glob
from collections import Counter
from input import train_batch_generator



frame_batch, label_batch, label_length = train_batch_generator(
    '/home/zyq/dataset/test1000', 30, 100, 4)
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.train.start_queue_runners(sess=sess)
for i in range(10):
    print('d')
    print(frame_batch, label_batch, label_length)
    _frame_batch, _label_batch, length = sess.run([frame_batch, label_batch, label_length])
    print('** batch %d' % i)
    print(_label_batch)