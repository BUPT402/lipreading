from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Lipreading_resnet(object):
    def __init__(self, iterator):
        self.data_format = 'channels_first'

        self.iterator = iterator

        self.is_training = tf.placeholder_with_default(True, [], 'is_training')
        self.dropout_prob = tf.placeholder_with_default(1, [], 'dropout_prob')

        self.initializer = tf.contrib.layers.xavier_initializer()

    def build_input(self):
        self.image_seqs, self.label_seqs, self.label_length = self.iterator.get_next()


    def conv_3d(self):
        with tf.variable_scope('conv3d') as scope:
            conv = tf.layers.conv3d(self.image_seqs, 64, [3, 3, 3], [1, 2, 2], padding='val', use_bias=True,
                                    kernel_initializer=self.initializer, name='conv')
            bn = tf.layers.batch_normalization(conv, axis=-1, training=self.is_training, name='conv_bn')
            relu = tf.nn.relu(bn, name='relu')
            drop = tf.nn.dropout(relu, self.dropout_prob, name='drop1')
            pooling = tf.layers.max_pooling3d(drop, [1, 2, 2], [1, 2, 2], padding='valid', name='maxpooling1')
            tf.summary.histogram('conv_kernel', conv.kernel)
            tf.summary.histogram('conv_bias', conv.bias)
            tf.summary.histogram('conv_out', pooling)

    def resnet(self):
