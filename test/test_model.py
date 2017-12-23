from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import argparse

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of images to process in a batch.')

parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                    help='Path to the CIFAR-10 data directory.')

parser.add_argument('--use_fp16', type=bool, default=False,
                    help='Train the model using fp16.')

FLAGS = parser.parse_args()


#Global Dataset Parameters
IMAGE_SIZES = 149 * 90 * 3
NUM_CLASSES = 773
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 5000

#Global Train Parameters
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

sess = tf.InteractiveSession()

def inference(videos):
    with tf.name_scope('input'):
        video = tf.placeholder(tf.float32, [None,250, 140, 90, 3])
        video_mask = tf.placeholder(tf.int32, [None, 77])
        caption = tf.placeholder(tf.int32, [None, 25, 773])
        caption_mask = tf.placeholder(tf.int32, [None, 25])
        keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope('conv1'):
        conv1 = tf.layers.conv3d(video, 32, [3, 5, 5], [1, 2, 2], padding='same',
                                 use_bias=False, kernel_initializer=tf.truncated_normal_initializer, name='conv1')
        batch1 = tf.layers.batch_normalization(conv1, axis=-1)
        relu1 = tf.nn.relu(batch1)
        drop1 = tf.nn.dropout(relu1, keep_prob)
        maxp1 = tf.layers.max_pooling3d(drop1, [1, 2, 2], [1, 2, 2], padding='valid', name='maxpooling1')
        print(maxp1)

    with tf.name_scope('conv2'):
        conv2 = tf.layers.conv3d(maxp1, 64, [3, 5, 5], [1, 1, 1], padding='same',
                                 use_bias=False, kernel_initializer=tf.truncated_normal_initializer, name='conv2')
        batch2 = tf.layers.batch_normalization(conv2, axis=2)
        relu2 = tf.nn.relu(batch2)
        drop2 = tf.nn.dropout(relu2, keep_prob)
        maxp2 = tf.layers.max_pooling3d(drop2, [1, 2, 2], [1, 2, 2], padding='valid', name='maxpooling2')
        print(maxp2)

    with tf.name_scope('conv3'):
        conv3 = tf.layers.conv3d(maxp2, 96, [3, 3, 3], [1, 1, 1], padding='same',
                                 use_bias=False, kernel_initializer=tf.truncated_normal_initializer, name='conv3')
        batch3 = tf.layers.batch_normalization(conv3, axis=-1)
        relu3 = tf.nn.relu(batch3)
        drop3 = tf.nn.dropout(relu3, keep_prob)
        maxp3 = tf.layers.max_pooling3d(drop3, [1, 2, 2], [1, 2, 2], padding='valid', name='maxpooling2')
        print(maxp3)
        resh = tf.reshape(maxp3, [-1, 77, 8*5*96])

    with tf.name_scope('encoder'):
        cells_fw = [tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer),
                tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer)]
        cells_bw = [tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer),
                tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer)]
        seq_len = tf.reduce_sum(video_mask, axis=1)
        #enc_outputs:[batch * max_time * laters_output],
        #layer_output:最后一层前向和后向结果的连接（512）
        #enc_fw_state:每一层前向的最终状态
        #enc_bw_state:每一次后向的最终状态
        #sequence_len:一个batch中每一个Sequence的长度
        enc_outputs, enc_fw_state, enc_bw_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, resh, sequence_length=seq_len, dtype=tf.float32)

    with tf.name_scope('decoder'):
        decoder_inputs = []
        for i in range(25):
            #shape:[batch_size, word_embedding/one-hot]
            decoder_inputs.append(tf.placeholder(tf.float32, shape=[None, 773],
                                                        name="decoder{0}".format(i)))
        #用encoder中最后一层双向gru的状态初始化
        init_state = tf.concat([enc_fw_state[1], enc_bw_state[1]], 1)
        decoder_cell = tf.nn.rnn_cell.GRUCell(512)
        #decoder的inputs是一个[batch_size x input_size]的列表:[[batch_size x input_size], ..... , [batch_size x input_size]]
        #initial_state:[batch_size x cell.state_size]
        #attention_state:[batch_size x attn_len x attn_size]
        #outputs:[batch_size * output_size]
        #state:decoder最后时刻的隐藏状态
        dec_outputs, dec_state = tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs=decoder_inputs,
                                                              initial_state=init_state,
                                                              attention_states=enc_outputs,
                                                              cell=decoder_cell)

    with tf.name_scope('logits'):
        logits = []
        W_output = tf.Variable(tf.random_uniform([512, 773], -0.1, 0.1), name='W_output')
        b_output = tf.Variable(tf.random_uniform([773], -0.1, 0.1), name='b_output')
        for i in range(len(dec_outputs)):
            logits.append(tf.nn.xw_plus_b(dec_outputs[i], W_output, b_output))
        logits = tf.reshape(logits, [-1, 25, 773])

    return logits

def loss(logits, labels):
    loss = tf.contrib.legacy_seq2seq.sequence_loss(logits, labels,
                                                   tf.ones([100, 25]))


inference()