from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

def parse_sequence_example_train(serialized_example):
    '''Parse a tensorflow.SequenceExample into video_length, label_length,
    frames, labels when training the model

    Args:
        serialized: a single serialized SequenceExample

    Returns:
        frames: video frames
        labels: labels for frame
        label_len: label's length
    '''

    context_features = {
        "video_length": tf.FixedLenFeature([], dtype=tf.int64),
        "label_length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "frames": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    video_length = tf.cast(context_parsed["video_length"], tf.int32)
    label_length = tf.cast(context_parsed["label_length"], tf.int32)

    frames = sequence_parsed["frames"]
    labels = sequence_parsed["labels"]

    frames = tf.decode_raw(frames, np.uint8)
    frames = tf.reshape(frames, (-1, 90, 140, 3))

    tf.logging.info(frames)
    return frames, labels, label_length

def parse_sequence_example_test(serialized_example):
    '''Parse a tensorflow.SequenceExample into video_length, label_length,
    frames, labels when testing

    Args:
        serialized: a single serialized SequenceExample

    Returns:
        frames: video frames
        labels: labels for frame
        label_len: label's length
    '''

    context_features = {
        "video_length": tf.FixedLenFeature([], dtype=tf.int64),
        "label_length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "frames": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    video_length = tf.cast(context_parsed["video_length"], tf.int32)
    label_length = tf.cast(context_parsed["label_length"], tf.int32)

    frames = sequence_parsed["frames"]
    labels = sequence_parsed["labels"]

    frames = tf.decode_raw(frames, np.uint8)
    frames = tf.reshape(frames, (-1, 90, 140, 3))

    tf.logging.info(frames)
    return frames, labels


def train_batch_generator(data_dir, batch_size, min_queue_examples):
    '''
    Generate batch for training.
    :param data_dir: Datadir fo tfrecords
    :param batch_size:
    :param min_queue_examples: 最小队列长度，在队列中取batch
    :return:
        frames_batch, labels_batch, labal_length
    '''


