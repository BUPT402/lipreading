from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import os
import glob


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
        # "video_length": tf.FixedLenFeature([], dtype=tf.int64),
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

    # video_length = tf.cast(context_parsed["video_length"], tf.int32)
    label_length = tf.cast(context_parsed["label_length"], tf.int32)

    frames = sequence_parsed["frames"]
    labels = sequence_parsed["labels"]

    frames = tf.decode_raw(frames, np.uint8)
    frames = tf.reshape(frames, (250, 90, 140, 3))
    frames = tf.image.convert_image_dtype(frames, dtype=tf.float32)

    labels = tf.reshape(labels, (60,))


    tf.logging.info(frames)
    return frames, labels, label_length


def train_batch_generator(data_dir, batch_size, min_queue_examples, num_thread):
    '''
    Generate batch for training.
    :param data_dir: Datadir fo tfrecords
    :param batch_size:
    :param min_queue_examples: 最小队列长度，在队列中取batch
    :return:
        frames_batch, labels_batch, labal_length
    '''

    file_lists = glob.glob(os.path.join(data_dir, '*train'))
    filename_queue = tf.train.string_input_producer(file_lists)
    train_data, train_label, label_length = parse_sequence_example_test(filename_queue)
    frame_batch, label_batch, label_length_batch = tf.train.shuffle_batch(
        [train_data, train_label, label_length],
        batch_size=batch_size,
        num_threads=num_thread,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples
    )
    return frame_batch, label_batch, label_length_batch