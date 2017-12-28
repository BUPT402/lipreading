from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import os
import glob
import math
from skimage import io

QUEUE_CAPACITY = 200
SHUFFLE_MIN_AFTER_DEQUEUE = QUEUE_CAPACITY // 5

class Vocabulary(object):
    '''vocabulary wrapper'''

    def __init__(self, dictionary):
        self.id_to_word, self.word_to_id = self._extract_charater_vocab(dictionary)

    def _extract_charater_vocab(self, dictionary):
        '''get label_to_text'''
        words = []
        with open(dictionary, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                words.append(line.split(' ')[0])
        words = sorted(words)

        special_words = ['<PAD>', '<EOS>', '<BOS>', '<unkonw>']
        int_to_vocab = {idx: word for idx, word in enumerate(special_words + words)}
        vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
        return int_to_vocab, vocab_to_int



def parse_sequence_example(serialized_example):
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
    frames = tf.subtract(frames, 0.5)
    frames = tf.multiply(frames, 2.0)

    #变长不用reshape
    # labels = tf.reshape(labels, (70,))


    tf.logging.info(frames)
    return frames, labels, label_length


def _shuffle_inputs(input_tensors, capacity, min_after_dequeue, num_threads):
    """Shuffles tensors in `input_tensors`, maintaining grouping."""
    shuffle_queue = tf.RandomShuffleQueue(
        capacity, min_after_dequeue, dtypes=[t.dtype for t in input_tensors])
    enqueue_op = shuffle_queue.enqueue(input_tensors)
    runner = tf.train.QueueRunner(shuffle_queue, [enqueue_op] * num_threads)
    tf.train.add_queue_runner(runner)

    output_tensors = shuffle_queue.dequeue()

    for i in range(len(input_tensors)):
        output_tensors[i].set_shape(input_tensors[i].shape)

    return output_tensors

def count_records(file_list, stop_at=None):
    """Counts number of records in files from `file_list` up to `stop_at`.

    Args:
      file_list: List of TFRecord files to count records in.
      stop_at: Optional number of records to stop counting at.

    Returns:
      Integer number of records in files from `file_list` up to `stop_at`.
    """
    num_records = 0
    for tfrecord_file in file_list:
        tf.logging.info('Counting records in %s.', tfrecord_file)
        for _ in tf.python_io.tf_record_iterator(tfrecord_file):
            num_records += 1
            if stop_at and num_records >= stop_at:
                tf.logging.info('Number of records is at least %d.', num_records)
                return num_records
    tf.logging.info('Total records: %d', num_records)
    return num_records

def prefetch_input_data(reader,
                        data_dir,
                        batch_size,
                        is_training,
                        num_threads,
                        queue_capcity=200,
                        min_queue_examples=100):

    if is_training:
        file_lists = glob.glob(os.path.join(data_dir, 'train*'))
        filename_queue = tf.train.string_input_producer(file_lists, shuffle=True)
        _, serialized_example = reader.read(filename_queue)
        train_data, train_label, label_length = parse_sequence_example(serialized_example)
        input_tensors = [train_data, train_label, label_length]

        #shuffle操作
        if num_threads < 2:
            raise ValueError(
                '`num_enqueuing_threads` must be at least 2 when shuffling.')
        shuffle_threads = int(math.ceil(num_threads) / 2.)

        # Since there may be fewer records than SHUFFLE_MIN_AFTER_DEQUEUE, take the
        # minimum of that number and the number of records.
        min_after_dequeue = count_records(
            file_lists, stop_at=min_queue_examples)
        input_tensors = _shuffle_inputs(
            input_tensors, capacity=queue_capcity,
            min_after_dequeue=min_after_dequeue,
            num_threads=shuffle_threads)

        num_threads -= shuffle_threads

    else:
        file_lists = glob.glob(os.path.join(data_dir, 'val*'))
        filename_queue = tf.train.string_input_producer(file_lists, shuffle=True)
        _, serialized_example = reader.read(filename_queue)
        train_data, train_label, label_length = parse_sequence_example(serialized_example)
        input_tensors = [train_data, train_label, label_length]

    tf.logging.info(input_tensors)

    return tf.train.batch(
        input_tensors,
        batch_size=batch_size,
        capacity=queue_capcity,
        num_threads=num_threads,
        dynamic_pad=True,
        allow_smaller_final_batch=False
    )


def load_video(frames_dir):
    frames = glob.glob(os.path.join(frames_dir, '*png'))
    frames = sorted(frames)

    frames_np = []
    length = len(frames)
    for frame in frames:
        frame = io.imread(frame)
        frames_np.append(frame)

    while length < 250:
        frames_np.append(io.imread(frames[-1]))
        length += 1

    return frames_np
