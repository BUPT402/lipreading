from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import glob
from collections import Counter
import math

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('word_counts_output_file',
                       '/home/zyq/video_pipline_data/test/word_counts.txt',
                       'Output vocabulary file of word counts.')

QUEUE_CAPACITY = 200
SHUFFLE_MIN_AFTER_DEQUEUE = QUEUE_CAPACITY // 5


class Vocabulary(object):
    '''vocabulary wrapper'''

    def __init__(self, label_dirs):
        self.id_to_word, self.word_to_id = self._extract_charater_vocab(label_dirs)

    def _extract_charater_vocab(self, label_dirs):
        '''get label_to_text'''
        word_counts = Counter()
        for label_dir in label_dirs:
            # print(label_dir)
            label_list = glob.glob(os.path.join(label_dir, '*align'))
            # print(label_list)
            label_list = sorted(label_list)
            for i in range(len(label_list)):
                label_path = label_list[i]
                f = open(label_path, 'r', encoding='utf-8')
                lines = f.readlines()
                for line in lines[1:-1]:
                    chara = line.split(' ')[-1]
                    word_counts.update(chara)
                f.close()

        # print('Total words:', len(word_counts))

        word_counts = [x for x in word_counts.items()]
        word_counts.sort(key=lambda x: x[1], reverse=True)
        # with open(FLAGS.word_counts_output_file, 'w', encoding='utf-8') as f:
        #     f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))

        words = [x[0] for x in word_counts]
        special_words = ['<PAD>', '<EOS>', '<BOS>', '<unkonw>']
        int_to_vocab = {idx: word for idx, word in enumerate(special_words + words)}
        vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
        return int_to_vocab, vocab_to_int


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
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(serialized_example)
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

    # 变长不用reshape
    # labels = tf.reshape(labels, (70,))


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

    file_lists = glob.glob(os.path.join(data_dir, 'train*'))
    filename_queue = tf.train.string_input_producer(file_lists)
    train_data, train_label, label_length = parse_sequence_example_test(filename_queue)
    frame_batch, label_batch, label_length_batch = tf.train.shuffle_batch(
        [train_data, train_label, label_length],
        batch_size=batch_size,
        num_threads=num_thread,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples
    )
    print(frame_batch)
    return frame_batch, label_batch, label_length_batch


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


def var_len_train_batch_generator(data_dir, batch_size, min_queue_examples, num_thread, shuffle=True):
    file_lists = glob.glob(os.path.join(data_dir, 'train*'))
    filename_queue = tf.train.string_input_producer(file_lists, shuffle=True)
    train_data, train_label, label_length = parse_sequence_example_test(filename_queue)
    input_tensors = [train_data, train_label, label_length]

    if shuffle:
        if num_thread < 2:
            raise ValueError(
                '`num_enqueuing_threads` must be at least 2 when shuffling.')
        shuffle_threads = int(math.ceil(num_thread) / 2.)

        # Since there may be fewer records than SHUFFLE_MIN_AFTER_DEQUEUE, take the
        # minimum of that number and the number of records.
        min_after_dequeue = count_records(
            file_lists, stop_at=SHUFFLE_MIN_AFTER_DEQUEUE)
        input_tensors = _shuffle_inputs(
            input_tensors, capacity=QUEUE_CAPACITY,
            min_after_dequeue=min_after_dequeue,
            num_threads=shuffle_threads)

        num_thread -= shuffle_threads

    tf.logging.info(input_tensors)

    return tf.train.batch(
        input_tensors,
        batch_size=batch_size,
        capacity=QUEUE_CAPACITY,
        num_threads=num_thread,
        dynamic_pad=True,
        allow_smaller_final_batch=False
    )


if __name__ == '__main__':
    data_dir = '/home/zyq/dataset/test1000'
    train_batch_generator(data_dir, 20, 100, 8)
