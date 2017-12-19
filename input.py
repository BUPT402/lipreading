from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import os
import glob
from collections import Counter


FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string('word_counts_output_file',
                       '/home/zyq/video_pipline_data/test/word_counts.txt',
                       'Output vocabulary file of word counts.')

class Vocabulary(object):
    '''vocabulary wrapper'''

    def __init__(self, label_dirs):
        self.id_to_word, self.word_to_id = self._extract_charater_vocab(label_dirs)

    def _extract_charater_vocab(self, label_dirs):
        '''get label_to_text'''
        word_counts = Counter()
        for label_dir in label_dirs:
            label_list = glob.glob(os.path.join(label_dir, '*align'))
            label_list = sorted(label_list)
            for i in range(len(label_list)):
                label_path = label_list[i]
                f = open(label_path, 'r', encoding='utf-8')
                lines = f.readlines()
                for line in lines[1:-1]:
                    chara = line.split(' ')[-1]
                    word_counts.update(chara)
                f.close()

        print('Total words:', len(word_counts))

        word_counts = [x for x in word_counts.items()]
        word_counts.sort(key=lambda x: x[1], reverse=True)
        with open(FLAGS.word_counts_output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))

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
    return frame_batch, label_batch, label_length_batch