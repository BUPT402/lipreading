from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import glob
import math
from skimage import io


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

def build_dataset(filenames, batch_size, buffer_size=200, repeat=None, shuffle=False):
    dataset = tf.data.TFRecordDataset(filenames)

    dataset = dataset.map(_parse_function)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.padded_batch(batch_size, padded_shapes=([77, 90, 140, 3], [None], []))
    # if repeat != None:
    dataset = dataset.repeat()

    return dataset


def _parse_function(example_proto):
    context_features = {
        "label": tf.FixedLenFeature([], dtype=tf.int64)
    }
    context_features = {
        # "video_length": tf.FixedLenFeature([], dtype=tf.int64),
        "label_length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "frames": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=example_proto,
        context_features=context_features,
        sequence_features=sequence_features
    )
    label_length = tf.cast(context_parsed["label_length"], tf.int32)

    frames = sequence_parsed["frames"]
    labels = sequence_parsed["labels"]

    frames = tf.decode_raw(frames, np.uint8)
    frames = tf.reshape(frames, (77, 90, 140, 3))
    frames = tf.image.convert_image_dtype(frames, dtype=tf.float32)
    frames = tf.subtract(frames, 0.5)
    frames = tf.multiply(frames, 2.0)

    return frames, labels, label_length
