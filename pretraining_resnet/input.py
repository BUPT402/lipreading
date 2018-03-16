from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

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

        int_to_vocab = {idx: word for idx, word in enumerate(words)}
        vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
        return int_to_vocab, vocab_to_int

def build_dataset(filenames, batch_size, buffer_size=2000, repeat=None, num_threads=8, shuffle=False, is_training=True):
    dataset = tf.data.TFRecordDataset(filenames)

    if is_training:
        dataset = dataset.map(_train_parse_function, num_parallel_calls=12)
    else:
        dataset = dataset.map(_val_parse_function, num_parallel_calls=12)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.padded_batch(batch_size, padded_shapes=([29, 112, 112, 3], []),
                                   padding_values=(0.0, 0))

    return dataset


def _train_parse_function(example_proto):
    context_features = {
        "label": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "frames": tf.FixedLenSequenceFeature([], dtype=tf.string),
    }
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=example_proto,
        context_features=context_features,
        sequence_features=sequence_features
    )
    label = tf.cast(context_parsed["label"], tf.int32)

    frames = sequence_parsed["frames"]

    frames = tf.decode_raw(frames, np.uint8)
    frames = tf.reshape(frames, (29, 112, 112, 3))
    frames = tf.image.convert_image_dtype(frames, dtype=tf.float32)

    norm_frames = None
    for i in range(29):
        if norm_frames == None:
            frame = frames[i, :, :, :]
            frame = tf.image.random_flip_left_right(frame)
            norm_frames = tf.expand_dims(frame, 0)
        else:
            frame = frames[i, :, :, :]
            frame = tf.image.random_flip_left_right(frame)
            frame = tf.expand_dims(frame, 0)
            norm_frames = tf.concat([norm_frames, frame], 0)

    norm_frames = tf.subtract(norm_frames, 0.5)
    norm_frames = tf.multiply(norm_frames, 2)

    return norm_frames, label


def _val_parse_function(example_proto):
    context_features = {
        "label": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "frames": tf.FixedLenSequenceFeature([], dtype=tf.string),
    }
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=example_proto,
        context_features=context_features,
        sequence_features=sequence_features
    )
    label = tf.cast(context_parsed["label"], tf.int32)

    frames = sequence_parsed["frames"]

    frames = tf.decode_raw(frames, np.uint8)
    frames = tf.reshape(frames, (29, 112, 112, 3))
    frames = tf.image.convert_image_dtype(frames, dtype=tf.float32)

    norm_frames = tf.subtract(frames, 0.5)
    norm_frames = tf.multiply(norm_frames, 2)

    return norm_frames, label
