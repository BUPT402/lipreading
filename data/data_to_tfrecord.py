from __future__ import absolute_import,division,print_function

import os
import tensorflow as tf
import numpy as np
import glob
import cv2
import skvideo.io
from skimage import io,color,transform,img_as_ubyte
import dlib
import random
import threading
from datetime import datetime
import sys
from collections import Counter
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_video_dir',
                           '/home/zyq/video_pipline_data/test/dataset_avi/test1000',
                           'Traing video directory')
tf.app.flags.DEFINE_string('val_video_dir',
                           '/home/zyq/video_pipline_data/test/dataset_avi/test',
                           'Validation video directory')

tf.app.flags.DEFINE_string('train_label_dir',
                           '/home/zyq/video_pipline_data/test/dataset_label/test1000',
                           'Traing label directory')
tf.app.flags.DEFINE_string('val_label_dir',
                           '/home/zyq/video_pipline_data/test/dataset_label/test',
                           'Validation label directory')

tf.app.flags.DEFINE_string('output_dir',
                           '/home/zyq/video_pipline_data/test/test1000',
                           'Output directory')

tf.app.flags.DEFINE_string('train_shards', 16,
                           'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_string('val_shards', 8,
                           'Number of shards in training TFRecord files.')

tf.flags.DEFINE_string('start_word', '<BOS>',
                       'Special word added to the beginning of each sentence.')
tf.flags.DEFINE_string('end_word', '<EOS>',
                       'Special word added to the end of each sentence.')

tf.flags.DEFINE_string('num_threads', 8,
                       'Numbers of threads to preprocess the videos.')

tf.flags.DEFINE_string('word_counts_output_file',
                       '/home/zyq/video_pipline_data/test/word_counts.txt',
                       'Output vocabulary file of word counts.')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/zyq/PycharmProjects/'
                                 'Video_pipline/bin/shape_predictor_68_face_landmarks.dat')


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

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature_list(values):
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _bytes_feature_list(values):
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def get_label(vocab, text_path):
    texts = [FLAGS.start_word]
    with open(text_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')[1:-2]
        for line in lines:
            texts.append(line.split(' ')[-1])
    texts.append('<EOS>')

    labels = [vocab.word_to_id.get(text) for text in texts]
    return labels

def get_frames(video_path):
    videogen = skvideo.io.vreader(video_path)
    frames = np.array([frame for frame in videogen])
    return frames

def get_frames_mouth(video_path):
    frames = get_frames(video_path)

    mouth_frames = []
    for frame in frames:
        dets = detector(frame, 1)
        shape = None
        for k, d in enumerate(dets):
            shape = predictor(frame, d)
            i = -1
        if shape is None or len(dets) > 1:  # Detector doesn't detect face, just return as is
            # return
            # mouth_crop_image = frame[mouth_centroid[1] - width:mouth_centroid[1] + width,
            #                    mouth_centroid[0] - width:mouth_centroid[0] + width]
            # mouth_frames.append(mouth_crop_image)
            continue
        mouth_points = []
        for part in shape.parts():
            i += 1
            if i == 4:
                left = part.x
            if i == 12:
                right = part.x
            if i < 48:
                continue
            mouth_points.append((part.x, part.y))
        # width = (right - left) // 2
        np_mouth_points = np.array(mouth_points)
        mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0).astype(int)
        # mouth_crop_image = frame[mouth_centroid[1] - width:mouth_centroid[1] + width,
        #                    mouth_centroid[0] - width:mouth_centroid[0] + width]
        mouth_crop_image = frame[mouth_centroid[1] - 45:mouth_centroid[1] + 45,
                           mouth_centroid[0] - 70:mouth_centroid[0] + 70]
        mouth_frames.append(mouth_crop_image)

    # mouth_frames = [transform.resize(frame, (120, 120), mode='constant') for frame in mouth_frames]
    # mouth_frames = [img_as_ubyte(frame) for frame in mouth_frames]

    return mouth_frames


def _to_sequence_example(video_path, label_path, vocab):
    '''Build a SequenceExample proto for an video-label pair.

    Args:
        video_path: Path of video data.
        label_path: Path of label data.
        vocab: A Vocabulary object.

    Returns:
        A SequenceExample proto.
    '''

    # mouth_frames = get_frames_mouth(video_path)
    # print(mouth_frames[0].shape)
    # avi_len = len(mouth_frames)
    frames_list = glob.glob(os.path.join(video_path, '*.png'))
    mouth_frames = []
    for frame in frames_list:
        mouth_frames.append(io.imread(frame))
    frames_byte = [frame.tostring() for frame in mouth_frames]

    labels = get_label(vocab, label_path)
    label_len = len(labels)

    padding = np.zeros((70 - label_len, ), dtype=np.int32)
    labels = np.concatenate((labels, padding), axis=0)

    example = tf.train.SequenceExample(
        context=tf.train.Features(feature={
            "label_length": _int64_feature(label_len)
        }),
        feature_lists=tf.train.FeatureLists(feature_list={
            "frames": _bytes_feature_list(frames_byte),
            "labels": _int64_feature_list(labels)
        })
    )

    return example


def process_batch_files(thread_index, ranges, name, video_list,
                        label_list, vocab, num_shards, dataset_list):
    '''Processes and saves a subset of video as TFRecord files in one thread.

    Args:
        thread_index: 线程序号
        ranges: 将数据集分成了几个部分，A list of pairs
        name: Unique identifier specifying the dataset
        video_dir：视频数据所在的文件夹
        label_dir：文本数据所在的文件夹
        vocab：A Vocabulary object
        num_shards： 数据集最终分成几个TFRecord

    '''
    num_threads = len(ranges)
    assert  not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_video_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        video_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in video_in_shard:
            idx = dataset_list[i]
            video_path = video_list[idx]
            label_path = label_list[idx]


            sequence_example = _to_sequence_example(video_path, label_path, vocab)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

            if not counter % 1000:
                print("%s [thread %d]: Processed %d of %d items in thread batch." %
                      (datetime.now(), thread_index, counter, num_video_in_thread))
                sys.stdout.flush()

        writer.close()
        print("%s [thread %d]: Wrote %d video-label pairs to %s" %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print("%s [thread %d]: Wrote %d video-label pairs to %d shards." %
          (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()

def process_dataset(name, video_dir, label_dir, vocab, num_shards):
    '''Process a complete data set and save it as a TFRecord.


    Args:
        name: 数据集的名称.
        video_dir: 数据集视频所在文件夹.
        label_dir: 标签所在的文件夹.
        vocab: A Vocabulary object.
    '''

    avi_list = glob.glob(os.path.join(video_dir, '00*'))
    avi_list = sorted(avi_list)
    label_list = glob.glob(os.path.join(label_dir, '*.align'))
    label_list = sorted(label_list)

    dataset_list = [i for i  in range(len(avi_list))]   #视频和label是分开的，分开shuffle就乱了，相当于一个索引
    # dataset_list = [i for i in range(16)]
    random.seed(1117)
    random.shuffle(dataset_list)

    num_threads = min(num_shards, FLAGS.num_threads)
    spacing = np.linspace(0, len(dataset_list), num_threads+1).astype(int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    coord = tf.train.Coordinator()

    print('Launching %d threads for spacing: %s' % (num_threads, ranges))
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, avi_list, label_list, vocab, num_shards, dataset_list)
        t = threading.Thread(target=process_batch_files, args=args)
        t.start()
        threads.append(t)

    coord.join(threads)
    print('%s: Finished processing all %d video-caption pairs in data set "%s".' %
          (datetime.now(), len(video_dir), name))



def main(unused_argv):
    def _is_valid_num_shards(num_shards):
        """Returns True if num_shards is compatible with FLAGS.num_threads."""
        return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

    assert _is_valid_num_shards(FLAGS.train_shards), (
        '''Please make the FALGS.num_threads commensurate with FLAGS.train_shards''')

    assert _is_valid_num_shards(FLAGS.val_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")

    vocab = Vocabulary([FLAGS.train_label_dir])

    process_dataset('train', FLAGS.train_video_dir, FLAGS.train_label_dir, vocab, FLAGS.train_shards)
    # process_dataset('v', FLAGS.val_video_dir, FLAGS.val_label_dir, vocab, FLAGS.val_shards)

if __name__ == '__main__':
    tf.app.run()