import os
import tensorflow as tf
from resnet_attention.infer_model import Lipreading as Model
from resnet_attention.input import Vocabulary
import datetime
from resnet_attention.configuration import ModelConfig, TrainingConfig
import numpy as np
import skvideo.io
import dlib

R_MEAN = 90.1618
G_MEAN = 62.8704
B_MEAN = 58.1928

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('vocab_path', '/home/zyq/dataset/ST-0/tfrecords/val_1and2/word_counts.txt', 'dictionary path')

tf.flags.DEFINE_string('input_file', '/home/zyq/dataset/ST-0/tfrecords/val_1and2', 'tfrecords路径')

tf.flags.DEFINE_string('checkpoint_dir', '/home/zyq/codes/lipreading/resnet_attention/accurcy_60',
                       '最近一次模型保存文件')

tf.flags.DEFINE_string('video_path', '', '测试视频的路径')

tf.flags.DEFINE_string('predictor_path', '/home/zyq/VIdeo_pipline/bin/shape_predictor_68_face_landmarks.dat', 'dlib predictor')

def main(unused_argv):

    model_dir = 'attention' + datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    model_name = 'ckp'
    model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    vocab = Vocabulary(FLAGS.vocab_path)

    model_config = ModelConfig()
    train_config = TrainingConfig()
    model_config.input_file = FLAGS.input_file

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 24
    config.inter_op_parallelism_threads = 24
    model = Model(model_config=model_config, word2idx=vocab.word_to_id)

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    print('—*-*-*-*-*-*-*-Model compiled-*-*-*-*-*-*-*-')

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FLAGS.predictor_path)
    video_frames = get_video_frames(FLAGS.video_path)
    mouth_frames = get_video_frames(video_frames, detector, predictor)
    mean = np.array([R_MEAN, G_MEAN, B_MEAN])
    mouth_frames = np.subtract(mouth_frames, mean)
    mouth_frames = np.divide(mouth_frames, 255)
    mouth_frames = np.expand_dims(mouth_frames, 0)
    out_indices = sess.run(model.predicting_ids, feed_dict={model.image_seqs: mouth_frames})
    print(''.join([vocab.id_to_word[k] for k in out_indices]))


def get_video_frames(path):
    # 原来使用skvideo读取视频文件
    videogen = skvideo.io.vreader(path)
    # print(videogen)
    frames = np.array([frame for frame in videogen])
    return frames

def get_frames_mouth(detector, predictor, frames):
    mouth_frames = []
    for frame in frames:
        dets = detector(frame, 1)
        shape = None
        for k, d in enumerate(dets):
            shape = predictor(frame, d)
            i = -1
        if shape is None:  # Detector doesn't detect face, just return as is
            return frames
        mouth_points = []
        for part in shape.parts():
            i += 1
            if i < 48:
                continue
            mouth_points.append((part.x, part.y))
        np_mouth_points = np.array(mouth_points)
        mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0).astype(int)
        mouth_crop_image = frame[mouth_centroid[1] - 45:mouth_centroid[1] + 45,
                           mouth_centroid[0] - 70:mouth_centroid[0] + 70]
        mouth_frames.append(mouth_crop_image)
    return mouth_frames

if __name__ == '__main__':
    tf.app.run()
