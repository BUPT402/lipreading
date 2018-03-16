import os
import tensorflow as tf
from resnet_attention.infer_model import Lipreading as Model
from resnet_attention.input import Vocabulary
import datetime
from resnet_attention.configuration import ModelConfig, TrainingConfig
import numpy as np
import skvideo.io
import dlib
import glob

R_MEAN = 90.1618
G_MEAN = 62.8704
B_MEAN = 58.1928

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('vocab_path', '/home/zyq/dataset/ST-0/tfrecords/val_1and2/word_counts.txt', 'dictionary path')

tf.flags.DEFINE_string('input_file', '/home/zyq/dataset/ST-0/tfrecords/val_1and2', 'tfrecords路径')

tf.flags.DEFINE_string('checkpoint_dir', '/home/zyq/codes/lipreading/resnet_attention/accurcy_60',
                       '最近一次模型保存文件')

tf.flags.DEFINE_string('video_dir', '/home/lin/2s', '测试视频的路径')

tf.flags.DEFINE_string('predictor_path', '/home/zyq/VIdeo_pipline/bin/shape_predictor_68_face_landmarks.dat', 'dlib predictor')



def main(unused_argv):

    model_dir = 'attention' + datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    model_name = 'ckp'
    # model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    model_path = '/home/zyq/codes/lipreading/resnet_attention/schedule_1.0/ckp1'

    vocab = Vocabulary(FLAGS.vocab_path)
    vocab.id_to_word['-1'] = -1
    vocab.id_to_word[-1] = -1

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
    video_list = glob.glob(os.path.join(FLAGS.video_dir, '*mp4'))
    # video_list = ['/home/lin/2s/VID_20171113_092350.mp4']
    for video in video_list:
        # try:
            video_frames = get_video_frames(video)
            mouth_frames = get_frames_mouth(detector, predictor, video_frames)
            r_mean = 0
            g_mean = 0
            b_mean = 0
            length = len(mouth_frames)
            for frame in mouth_frames:
                r_mean += np.sum(frame[:, :, 0])
                g_mean += np.sum(frame[:, :, 1])
                b_mean += np.sum(frame[:, :, 2])
            mean = [r_mean/(length * 140 * 90), g_mean/(length * 140 * 90), b_mean/(length * 140 * 90)]
            mouth_frames = np.subtract(mouth_frames, mean)
            mouth_frames = np.divide(mouth_frames, 255)
            mouth_frames = np.expand_dims(mouth_frames, 0)
            print('inputs:', mouth_frames)
            mouth_frames = np.transpose(mouth_frames, [0, 1, 3, 2, 4])
            out_indices = sess.run(model.predicting_ids, feed_dict={model.image_seqs: mouth_frames,
                                                                model.image_length: length})
            res = out_indices[0]
            print(video + ' result : ',[vocab.id_to_word[k] for k in res])
        # except:
        #     print('error')
        #     continue
    # print(''.join([vocab.id_to_word[k] for k in out_indices[0]]))
    # print(''.join([vocab.id_to_word[k] for k in out_indices[1]]))
    # print(''.join([vocab.id_to_word[k] for k in out_indices[2]]))
    # print(''.join([vocab.id_to_word[k] for k in out_indices[3]]))
    # print(''.join([vocab.id_to_word[k] for k in out_indices[4]]))


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
    print(11111)
    tf.app.run()
