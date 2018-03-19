import os
import tensorflow as tf
from resnet_attention.infer_model import Lipreading as Model
from resnet_attention.input import Vocabulary
import datetime
from resnet_attention.configuration import ModelConfig, TrainingConfig
from skimage import transform
import numpy as np
import skvideo.io
import skimage
import dlib
import glob


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('vocab_path', '/home/zyq/dataset/ST0_112/tfrecord/word_counts.txt', 'dictionary path')

tf.flags.DEFINE_string('model_path', '/home/zyq/codes/lipreading/resnet_attention/endtoend_schedule_adam1e-4/ckp3',
                       '最近一次模型保存文件')

tf.flags.DEFINE_string('video_dir', '/home/lin/2s', '测试视频的路径')

tf.flags.DEFINE_string('predictor_path', '/home/zyq/VIdeo_pipline/bin/shape_predictor_68_face_landmarks.dat', 'dlib predictor')



def main(unused_argv):


    vocab = Vocabulary(FLAGS.vocab_path)
    vocab.id_to_word['-1'] = -1
    vocab.id_to_word[-1] = -1

    model_config = ModelConfig()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 24
    config.inter_op_parallelism_threads = 24
    model = Model(model_config=model_config, word2idx=vocab.word_to_id)

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.model_path)

    print('—*-*-*-*-*-*-*-Model compiled-*-*-*-*-*-*-*-')

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FLAGS.predictor_path)
    video_list = glob.glob(os.path.join(FLAGS.video_dir, '*mp4'))
    for video in video_list:
        # try:
            video_frames = get_video_frames(video)
            mouth_frames = get_frames_mouth(detector, predictor, video_frames)

            length = len(mouth_frames)
            mouth_frames = np.subtract(mouth_frames, 0.5)
            mouth_frames = np.multiply(mouth_frames, 2)
            mouth_frames = np.expand_dims(mouth_frames, 0)
            out_indices = sess.run(model.predicting_ids, feed_dict={model.image_seqs: mouth_frames,
                                                                    model.image_length: length})
            res = out_indices[0]
            print(video + ' result : ',[vocab.id_to_word[k] for k in res])


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
            return
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
        width = (right - left) // 2
        np_mouth_points = np.array(mouth_points)
        mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0).astype(int)
        mouth_crop_image = frame[mouth_centroid[1] - width:mouth_centroid[1] + width,
                           mouth_centroid[0] - width:mouth_centroid[0] + width]
        mouth_crop_image = transform.resize(mouth_crop_image, [112, 112])
        mouth_crop_image = skimage.img_as_float(mouth_crop_image)
        mouth_frames.append(mouth_crop_image)
    return mouth_frames

if __name__ == '__main__':
    print(11111)
    tf.app.run()
