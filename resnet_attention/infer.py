import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from models.infer_model import Lipreading as Model
from resnet_attention.input import Vocabulary
from resnet_attention.configuration import ModelConfig
from skimage import transform
import numpy as np
import skvideo.io
import skimage
import dlib

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('vocab_path', '/home/zyq/dataset/ST0_112/tfrecord/word_counts.txt', 'dictionary path')
# tf.flags.DEFINE_string('model_path', '/home/zyq/codes/lipreading/resnet_attention/plus_endtoend/ckp10',
tf.flags.DEFINE_string('model_path', '/home/zyq/codes/lipreading/resnet_attention/endtoend_schedule_adam1e-4/ckp3',
                       '最近一次模型保存文件')

tf.flags.DEFINE_string('video_dir', '/home/zyq/test/test', '测试视频的路径')

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
    restore_saver = tf.train.Saver()
    restore_saver.restore(sess, FLAGS.model_path)

    img_seq = tf.get_default_graph().get_tensor_by_name('img:0')
    img_length = tf.get_default_graph().get_tensor_by_name('img_len:0')
    prediction = tf.get_default_graph().get_tensor_by_name('decoder/pre_result/strided_slice:0')
    # vifeat = tf.get_default_graph().get_tensor_by_name('conv3d/conv1/strided_slice:0')
    # reinputs = tf.get_default_graph().get_tensor_by_name('resnet/strided_slice:0')
    # bl_layer1 = tf.get_default_graph().get_tensor_by_name('resnet/strided_slice_1:0')
    # bl_layer2 = tf.get_default_graph().get_tensor_by_name('resnet/strided_slice_2:0')
    # bl_layer3 = tf.get_default_graph().get_tensor_by_name('resnet/strided_slice_3:0')
    # en_out = tf.get_default_graph().get_tensor_by_name('encoder/strided_slice:0')
    # out_prob = tf.get_default_graph().get_tensor_by_name('decoder/strided_slice_4:0')
    # out_idx = tf.get_default_graph().get_tensor_by_name('decoder/strided_slice_5:0')


    inputs = {
        "img": tf.saved_model.utils.build_tensor_info(img_seq),
        "img_length": tf.saved_model.utils.build_tensor_info(img_length)
    }

    outputs = {
        "prediction": tf.saved_model.utils.build_tensor_info(prediction),
        # "vifeat": tf.saved_model.utils.build_tensor_info(vifeat),
        # "reinputs": tf.saved_model.utils.build_tensor_info(reinputs),
        # "bl_layer1": tf.saved_model.utils.build_tensor_info(bl_layer1),
        # "bl_layer2": tf.saved_model.utils.build_tensor_info(bl_layer2),
        # "bl_layer3": tf.saved_model.utils.build_tensor_info(bl_layer3),
        # "en_out": tf.saved_model.utils.build_tensor_info(en_out),
        # "out_prob": tf.saved_model.utils.build_tensor_info(out_prob),
        # "out_idx": tf.saved_model.utils.build_tensor_info(out_idx)
    }

    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs, outputs=outputs, method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    signature_def_map = {
        "predict_image": prediction_signature}

    builder = tf.saved_model.builder.SavedModelBuilder('/home/zyq/codes/lipreading/resnet_attention/tmp')
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map=signature_def_map)
    builder.save()


    saver = tf.train.Saver()
    dense = sess.graph.get_operation_by_name('decoder/dense/kernel')
    # for op in sess.graph.get_operations():
    #     if "decode" in op.name:
    #         print(op.name)
    # print([op for op in sess.graph.get_operations() if "decode" in op.name])
    saver.restore(sess, FLAGS.model_path)
    # saver.save(sess, 'infer_plus/infer')
    print('—*-*-*-*-*-*-*-Model compiled-*-*-*-*-*-*-*-')

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FLAGS.predictor_path)
    video_list = ['/home/lin/2s/VID_20171113_092350.mp4']
    with tf.device('/gpu:0'):
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
                print(video + ' result : ', [vocab.id_to_word[k] for k in res])


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
    tf.app.run()
