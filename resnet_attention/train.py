import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from resnet_attention.without_encoder_model import Lipreading as Model
from resnet_attention.input import Vocabulary, build_dataset
import datetime
from resnet_attention.configuration import ModelConfig, TrainingConfig
import numpy as np
from statistic import cer_s


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('vocab_path', '/home/zyq/dataset/ST0_112/tfrecord/word_counts.txt', 'dictionary path')

tf.flags.DEFINE_integer('NUM_EPOCH', 100, 'epoch次数')

tf.flags.DEFINE_string('input_file', '/home/zyq/dataset/ST0_112/tfrecord', 'tfrecords路径')

tf.flags.DEFINE_string('checkpoint_dir', '/home/zyq/codes/lipreading/resnet_attention/image112_drop0.5_noregu_schedule0.5_2',
                       '最近一次模型保存文件')

tf.flags.DEFINE_string('log_path', 'logs_resnet/without_encoder_freeze',
                       '保存log的地址')


def main(unused_argv):

    # model_dir = 'plus_freezeConv_adam1e-4'
    model_dir = 'without_encoder_freeze'
    model_name = 'ckp'
    model_path = '/home/zyq/codes/lipreading/resnet_attention/infer_model/infer'
    vocab = Vocabulary(FLAGS.vocab_path)
    vocab.id_to_word[-1] = '-1'

    model_config = ModelConfig()
    train_config = TrainingConfig()

    model_config.input_file = FLAGS.input_file
    train_dataset = build_dataset(model_config.train_tfrecord_list, batch_size=model_config.batch_size, shuffle=True)
    # train_dataset = build_dataset(model_config.train_tfrecord_list, batch_size=model_config.batch_size, shuffle=True)
    val_dataset = build_dataset(model_config.val_tfrecord_list, batch_size=model_config.batch_size, is_training=False)
    iterator = tf.contrib.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    val_init_op = iterator.make_initializer(val_dataset)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 24
    config.inter_op_parallelism_threads = 24
    model = Model(model_config=model_config, iterator=iterator, train_config=train_config, word2idx=vocab.word_to_id)

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    var_list = []
    for v in tf.trainable_variables():
        if 'resnet' in v.name or 'conv3d' in v.name:
            var_list.append(v)
        # if v.name == 'decoder/decode_embedding':
        #     var_list.append(v)
    restore_saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=20)
    restore_saver.restore(sess, model_path)

    saver = tf.train.Saver(max_to_keep=20)
    # saver.restore(sess, model_path)

    summary_writer = tf.summary.FileWriter(FLAGS.log_path, graph=sess.graph)

    print('—*-*-*-*-*-*-*-Model compiled-*-*-*-*-*-*-*-')


    count = 0
    for epoch in range(FLAGS.NUM_EPOCH):
        print('[Epoch %d] train begin ' % epoch)
        sess.run(train_init_op)
        i = 0
        while True:
            try:
                loss = model.train(sess)
                print('\n   [%d ] Loss: %.4f' % (i, loss))
                # print([''.join([vocab.id_to_word[i] for i in id]) for id in idx ])
                if count % 100 == 0:
                    train_summary = model.merge(sess)
                    summary_writer.add_summary(train_summary, count)
                count += 1
                i += 1
            except:
                break
        saver.save(sess, os.path.join(model_dir, model_name + str(epoch)))
        print('[Epoch %d] train end ' % epoch)

        print('Epoch %d] eval begin' % epoch)
        sess.run(val_init_op)
        val_pairs = []
        i = 0
        while True:
            try:
                out_indices, loss, y = model.eval(sess)
                print('\n   [%d ] Loss: %.4f' % (i, loss))
                for j in range(len(y)):
                    unpadded_out = None
                    if 1 in out_indices[j]:
                        idx_1 = np.where(out_indices[j] == 1)[0][0]
                        unpadded_out = out_indices[j][:idx_1]
                    else:
                        unpadded_out = out_indices[j]
                    idx_1 = np.where(y[j] == 1)[0][0]
                    unpadded_y = y[j][:idx_1]
                    predic = ''.join([vocab.id_to_word[k] for k in unpadded_out])
                    label = ''.join([vocab.id_to_word[i] for i in unpadded_y])
                    val_pairs.append((predic, label))
                i += 1
            except:
                break
        counts, cer = cer_s(val_pairs)

        summary = tf.Summary(value=[tf.Summary.Value(tag="cer", simple_value=cer)])
        summary_writer.add_summary(summary, epoch)

        print('Current error rate is : %.4f' % cer)
        print('Epoch %d] eval end' % epoch)
        #############################################################

    summary_writer.close()



if __name__ == '__main__':
    tf.app.run()
