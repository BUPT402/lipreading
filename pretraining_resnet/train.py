import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from pretraining_resnet.conv_back import Lipreading as Model
from pretraining_resnet.input import Vocabulary, build_dataset
import datetime
from pretraining_resnet.configuration import ModelConfig, TrainingConfig
import numpy as np
from statistic import cer_s


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('vocab_path', '/home/zyq/dataset/LRW/tfrecord/word_counts.txt', 'dictionary path')

tf.flags.DEFINE_integer('NUM_EPOCH', 100, 'epoch次数')

tf.flags.DEFINE_string('input_file', '/home/zyq/dataset/LRW/tfrecord', 'tfrecords路径')

tf.flags.DEFINE_string('checkpoint_dir', '/home/zyq/codes/lipreading/resnet_attention/image112_drop0.5_noregu_schedule0.5_2',
                       '最近一次模型保存文件')

tf.flags.DEFINE_string('save_model_path', 'logs_resnet/lr0.005_nodrop_noregular_02',
                       '保存模型的地址')


def main(unused_argv):

    model_dir = 'attention' + datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    model_dir = 'nodrop_noregu_'
    model_name = 'ckp'
    model_path = '/home/zyq/codes/lipreading/pretraining_resnet/image112_drop0.4_noregu_schedule0.2/ckp0'
    # model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    vocab = Vocabulary(FLAGS.vocab_path)

    model_config = ModelConfig()
    train_config = TrainingConfig()
    # train_config.global_step = tf.Variable(0, trainable=False)
    # train_config.learning_rate = tf.train.exponential_decay(train_config.learning_rate,
    #                                                         train_config.global_step,
    #                                                         train_config.num_iteration_per_decay,
    #                                                         train_config.learning_rate_decay,
    #                                                         staircase=True)


    model_config.input_file = FLAGS.input_file
    train_dataset = build_dataset(model_config.train_tfrecord_list, batch_size=model_config.batch_size, shuffle=True)
    # train_dataset = build_dataset(model_config.train_tfrecord_list, batch_size=model_config.batch_size)
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
    for v in tf.global_variables():
        if 'Variable' not in v.name:
            var_list.append(v)

    # saver = tf.train.Saver(max_to_keep=20)
    saver = tf.train.Saver(var_list=var_list, max_to_keep=20)
    saver.restore(sess, model_path)


    summary_writer = tf.summary.FileWriter(FLAGS.save_model_path, graph=sess.graph)

    print('—*-*-*-*-*-*-*-Model compiled-*-*-*-*-*-*-*-')


    count = 0
    for epoch in range(FLAGS.NUM_EPOCH):
        print('[Epoch %d] train begin ' % epoch)
        train_total_loss = 0
        sess.run(train_init_op)
        while True:
            try:
                loss = model.train(sess)
                train_total_loss += loss
                print('\n   [%d ] Loss: %.4f' % (count, loss))
                # print([''.join([vocab.id_to_word[i] for i in id]) for id in idx ])
                if count % 100 == 0:
                    train_summary = model.merge(sess)
                    summary_writer.add_summary(train_summary, count)
                count += 1
            except:
                break
        saver.save(sess, os.path.join(model_dir, model_name + str(epoch)))
        print('[Epoch %d] train end ' % epoch)

        print('Epoch %d] eval begin' % epoch)
        val_total_accuracy = 0
        sess.run(val_init_op)
        i = 0
        while True:
        # for _ in range(10):
            try:
                accuracy = model.eval(sess)
                val_total_accuracy += accuracy[0]
                # print('\n   [%d ] Accuracy: %d' % (i, accuracy))
                # val_total_accuracy += accuracy
                i += 1
            except:
                break
        avg_accuracy = val_total_accuracy / i

        summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=avg_accuracy)])
        summary_writer.add_summary(summary, epoch)

        print('Epoch %d] eval end' % epoch)
        #############################################################

    summary_writer.close()



if __name__ == '__main__':
    tf.app.run()
