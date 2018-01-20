import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from test.test_model import Lipreading as Model
from input_ST0 import Vocabulary, build_dataset
import datetime
from tqdm import tqdm
from configuration_ST0 import ModelConfig, TrainingConfig
import numpy as np
from statistic import cer_s


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('vocab_path', '/home/zyq/dataset/ST-0/tfrecords/2/word_counts.txt', 'dictionary path')

tf.flags.DEFINE_integer('NUM_EPOCH', 100, 'epoch次数')

tf.flags.DEFINE_string('input_file', '/home/zyq/dataset/ST-0/tfrecords/2', 'tfrecords路径')

tf.flags.DEFINE_string('checkpoint_dir', '/home/zyq/codes/lipreading/attention2018:01:17:19:56:27',
                       '最近一次模型保存文件')

def main(unused_argv):

    model_dir = 'attention' + datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    model_name = 'ckp'
    model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    vocab = Vocabulary(FLAGS.vocab_path)

    model_config = ModelConfig()
    train_config = TrainingConfig()
    model_config.input_file = FLAGS.input_file


    train_dataset = build_dataset(model_config.train_tfrecord_list, batch_size=model_config.batch_size)
    val_dataset = build_dataset(model_config.val_tfrecord_list, batch_size=model_config.batch_size)
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

    saver = tf.train.Saver()
    # saver.restore(sess, model_path)

    summary_writer = tf.summary.FileWriter('logs_all/log' +
                                                    datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S'),
                                                    graph=sess.graph)

    print('—*-*-*-*-*-*-*-Model compiled-*-*-*-*-*-*-*-')

    # val_total_loss = 0
    # sess.run(val_init_op)
    # val_pairs = []
    # i = 0
    # while True:
    #     try:
    #         loss, out_indices, y = model.eval(sess)
    #         print("loss", loss, "\n")
    #         print("out", out_indices[0].shape, "\n")
    #         print("y", y[0].shape, "\n")
    #         val_total_loss += loss
    #         for k in range(len(y)):
    #             print(''.join([vocab.id_to_word[id] for id in out_indices[k][0]]))
    #             print(''.join([vocab.id_to_word[id] for id in y[k]]))
    #         print('\n   [%d ] Loss: %.4f' % (i, loss))
    #         for j in range(len(y)):
    #             unpadded_out = None
    #             if 1 in out_indices[j][0]:
    #                 idx_1 = np.where(out_indices[j][0] == 1)[0][0]
    #                 unpadded_out = out_indices[j][0][:idx_1]
    #             else:
    #                 unpadded_out = out_indices[j][0]
    #             idx_1 = np.where(y[j] == 1)[0][0]
    #             unpadded_y = y[j][1:idx_1]
    #             predic = ''.join([vocab.id_to_word[k] for k in unpadded_out])
    #             label = ''.join([vocab.id_to_word[i] for i in unpadded_y])
    #             val_pairs.append((predic, label))
    #         i += 1
    #     except:
    #         break
    # avg_loss = val_total_loss / i
    # counts, cer = cer_s(val_pairs)
    #
    # summary = tf.Summary(value=[tf.Summary.Value(tag="cer", simple_value=cer),
    #                             tf.Summary.Value(tag="val_loss", simple_value=avg_loss)])
    # summary_writer.add_summary(summary, 0)
    # print('Current error rate is : %.4f' % cer)


    count = 0
    for epoch in range(1, FLAGS.NUM_EPOCH):
        print('[Epoch %d] train begin ' % epoch)
        train_total_loss = 0
        sess.run(train_init_op)
        i = 0
        while True:
            try:
                loss, train_summary = model.train(sess)
                train_total_loss += loss
                print('\n   [%d ] Loss: %.4f' % (i, loss))
                if count % 100 == 0:
                    summary_writer.add_summary(train_summary, count)
                count += 1
                i += 1
            except:
                break
        train_loss = train_total_loss / i
        epoch_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=train_loss)])
        summary_writer.add_summary(epoch_summary, epoch)
        saver.save(sess, os.path.join(model_dir, model_name + str(epoch)))
        print('[Epoch %d] train end ' % epoch)

        print('Epoch %d] eval begin' % epoch)
        val_total_loss = 0
        sess.run(val_init_op)
        val_pairs = []
        i = 0
        while True:
            try:
                loss, out_indices, y = model.eval(sess)
                val_total_loss += loss
                print('\n   [%d ] Loss: %.4f' % (i, loss))
                for j in range(len(y)):
                    unpadded_out = None
                    if 1 in out_indices[j]:
                        idx_1 = np.where(out_indices[j] == 1)[0][0]
                        unpadded_out = out_indices[j][:idx_1]
                    else:
                        unpadded_out = out_indices[j]
                    idx_1 = np.where(y[j] == 1)[0][0]
                    unpadded_y = y[j][1:idx_1]
                    predic = ''.join([vocab.id_to_word[k] for k in unpadded_out])
                    label = ''.join([vocab.id_to_word[i] for i in unpadded_y])
                    val_pairs.append((predic, label))
                i += 1
            except:
                break
        avg_loss = val_total_loss / i
        counts, cer = cer_s(val_pairs)

        summary = tf.Summary(value=[tf.Summary.Value(tag="cer", simple_value=cer),
                                    tf.Summary.Value(tag="val_loss", simple_value=avg_loss)])
        summary_writer.add_summary(summary, epoch)
        print('Current error rate is : %.4f' % cer)
        print('Epoch %d] eval end' % epoch)
        #############################################################

    summary_writer.close()



if __name__ == '__main__':
    tf.app.run()
