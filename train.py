import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from model import Lipreading as Model
from input import Vocabulary, build_dataset
import datetime
from tqdm import tqdm
from configuration import ModelConfig, TrainingConfig
import numpy as np
from statistic import cer_s


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('vocab_path', '/home/zyq/dataset/word_counts.txt', 'dictionary path')

tf.flags.DEFINE_integer('NUM_EPOCH', 100, 'epoch次数')

tf.flags.DEFINE_integer('num_eval_examples', 3151, '验证集的数量')

tf.flags.DEFINE_string('input_file', '/home/zyq/dataset/tfrecords', 'tfrecords路径')

tf.flags.DEFINE_string('checkpoint_dir', '/home/zyq/codes/lipreading/attention2018:01:09:21:05:22',
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
    iterator = tf.contrib.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 24
    config.inter_op_parallelism_threads = 24
    model = Model(model_config=model_config, iterator=iterator, train_config=train_config, word2idx=vocab.word_to_id, sess=tf.Session(config=config))

    model.sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # saver.restore(model.sess, model_path)

    summary_writer = model.summary_writer

    print('—*-*-*-*-*-*-*-Model compiled-*-*-*-*-*-*-*-')

    if train_config.num_examples_per_epoch % model_config.batch_size == 0:
        num_iteration = train_config.num_examples_per_epoch / model_config.batch_size
    else:
        num_iteration = train_config.num_examples_per_epoch / model_config.batch_size + 1

    count = 0
    for epoch in range(FLAGS.NUM_EPOCH):
        print('[Epoch %d] train begin ' % epoch)
        train_total_loss = 0
        model.sess.run(model.train_init_op)
        # for i in tqdm(range(int(num_iteration))):
        for i in tqdm(range(int(3510))):
            loss = model.train()
            count += 1
            train_total_loss += loss
            print('\n   [%d ] Loss: %.4f' % (i, loss))
            if count % 100 == 0:
                summary = model.merge()
                summary_writer.add_summary(summary, count)

        train_loss = train_total_loss / 3510
        epoch_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=train_loss)])
        summary_writer.add_summary(epoch_summary, epoch)
        saver.save(model.sess, os.path.join(model_dir, model_name + str(epoch)))
        print('[Epoch %d] train end ' % epoch)

        if FLAGS.num_eval_examples % model_config.batch_size == 0:
            num_iteration = FLAGS.num_eval_examples // model_config.batch_size
        else:
            num_iteration = FLAGS.num_eval_examples // model_config.batch_size + 1

        print('Epoch %d] eval begin' % epoch)
        val_total_loss = 0
        model.sess.run(model.val_init_op)
        val_pairs = []
        for i in tqdm(range(int(360))):
            try:
                out_indices, loss, y = model.eval()
                val_total_loss += loss
                tf.logging.info('\n   [%d ] Loss: %.4f' % (i, loss))
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
            except tf.errors.OutOfRangeError:
                break
        avg_loss = val_total_loss / 360
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
