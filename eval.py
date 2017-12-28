import tensorflow as tf
from model import Lipreading as Model
from input import Vocabulary, load_video
import datetime
from tqdm import tqdm
import os
import numpy as np
from statistic import cer_s
from configuration import ModelConfig, TrainingConfig

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('vocab_path', '/home/zyq/dataset/word_counts.txt', 'dictionary path')

tf.flags.DEFINE_string('input_file', '/home/zyq/dataset/tfrecords', 'tfrecords路径')

tf.flags.DEFINE_integer('num_eval_examples', 3151, '验证集的数量')

tf.flags.DEFINE_string('checkpoint_dir', '/home/zyq/codes/lipreading/attention2017:12:28:14:32:13',
                       '最近一次模型保存文件')

def main(unused_argv):
    model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    vocab = Vocabulary(FLAGS.vocab_path)

    model_config = ModelConfig()
    train_config = TrainingConfig()
    model_config.input_file = FLAGS.input_file

    model = Model(model_config=model_config, train_config=train_config, word2idx=vocab.word_to_id, mode='eval')

    model.sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(model.sess, model_path)
    summary_writer = model.summary_writer

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(model.sess, coord)

    print('—*-*-*-*-*-*-*-Model compiled-*-*-*-*-*-*-*-')

    if FLAGS.num_eval_examples % model_config.batch_size == 0:
        num_iteration = FLAGS.num_eval_examples // model_config.batch_size
    else:
        num_iteration = FLAGS.num_eval_examples // model_config.batch_size + 1

    val_pairs = []
    for i in tqdm(range(int(num_iteration))):
        out_indices, loss, y = model.run()
        print('\n   [%d ] Loss: %.4f' % (i, loss))
        if i % 100 == 0:
            summary = model.merged_summary()
            summary_writer.add_summary(summary, i)
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
    count, cer = cer_s(val_pairs)
    print('Current error rate is : %.4f' % cer)
    summary_writer.close()

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
