import tensorflow as tf
from model import Lipreading as Model
from input import Vocabulary, load_video
import datetime
from tqdm import tqdm
import os
from configuration import ModelConfig, TrainingConfig

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('vocab_path', '/home/zyq/dataset/word_counts.txt', 'dictionary path')

tf.flags.DEFINE_integer('NUM_EPOCH', 100, 'epoch次数')

tf.flags.DEFINE_string('input_file', '/home/zyq/dataset/tfrecords', 'tfrecords路径')

tf.flags.DEFINE_string('checkpoint_dir', '/home/zyq/codes/lipreading/attention2017:12:28:14:32:13',
                       '最近一次模型保存文件')

def main(unused_argv):
    model_dir = 'attention' + datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    model_name = 'ckp'

    model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    vocab = Vocabulary(FLAGS.vocab_path)

    model_config = ModelConfig()
    train_config = TrainingConfig()
    model_config.input_file = FLAGS.input_file

    model = Model(model_config=model_config, train_config=train_config, word2idx=vocab.word_to_id, mode='train')

    model.sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(model.sess, model_path)

    summary_writer = model.summary_writer

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(model.sess, coord)

    print('—*-*-*-*-*-*-*-Model compiled-*-*-*-*-*-*-*-')

    if train_config.num_examples_per_epoch % model_config.batch_size == 0:
        num_iteration = train_config.num_examples_per_epoch / model_config.batch_size
    else:
        num_iteration = train_config.num_examples_per_epoch / model_config.batch_size + 1
    for epoch in range(FLAGS.NUM_EPOCH):
        print('[Epoch %d] begin ' % epoch)
        for i in tqdm(range(int(num_iteration))):
            loss = model.run()
            print('\n   [%d ] Loss: %.4f' % (i, loss))
            if i % 100 == 0:
                summary = model.merged_summary()
                summary_writer.add_summary(summary, i)
        saver.save(model.sess, os.path.join(model_dir, model_name + str(i)))
        print('[Epoch %d] end ' % epoch)
        summary_writer.close()

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
