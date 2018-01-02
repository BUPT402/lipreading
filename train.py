import tensorflow as tf
from model import Lipreading as Model
from input import Vocabulary, load_video
import datetime
from tqdm import tqdm
import os
from configuration import ModelConfig, TrainingConfig
from eval import run_once

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('vocab_path', '/home/zyq/dataset/word_counts.txt', 'dictionary path')

tf.flags.DEFINE_integer('NUM_EPOCH', 100, 'epoch次数')

tf.flags.DEFINE_string('input_file', '/home/zyq/dataset/tfrecords', 'tfrecords路径')

tf.flags.DEFINE_string('checkpoint_dir', '/home/zyq/codes/lipreading/attention2017:12:31:15:06:24',
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
    count = 0
    for epoch in range(FLAGS.NUM_EPOCH):
        model.mode = 'train'
        print('[Epoch %d] begin ' % epoch)
        final_loss = 0
        for i in tqdm(range(int(num_iteration))):
        # for i in tqdm(range(200)):
            count += 1
            loss = model.run()
            print('\n   [%d ] Loss: %.4f' % (i, loss))
            if count % 100 == 0:
                summary = model.merged_summary()
                summary_writer.add_summary(summary, count)
            final_loss = loss
        epoch_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=final_loss)])
        summary_writer.add_summary(epoch_summary, epoch)
        saver.save(model.sess, os.path.join(model_dir, model_name + str(epoch)))
        print('[Epoch %d] end ' % epoch)

        #############################################################
        print('Epoch %d] eval begin' % epoch)
        model.mode = 'eval'
        run_once(model_config, model, summary_writer, vocab, epoch)
        print('Epoch %d] eval end' % epoch)
        #############################################################

    summary_writer.close()

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
