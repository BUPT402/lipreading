# encoding = utf-8
import tensorflow as tf
from model import Lipreading as Model
from input import Vocabulary
# from data.data_to_tfrecord import get_loader
from input import train_batch_generator
from input import var_len_train_batch_generator
import os

depth = 250
beam_width = 3
epoch_num = 100
num_thread = 4
min_after_dequeue = 100
keep_prob = 0.5
batch_size = 10
label_dir = ['/home/zyq/dataset/dataset_label/test1000']
dictionary = '/home/zyq/dataset/'
output_dir ='/home/lin/liprading-master/result'
data_dir = '/home/zyq/dataset/tfrecords'
data_loader = var_len_train_batch_generator(data_dir, batch_size, min_after_dequeue, num_thread, shuffle=True)
summary_dir = '/tmp/lipreading'


def run():
    # saver = tf.train.Saver()
    model = Model(word2idx=Vocabulary(dictionary=dictionary).word_to_id, depth=depth, img_height=90, img_width=140,
                  beam_width=beam_width , batch_size=batch_size, keep_prob=keep_prob)
    model.sess.run(tf.global_variables_initializer())
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    wirter = tf.summary.FileWriter(summary_dir, model.sess.graph)
    print('model compiled')
    images, captions, label_length = data_loader

    for i in range(epoch_num):
        # for i in range(2):
        #     with tf.device('/gpu:%d' % i):
        loss = model.partial_fit(images=images, captions=captions, lengths=label_length)
        print('%d Loss: %.4f' % (i, loss))
        # if i % 10 == 0:
        # #   valid_images = train_batch_generator()
        # #     model.infer(valid_images, Vocabulary(label_dirs=label_dir).id_to_word)
        #     saver.save(sess, output_dir.join(str(i)))

run()
