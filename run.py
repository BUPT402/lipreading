# encoding = utf-8
import tensorflow as tf
from image2seq import Image2Seq as Model
from input import Vocabulary
from input import var_len_train_batch_generator
import os

depth = 250
beam_width = 3
epoch_num = 100
keep_prob = 0.5
label_dir = ['/home/zyq/dataset/dataset_label/test1000']
output_dir = '/home/lin/liprading-master/result'
data_dir = '/home/zyq/dataset/var_len_tfrecord/test1000'
batch_size = 10
data_loader = var_len_train_batch_generator(data_dir, batch_size, 100, 8, shuffle=True)
summary_dir = '/tmp/lin'


def run():
    # saver = tf.train.Saver()
    model = Model(word2idx=Vocabulary(label_dirs=label_dir).word_to_id, depth=depth, img_height=90, img_width=140,
                  beam_width=beam_width, batch_size=batch_size, keep_prob=keep_prob)
    model.sess.run(tf.global_variables_initializer())
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    wirter = tf.summary.FileWriter(summary_dir, model.sess.graph)
    print('model compiled')
    images, captions, label_length = data_loader

    for i in range(epoch_num):
        # print('111111111111')
        print('Epoch %d begin' % i)
        loss = model.partial_fit(images=images, captions=captions, lengths=label_length)
        print('%d Loss: %.4f' % (i, loss))
        # print(label_length)
        # if i % 10 == 0:
        #     valid_images, captions, label_length = data_loader
        #     model.infer(valid_images, Vocabulary(label_dirs=label_dir).id_to_word)
        #     print(captions)
        # #     saver.save(sess, output_dir.join(str(i)))

if __name__=='__main__':
    run()
