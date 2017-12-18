# encoding = utf-8
# print('123')
import tensorflow as tf
from image2seq import Image2Seq as Model
from data.data_to_tfrecord import Vocabulary
from data.data_to_tfrecord import get_loader

depth = 250
beam_width = 3
epoch_num = 100
keep_prob = 0.5
label_dir = 'home/lin/datasets/label'
output_dir ='home/lin/liprading-master/result'
data_loader = get_loader()


def run():
    saver = tf.train.Saver()
    model = Model(word2idx=Vocabulary(label_dir=label_dir).word_to_id, depth=depth, img_height=90, img_width=140,
                  beam_width=beam_width,
                  keep_prob=keep_prob)
    sess = model.sess.run(tf.global_variables())
    print('model compiled')
    for i in range(epoch_num):
        images, captions, label_length = get_loader(train_flag=True)
        loss = model.partial_fit(images=images, captions=captions, lengths=label_length)
        print('%d Loss: %.4f' % (i, loss))
        if i % 10 == 0:
            valid_images = get_loader(train_flag=False)
            model.infer(valid_images, Vocabulary(label_dir=label_dir).id_to_word)
            saver.save(sess, output_dir.join(str(i)))
