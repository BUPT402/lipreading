# encoding = utf-8
import datetime
import tensorflow as tf
import os
from model import Lipreading as Model
from input import Vocabulary



def main(args):
    model_dir = 'attention' + datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    model_name = 'ckp'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    vocab = Vocabulary(args['vocab_path'])
    model = Model(datadir=args['data_dir'], word2idx=vocab.word_to_id, depth=args['depth'], img_height=args['height'],
                  img_width=args['weight'], beam_width=args['beam_width'],
                  batch_size=args['batch_size'], idx2word=vocab.id_to_word)

    model.sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    print('-*-*-*-*-*-Model compiled-*-*-*-*-*-')
    for epoch in range(args['num_epochs']):
        print('[Epoch %d] begin ' % epoch)
        for i in range(args['num_iterations']):
            # loss = model.partial_fit(data_loader)
            loss = model.partial_fit()
            print('[ batch %d ] Loss: %.4f' % (i, loss))
        print('[Epoch %d] end ' % epoch)
        if epoch % 2 == 0:
            print(model_dir)
            # image = []
            # model.infer(image=image, idx2word=vocab.id_to_word)
            saver.save(model.sess, os.path.join(model_dir, model_name+str(epoch)))
        epoch = epoch + 1
    print("训练完成")


if __name__ == '__main__':
    args = {
        'vocab_path': '/home/zyq/dataset/word_counts.txt',   # label地址
        'data_dir': '/home/zyq/dataset/tfrecords',  # 数据地址
        'batch_size': 10,
        'num_threads': 4,
        'num_epochs': 6,
        'num_iterations': 10,
        'depth': 250,
        'height': 90,
        'weight': 140,
        'beam_width': 4,
        'output_dir': '/tmp/lin'
    }
    main(args)
