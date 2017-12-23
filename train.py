# encoding = utf-8
import tensorflow as tf
from model import Lipreading as Model
from input import Vocabulary


def main(args):
    vocab = Vocabulary(args['vocab_path'])

    # data_loader = var_len_train_batch_generator(args['data_dir'], args['batch_size'], args['num_threads'])

    model = Model(datadir=args['data_dir'], word2idx=vocab.word_to_id, depth=args['depth'], img_height=args['height'],
                  img_width=args['weight'], beam_width=args['beam_width'],
                  batch_size=args['batch_size'])

    model.sess.run(tf.global_variables_initializer())
    print('Model compiled')

    for epoch in range(args['num_epochs']):
        print('[Epoch %d] begin: ' % epoch)
        for i in range(args['num_iterations']):
            # loss = model.partial_fit(data_loader)
            loss = model.partial_fit()
            print('[ batch %d ] Loss: %.4f' % (i, loss))


if __name__ == '__main__':
    args = {
        'vocab_path': '/home/zyq/dataset/word_counts.txt',
        'data_dir': '/home/zyq/dataset/tfrecords',
        'batch_size': 10,
        'num_threads': 4,
        'num_epochs': 10,
        'num_iterations': 100,
        'depth': 250,
        'height': 90,
        'weight': 140,
        'beam_width': 4,
        'output_dir': '/tmp/lin'
    }
    main(args)
