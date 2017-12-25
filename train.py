import tensorflow as tf
from model import Lipreading as Model
from input import Vocabulary, load_video
import datetime
from tqdm import tqdm
import os

NUM_TRAIN_SAMPLE = 28363


def main(args):
    model_dir = 'attention' + datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    model_name = 'ckp'

    vocab = Vocabulary(args['vocab_path'])

    # data_loader = var_len_train_batch_generator(args['data_dir'], args['batch_size'], args['num_threads'])

    model = Model(data_dir=args['data_dir'], word2idx=vocab.word_to_id, depth=args['depth'], img_height=args['height'],
                  img_width=args['weight'], beam_width=args['beam_width'],
                  batch_size=args['batch_size'])

    model.sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    print('—*-*-*-*-*-*-*-Model compiled-*-*-*-*-*-*-*-')

    if NUM_TRAIN_SAMPLE % args['batch_size'] == 0:
        num_iteration = NUM_TRAIN_SAMPLE // args['batch_size']
    else:
        num_iteration = NUM_TRAIN_SAMPLE // args['batch_size'] + 1
    with tf.device('/device:GPU:0'):
        for epoch in range(args['num_epochs']):
            print('[Epoch %d] begin ' % epoch)

            for i in tqdm(range(num_iteration)):
                loss = model.partial_fit()
                print('\n   [%d ] Loss: %.4f' % (i, loss))

            print('[Epoch %d] end ' % epoch)
            model.infer(vocab.id_to_word)
            saver.save(model.sess, os.path.join(model_dir, model_name + str(epoch)))


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
        'output_dir': '/tmp/lipreading',
        'sample_video': '/home/zyq/dataset/video_frames/train_set/0004001',
        'log_dir': '/tmp/lipreading'
    }
    main(args)
