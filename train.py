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

    model = Model(data_dir=args['data_dir'], word2idx=vocab.word_to_id, depth=args['depth'], img_height=args['height'],
                  img_width=args['weight'], beam_width=args['beam_width'],
                  batch_size=args['batch_size'])

    model.sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    summary_writer = model.summary_writer

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(model.sess, coord)

    print('—*-*-*-*-*-*-*-Model compiled-*-*-*-*-*-*-*-')

    if NUM_TRAIN_SAMPLE % args['batch_size'] == 0:
        num_iteration = NUM_TRAIN_SAMPLE // args['batch_size']
    else:
        num_iteration = NUM_TRAIN_SAMPLE // args['batch_size'] + 1
    with tf.device('/device:GPU:0'):
        for epoch in range(args['num_epochs']):
            print('[Epoch %d] begin ' % epoch)
            for i in tqdm(range(num_iteration)):
                loss = model.train()
                # summary, loss = model.partial_fit()
                print('\n   [%d ] Loss: %.4f' % (i, loss))
                if i % 100 == 0:
                    summary = model.merged_summary()
                    summary_writer.add_summary(summary, i)

            print('[Epoch %d] end ' % epoch)
            cer = model.eval(vocab.id_to_word)
            print('Current cer: %.4f' % cer)
            saver.save(model.sess, os.path.join(model_dir, model_name + str(epoch)))
            summary_writer.close()

    coord.request_stop()
    coord.join(threads)

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