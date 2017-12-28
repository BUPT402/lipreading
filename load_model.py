import tensorflow as tf
import datetime
import os
from tqdm import tqdm
from model import Lipreading as Model
from input import Vocabulary, var_len_train_batch_generator


#
# def load_model():
#     data_dir = '/home/zyq/dataset/tfrecords'
#     batch_size = 10
#     num_threads = 4
#     sess = tf.Session()
#     X, Y, Y_seq_len = var_len_train_batch_generator(data_dir, batch_size, num_threads)
#     ckpt = tf.train.get_checkpoint_state('/home/lin/lipreading-master/attention2017:12:27:09:45:10/')
#     print(ckpt.model_checkpoint_path)
#     saver = tf.train.import_meta_graph('/home/lin/lipreading-master/attention2017:12:27:09:45:10/ckp1.meta')
#     saver.restore(sess, tf.train.latest_checkpoint('/home/lin/lipreading-master/attention2017:12:27:09:45:10/'))
#
#
#     #     print('[Epoch %d] end ' % epoch)
#     var = tf.trainable_variables()
#     for v in var:
#         print(v)
#     x = tf.get_default_graph().get_tensor_by_name('decode/decoder/dense/kernel:0')
#     print(sess.run(x))
#
#     # graph = tf.get_default_graph()
#     # for op in graph.get_operations():
#     #     if op.name == 'loss':
#     #         print(op.name, op.values())
#
#     print('------------')
#
#     # loss = tf.get_collection('lss')
#     # print(loss)
# if __name__ == '__main__':
#     load_model()

NUM_TRAIN_SAMPLE = 28363


def main(args):
    ckpt = tf.train.get_checkpoint_state('/home/lin/lipreading-master/attention2017:12:27:21:57:06/')
    print(ckpt.model_checkpoint_path)
    model_dir = 'attention' + datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    model_name = 'ckp'

    vocab = Vocabulary(args['vocab_path'])

    model = Model(data_dir=args['data_dir'], word2idx=vocab.word_to_id, depth=args['depth'], img_height=args['height'],
                  img_width=args['weight'], beam_width=args['beam_width'],
                  batch_size=args['batch_size'])

    model.sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(model.sess, ckpt.model_checkpoint_path)
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
                print('\n   [%d ] Loss: %.4f' % (i, loss))
                if i % 200 == 0:
                    summary = model.merged_summary()
                    summary_writer.add_summary(summary, i)
                    # saver.save(model.sess, os.path.join(model_dir, model_name + str(epoch)))
                    # cer = model.eval(vocab.id_to_word)
                    # print('Epoch %d cer: %.4f' % (epoch, cer))
            print('[Epoch %d] end ' % epoch)
            saver.save(model.sess, os.path.join(model_dir, model_name + str(epoch)))
            print('Epoch %d saved' % epoch)
            cer = model.eval(vocab.id_to_word)
            print('Epoch %d cer: %.4f' % (epoch, cer))

            summary_writer.close()

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    args = {
        'vocab_path': '/home/zyq/dataset/word_counts.txt',
        'data_dir': '/home/zyq/dataset/tfrecords',
        'batch_size': 10,
        'num_threads': 4,
        'num_epochs': 5,
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