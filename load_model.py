import tensorflow as tf
from tqdm import tqdm

from input import var_len_train_batch_generator, var_len_val_batch_generator

data_dir = '/home/zyq/dataset/tfrecords'
batch_size = 10
num_threads = 4
NUM_TRAIN_SAMPLE = 28363

def load_model():
    sess = tf.Session()
    X, Y, Y_seq_len = var_len_train_batch_generator(data_dir, batch_size, num_threads)
    # ckpt = tf.train.get_checkpoint_state('/home/lin/lipreading-master/attention2017:12:27:09:45:10/')
    # print(ckpt.model_checkpoint_path)
    saver = tf.train.import_meta_graph('/home/lin/lipreading-master/attention2017:12:27:09:45:10/ckp1.meta')
    saver.restore(sess, tf.train.latest_checkpoint('/home/lin/lipreading-master/attention2017:12:27:09:45:10/'))

    # train = tf.get_collection('train')
    # if NUM_TRAIN_SAMPLE % 10 == 0:
    #     num_iteration = NUM_TRAIN_SAMPLE // 10
    # else:
    #     num_iteration = NUM_TRAIN_SAMPLE // 10 + 1
    # for epoch in range(5):
    #     print('[Epoch %d] begin ' % epoch)
    #     for i in tqdm(range(num_iteration)):
    #         loss = train
    #         print('\n   [%d ] Loss: %.4f' % (i, loss))
    #     print('[Epoch %d] end ' % epoch)
    x = tf.get_default_graph().get_tensor_by_name('conv1/kernel:0')
    # print(sess.run(x))
    print()
    var = tf.trainable_variables()
    for v in var:
        print(v)
    loss = tf.get_collection('loss')
    print(loss)
if __name__ == '__main__':
    load_model()
