import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from resnet_attention.model import Lipreading as Model
from resnet_attention.input import Vocabulary, build_dataset
import datetime
from resnet_attention.configuration import ModelConfig, TrainingConfig
import numpy as np
from statistic import cer_s


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('vocab_path', '/home/zyq/dataset/ST-0/tfrecords/val_1and2/word_counts.txt', 'dictionary path')

tf.flags.DEFINE_integer('NUM_EPOCH', 100, 'epoch次数')

tf.flags.DEFINE_string('input_file', '/home/zyq/dataset/ST-0/tfrecords/val_1and2', 'tfrecords路径')

tf.flags.DEFINE_string('checkpoint_dir', '/home/zyq/codes/lipreading/resnet_attention/attention2018:01:30:21:36:59',
                       '最近一次模型保存文件')

def main(unused_argv):

    model_dir = 'attention' + datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    model_name = 'ckp'
    model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    vocab = Vocabulary(FLAGS.vocab_path)

    model_config = ModelConfig()
    train_config = TrainingConfig()
    model_config.input_file = FLAGS.input_file

    train_dataset = build_dataset(model_config.train_tfrecord_list, batch_size=model_config.batch_size)
    val_dataset = build_dataset(model_config.val_tfrecord_list, batch_size=model_config.batch_size)
    iterator = tf.contrib.data.Iterator.from_structure(val_dataset.output_types, val_dataset.output_shapes)
    train_init_op  = iterator.make_initializer(train_dataset)
    val_init_op = iterator.make_initializer(val_dataset)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 24
    config.inter_op_parallelism_threads = 24
    model = Model(model_config=model_config, iterator=iterator, train_config=train_config, word2idx=vocab.word_to_id)

    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, model_path)


    print('—*-*-*-*-*-*-*-Model compiled-*-*-*-*-*-*-*-')

    val_total_loss = 0
    sess.run(train_init_op)
    val_pairs = []
    i = 0
    while True:
        try:
            out_indices, loss, y = model.eval(sess)
            val_total_loss += loss
            print('\n   [%d ] Loss: %.4f' % (i, loss))
            for j in range(len(y)):
                unpadded_out = None
                if 1 in out_indices[j]:
                    idx_1 = np.where(out_indices[j] == 1)[0][0]
                    unpadded_out = out_indices[j][:idx_1]
                else:
                    unpadded_out = out_indices[j]
                idx_1 = np.where(y[j] == 1)[0][0]
                unpadded_y = y[j][:idx_1]
                predic = ''.join([vocab.id_to_word[k] for k in unpadded_out])
                label = ''.join([vocab.id_to_word[i] for i in unpadded_y])
                val_pairs.append((predic, label))
            i += 1
        except:
            break
    avg_loss = val_total_loss / i
    counts, cer = cer_s(val_pairs)

    print('Average loss is : % .4f' % avg_loss)
    print('Current error rate is : %.4f' % cer)


if __name__ == '__main__':
    tf.app.run()
