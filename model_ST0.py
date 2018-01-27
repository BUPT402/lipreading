from tensorflow.python.layers import core as core_layers
import tensorflow as tf
import  numpy as np
from tqdm import tqdm
from statistic import cer_s
from input_ST0 import build_dataset
import datetime

NUM_VAL_SAMPLE = 3151


class Lipreading(object):

    def __init__(self, model_config, iterator, train_config, word2idx, sess=tf.Session()):

        self.config = model_config
        self.train_config = train_config
        self.is_training = tf.placeholder_with_default(True, [])
        self.dropout_prob = tf.placeholder_with_default(1.0, [])

        self.iterator = iterator

        self.train_dataset = build_dataset(model_config.train_tfrecord_list, batch_size=model_config.batch_size, shuffle=True)
        self.val_dataset = build_dataset(model_config.val_tfrecord_list, batch_size=model_config.batch_size, shuffle=False,
                                         is_training=False)
        self.train_init_op = iterator.make_initializer(self.train_dataset)
        self.val_init_op = iterator.make_initializer(self.val_dataset)

        self.sess = sess

        self.word2idx = word2idx

        self.build(train_config)

        self.summary_writer = tf.summary.FileWriter('logs_eval/' +
                                                    datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S'),
                                                    graph=self.sess.graph)

    def build(self, train_config):
            self.build_inputs()
            self.build_conv3d()
            self.build_encode()
            self.build_decode_for_train()
            self.build_decode_for_infer()
            self.build_train(train_config.max_gradient_norm, train_config.learning_rate)
            self.merged_summary = tf.summary.merge_all('train')


    def build_inputs(self):
        # if self.mode == 'inference':
        # self.image_seqs = tf.placeholder(tf.float32, [None,
        #                                               self.config.image_depth,
        #                                               self.config.image_weight,
        #                                               self.config.image_height,
        #                                               self.config.image_channel])
        # self.label_seqs = tf.placeholder(tf.int32, [None, None])
        # self.label_length = tf.placeholder(tf.int32, [None])
        # else:
        with tf.name_scope('input'), tf.device('/cpu: 0'):
            self.image_seqs, self.tgt_in, self.tgt_out, self.label_length = self.iterator.get_next()


    def build_conv3d(self):
        with tf.variable_scope('conv3d') as scope:
            with tf.variable_scope('conv1'):
                conv1 = tf.layers.conv3d(self.image_seqs, 32, [3, 5, 5], [1, 2, 2], padding='same',
                                         use_bias=True, name='conv1')
                batch1 = tf.layers.batch_normalization(conv1, axis=-1, training=self.is_training, name='bn1')
                relu1 = tf.nn.relu(batch1, name='relu1')
                drop1 = tf.nn.dropout(relu1, self.dropout_prob, name='drop1')
                maxp1 = tf.layers.max_pooling3d(drop1, [1, 2, 2], [1, 2, 2], padding='valid', name='maxpooling1')
                tf.summary.histogram('conv1', conv1, collections=['train'])
                tf.summary.histogram('activations_1', maxp1, collections=['train'])

            with tf.variable_scope('conv2'):
                conv2 = tf.layers.conv3d(maxp1, 64, [3, 5, 5], [1, 1, 1], padding='same',
                                         use_bias=True, name='conv2')
                batch2 = tf.layers.batch_normalization(conv2, axis=-1, training=self.is_training, name='bn2')
                relu2 = tf.nn.relu(batch2, name='relu2')
                drop2 = tf.nn.dropout(relu2, self.dropout_prob, name='drop2')
                maxp2 = tf.layers.max_pooling3d(drop2, [1, 2, 2], [1, 2, 2], padding='valid', name='maxpooling2')
                tf.summary.histogram('conv2', conv2, collections=['train'])
                tf.summary.histogram('activations_2', maxp2, collections=['train'])

            with tf.variable_scope('conv3'):
                conv3 = tf.layers.conv3d(maxp2, 96, [3, 3, 3], [1, 1, 1], padding='same',
                                         use_bias=True, name='conv3')
                batch3 = tf.layers.batch_normalization(conv3, axis=-1, training=self.is_training, name='bn3')
                relu3 = tf.nn.relu(batch3, name='relu3')
                drop3 = tf.nn.dropout(relu3, self.dropout_prob, name='drop3')
                maxp3 = tf.layers.max_pooling3d(drop3, [1, 2, 2], [1, 2, 2], padding='valid', name='maxpooling3')
                tf.summary.histogram('conv3', conv3, collections=['train'])
                tf.summary.histogram('activations_3', maxp3, collections=['train'])
                self.video_feature = tf.reshape(maxp3, [-1, 77, 8 * 5 * 96])
                tf.summary.histogram('video_feature', self.video_feature, collections=['train'])

    def build_encode(self):
        with tf.variable_scope('encoder') as scope:
                encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=tf.nn.rnn_cell.MultiRNNCell([self.gru_cell() for _ in range(self.config.num_layers)]),
                    cell_bw=tf.nn.rnn_cell.MultiRNNCell([self.gru_cell() for _ in range(self.config.num_layers)]),
                    inputs=self.video_feature, dtype=tf.float32, scope=scope)
                self.encoder_state = tuple([bi_encoder_state[0][1], bi_encoder_state[1][1]])
                self.encoder_out = tf.concat(encoder_outputs, -1)
                tf.summary.histogram('encoder_out', self.encoder_out, collections=['train'])
                tf.summary.histogram('encoder_state', self.encoder_state, collections=['train'])

    def build_decode_for_train(self):
        '''decoder部分

        :return:
        '''
        with tf.variable_scope('decoder') as scope:
            decode_embedding = tf.get_variable('decode_embedding',
                                                    [len(self.word2idx), self.config.embedding_size],
                                                    tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
            seq_embedding = tf.nn.embedding_lookup(decode_embedding, self.tgt_in)

            # attention机制使用的是LuongAttention, num_units表示attention机制的深度，memory通常是RNN encoder的输入
            # atten_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.hidden_size, memory=self.encoder_out)
            # cell an instance of RNNcell atte_layer_size代表attention layer输出层的大小，if None表示无attention机制，直接将encode输出输入到decode中
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=self.config.num_units,
                memory=self.encoder_out)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=tf.nn.rnn_cell.MultiRNNCell([self.gru_cell() for _ in range(self.config.num_layers)]),
                attention_mechanism=attention_mechanism,
                attention_layer_size=self.config.num_units)

            # inputs为实际的label, sequence_length为当前batch中每个序列的长度 ，timemajor=false时,[batch_size,sequence_length,embedding_size]
            # print("-------",self.processed_decoder_input()[0])
            training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                inputs=seq_embedding,
                sequence_length=self.label_length,
                embedding=decode_embedding,
                sampling_probability=1 - self.config.force_teaching_ratio,
                time_major=False,
                name='traininig_helper')
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=training_helper,
                initial_state=decoder_cell.zero_state(self.config.batch_size, tf.float32).clone(
                    cell_state=self.encoder_state),
                output_layer=core_layers.Dense(len(self.word2idx), use_bias=False))
            # decoder表示一个decoder实例 ，maxinum_interations表示为最大解码步长，默认为None解码至结束，return(final_outputs,final_state
            # final_sequence_lengths)
            self.training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                              impute_finished=True,
                                                                              maximum_iterations=tf.reduce_max(
                                                                                  self.label_length),
                                                                              scope=scope)
            # 训练结果
            with tf.variable_scope('logits'):
                self.training_logits = self.training_decoder_output.rnn_output  # [10, ?, 1541]
                tf.summary.histogram('training_logits', self.training_logits, collections=['train'])
                self.sample_id = self.training_decoder_output.sample_id
                tf.summary.histogram('training_sample_id', self.sample_id, collections=['train'])


    def build_decode_for_infer(self):
        with tf.variable_scope('decoder', reuse=True) as scope:
            encoder_out_tiled = tf.contrib.seq2seq.tile_batch(self.encoder_out, self.config.beam_width)
            encoder_state_tiled = tf.contrib.seq2seq.tile_batch(self.encoder_state, self.config.beam_width)

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.config.num_units,
                                                                    memory=encoder_out_tiled)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=tf.nn.rnn_cell.MultiRNNCell([self.gru_cell(reuse=True) for _ in range(self.config.num_layers)]),
                attention_mechanism=attention_mechanism, attention_layer_size=self.config.num_units)

            predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell, embedding=tf.get_variable('decode_embedding'),
                start_tokens=tf.tile(tf.constant([self.word2idx['<BOS>']], dtype=tf.int32), [self.config.batch_size]),
                end_token=self.word2idx['<EOS>'],
                initial_state=decoder_cell.zero_state(self.config.batch_size * self.config.beam_width, tf.float32).clone(
                    cell_state=encoder_state_tiled),
                beam_width=self.config.beam_width,
                output_layer=core_layers.Dense(len(self.word2idx), use_bias=False, _reuse=True),
                length_penalty_weight=0.0)
            self.predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=predicting_decoder,
                impute_finished=False,
                maximum_iterations=2 * tf.reduce_max(self.label_length),
                scope=scope)
            with tf.variable_scope('pre_result'):
                # self.predicting_ids = self.predicting_decoder_output.predicted_ids
                self.predicting_ids = self.predicting_decoder_output.predicted_ids[:, :, 0]

    def lstm_cell(self, reuse=False):
        return tf.nn.rnn_cell.LSTMCell(self.config.num_units, initializer=tf.orthogonal_initializer(), reuse=reuse)

    def gru_cell(self, reuse=False):
        cell = tf.nn.rnn_cell.GRUCell(self.config.num_units, reuse=reuse)
        dropout_cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.dropout_prob)
        return dropout_cell


    def build_train(self, clip_gradients, learning_rate):
        self.masks = tf.sequence_mask(self.label_length, tf.reduce_max(self.label_length), dtype=tf.float32)   # [?, ?] 动态的掩码
        self.l2_losses  = [self.train_config.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'kernel' in v.name]
        self.loss = tf.contrib.seq2seq.sequence_loss(
            # logits=self.training_logits, targets=self.tgt_out, weights=self.masks)
            logits=self.training_logits, targets=self.tgt_out, weights=self.masks) + tf.add_n(self.l2_losses)
        tf.summary.scalar('loss', self.loss, collections=['train'])
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            for grad in gradients:
                tf.summary.histogram(grad.name, grad, collections=['train'])
            clipped_gradients, grad_norm = tf.clip_by_global_norm(gradients, clip_gradients)
            tf.summary.scalar("grad_norm", grad_norm, collections=['train'])
            tf.summary.scalar("learning_rate", self.train_config.learning_rate, collections=['train'])
            self.train_op = tf.train.AdamOptimizer(self.train_config.learning_rate).apply_gradients(
                zip(clipped_gradients, params), global_step=self.train_config.global_step)
            # self.train_op = tf.train.GradientDescentOptimizer(self.train_config.learning_rate).apply_gradients(zip(clipped_gradients, params),
            #                                                                   global_step=self.train_config.global_step)

    def train(self):
        _, loss, sample_id = self.sess.run([self.train_op, self.loss, self.sample_id], feed_dict={self.is_training: True, self.dropout_prob: 0.5})
        return loss, sample_id

    def eval(self):
        pred, loss, label = self.sess.run([self.predicting_ids, self.loss, self.tgt_out],
                                          feed_dict={self.is_training: False, self.dropout_prob: 1.0})
        return pred, loss, label


    def merge(self):
        summary = self.sess.run(self.merged_summary)
        return summary

    # def processed_decoder_input(self):
    #     return tf.strided_slice(self.label_seqs, [0, 0], [self.config.batch_size, -1], [1, 1])  # remove last char
    # #
    # def processed_decoder_output(self):
    #     return tf.strided_slice(self.label_seqs, [0, 1], [self.config.batch_size, tf.shape(self.label_seqs)[1]], [1, 1])  # remove first char


if __name__ == '__main__':
    pass
