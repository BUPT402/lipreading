from tensorflow.python.layers import core as core_layers
import tensorflow as tf
import  numpy as np
from tqdm import tqdm
from statistic import cer_s
from input import prefetch_input_data
import datetime

NUM_VAL_SAMPLE = 3151

class Lipreading(object):

    def __init__(self, model_config, train_config, mode, word2idx, sess=tf.Session()):

        self.reader = tf.TFRecordReader()
        self.config = model_config
        self.train_config = train_config
        self.mode = mode

        self.sess = sess

        self.word2idx = word2idx

        self.image_seqs = None

        self.label_seqs = None

        self.label_length = None

        self.decode_embedding = None

        self.seq_embedding = None

        self.loss = None

        self.video_feature = None

        self.attention_mechanism = None

        self.decoder_cell = None

        self.training_helper = None

        self.training_decoder = None

        self.training_decoder_output = None


        self.build(train_config)

        # if mode == 'train':
        self.summary_writer = tf.summary.FileWriter('logs_train/log' + datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S'),
                                                        graph=self.sess.graph)
        # elif mode == 'eval':
        #     self.summary_writer = tf.summary.FileWriter(
        #         'logs_eval/log' + datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S'),
        #         graph=self.sess.graph)

    def is_training(self):
        return self.mode == 'train'

    def build(self, train_config):
        self.build_inputs()
        self.build_conv3d()
        self.buils_encode()
        self.build_decode()
        self.build_train(train_config.clip_gradients, train_config.initial_learning_rate)
        self.train_summary = tf.summary.merge_all('train')
        self.eval_summary = tf.summary.merge_all('eval')

    def build_inputs(self):
        if self.mode == 'inference':
            self.image_seqs = tf.placeholder(tf.float32, [None,
                                                          self.config.image_depth,
                                                          self.config.image_weight,
                                                          self.config.image_height,
                                                          self.config.image_channel])
            self.label_seqs = tf.placeholder(tf.int32, [None, None])
            self.label_length = tf.placeholder(tf.int32, [None])
        else:
            with tf.name_scope('input'):
                self.image_seqs, self.label_seqs, self.label_length = prefetch_input_data(reader=self.reader,
                                                                     data_dir=self.config.input_file,
                                                                     batch_size=self.config.batch_size,
                                                                     is_training=self.is_training(),
                                                                     num_threads=self.config.num_threads,)

    def build_conv3d(self):
        with tf.variable_scope('conv3d') as scope:
            with tf.variable_scope('conv1'):
                conv1 = tf.layers.conv3d(self.image_seqs, 32, [3, 5, 5], [1, 2, 2], padding='same',
                                         use_bias=True, kernel_initializer=tf.truncated_normal_initializer, name='conv1')
                if self.is_training():
                    batch1 = tf.layers.batch_normalization(conv1, axis=-1, training=True, name='bn1')
                else:
                    batch1 = tf.layers.batch_normalization(conv1, axis=-1, training=False, name='bn1')
                relu1 = tf.nn.relu(batch1, name='relu1')
                if self.is_training():
                    drop1 = tf.nn.dropout(relu1, self.config.conv_dropout_keep_prob, name='drop1')
                    maxp1 = tf.layers.max_pooling3d(drop1, [1, 2, 2], [1, 2, 2], padding='valid', name='maxpooling1')
                else:
                    maxp1 = tf.layers.max_pooling3d(relu1, [1, 2, 2], [1, 2, 2], padding='valid', name='maxpooling1')
                tf.summary.histogram('conv1', conv1, collections=['train'])
                tf.summary.histogram('activations_1', maxp1, collections=['train'])

            with tf.variable_scope('conv2'):
                conv2 = tf.layers.conv3d(maxp1, 64, [3, 5, 5], [1, 1, 1], padding='same',
                                         use_bias=True, kernel_initializer=tf.truncated_normal_initializer, name='conv2')
                if self.is_training():
                    batch2 = tf.layers.batch_normalization(conv2, axis=-1, training=True, name='bn2')
                else:
                    batch2 = tf.layers.batch_normalization(conv2, axis=-1, training=False, name='bn2')
                relu2 = tf.nn.relu(batch2, name='relu2')
                if self.is_training():
                    drop2 = tf.nn.dropout(relu2, self.config.conv_dropout_keep_prob, name='drop2')
                    maxp2 = tf.layers.max_pooling3d(drop2, [1, 2, 2], [1, 2, 2], padding='valid', name='maxpooling2')
                else:
                    maxp2 = tf.layers.max_pooling3d(relu2, [1, 2, 2], [1, 2, 2], padding='valid', name='maxpooling2')
                tf.summary.histogram('conv2', conv2, collections=['train'])
                tf.summary.histogram('activations_2', maxp2, collections=['train'])

            with tf.variable_scope('conv3'):
                conv3 = tf.layers.conv3d(maxp2, 96, [3, 3, 3], [1, 1, 1], padding='same',
                                         use_bias=False, kernel_initializer=tf.truncated_normal_initializer, name='conv3')
                if self.is_training():
                    batch3 = tf.layers.batch_normalization(conv3, axis=-1, training=True, name='bn3')
                else:
                    batch3 = tf.layers.batch_normalization(conv3, axis=-1, training=False, name='bn3')
                relu3 = tf.nn.relu(batch3, name='relu3')
                if self.is_training():
                    drop3 = tf.nn.dropout(relu3, self.config.conv_dropout_keep_prob, name='drop3')
                    maxp3 = tf.layers.max_pooling3d(drop3, [1, 2, 2], [1, 2, 2], padding='valid', name='maxpooling3')
                else:
                    maxp3 = tf.layers.max_pooling3d(relu3, [1, 2, 2], [1, 2, 2], padding='valid', name='maxpooling3')
                tf.summary.histogram('conv3', conv3, collections=['train'])
                tf.summary.histogram('activations_3', maxp3, collections=['train'])
                self.video_feature = tf.reshape(maxp3, [-1, 250, 8 * 5 * 96])

    def buils_encode(self):
        with tf.variable_scope('encoder') as scope:
            with tf.name_scope('stack_bigru'):
                cells_fw = [tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer),
                            tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer)]
                cells_bw = [tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer),
                            tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer)]
                if self.is_training():
                    for cell in cells_fw:
                        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                             input_keep_prob=self.config.gru_dropout_keep_prob,
                                                             output_keep_prob=self.config.gru_dropout_keep_prob)
                    for cell in cells_bw:
                        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                             input_keep_prob=self.config.gru_dropout_keep_prob,
                                                             output_keep_prob=self.config.gru_dropout_keep_prob)
                encode_out, enc_fw_state, enc_bw_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw,
                                                                                                        cells_bw,
                                                                                                        self.video_feature,
                                                                                                        dtype=tf.float32,
                                                                                                        scope=scope)
                self.encoder_out = encode_out
                tf.summary.histogram('encoder_out', self.encoder_out, collections=['train'])
                state_0 = tf.concat([enc_fw_state[0], enc_bw_state[0]], 1)
                state_1 = tf.concat([enc_fw_state[1], enc_bw_state[1]], 1)
                self.encoder_state = (state_0, state_1)
                tf.summary.histogram('encoder_state', self.encoder_state, collections=['train'])

    def build_decode(self):
        '''decoder部分

        :return:
        '''
        with tf.variable_scope('decoder') as scope:
            self.decode_embedding = tf.get_variable('decode_embedding',
                                                    [len(self.word2idx), self.config.embedding_size],
                                                    tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
            self.seq_embedding = tf.nn.embedding_lookup(self.decode_embedding, self.processed_decoder_input())

            # attention机制使用的是LuongAttention, num_units表示attention机制的深度，memory通常是RNN encoder的输入
            # atten_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.hidden_size, memory=self.encoder_out)
            # cell an instance of RNNcell atte_layer_size代表attention layer输出层的大小，if None表示无attention机制，直接将encode输出输入到decode中
            self.attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=self.config.hidden_size,
                memory=self.encoder_out)
            self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=tf.nn.rnn_cell.MultiRNNCell([self.gru_cell() for _ in range(self.config.num_layers)]),
                attention_mechanism=self.attention_mechanism,
                attention_layer_size=self.config.hidden_size)

            # inputs为实际的label, sequence_length为当前batch中每个序列的长度 ，timemajor=false时,[batch_size,sequence_length,embedding_size]
            # print("-------",self.processed_decoder_input()[0])
            self.training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                inputs=self.seq_embedding,
                sequence_length=self.label_length - 1,
                embedding=self.decode_embedding,
                sampling_probability=1 - self.config.force_teaching_ratio,
                time_major=False,
                name='traininig_helper')
            self.training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=self.decoder_cell,
                helper=self.training_helper,
                initial_state=self.decoder_cell.zero_state(self.config.batch_size, tf.float32).clone(
                    cell_state=self.encoder_state),
                output_layer=core_layers.Dense(len(self.word2idx)))
            # decoder表示一个decoder实例 ，maxinum_interations表示为最大解码步长，默认为None解码至结束，return(final_outputs,final_state
            # final_sequence_lengths)
            self.training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=self.training_decoder,
                                                                              impute_finished=True,
                                                                              maximum_iterations=tf.reduce_max(
                                                                                  self.label_length - 1),
                                                                              scope=scope)
            # print('train_decoder_output:', training_decoder_output)
            # 训练结果
            with tf.variable_scope('logits'):
                self.training_logits = self.training_decoder_output.rnn_output  # [10, ?, 1541]
                tf.summary.histogram('training_logits', self.training_logits, collections=['train'])

        with tf.variable_scope('decoder', reuse=True) as scope:
            self.encoder_out_tiled = tf.contrib.seq2seq.tile_batch(self.encoder_out, self.config.beam_width)
            self.encoder_state_tiled = tf.contrib.seq2seq.tile_batch(self.encoder_state, self.config.beam_width)

            self.attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.config.hidden_size,
                                                                    memory=self.encoder_out_tiled)
            self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=tf.nn.rnn_cell.MultiRNNCell([self.gru_cell(reuse=True) for _ in range(self.config.num_layers)]),
                attention_mechanism=self.attention_mechanism, attention_layer_size=self.config.hidden_size)

            self.predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=self.decoder_cell, embedding=tf.get_variable('decode_embedding'),
                start_tokens=tf.tile(tf.constant([self.word2idx['<BOS>']], dtype=tf.int32), [self.config.batch_size]),
                end_token=self.word2idx['<EOS>'],
                initial_state=self.decoder_cell.zero_state(self.config.batch_size * self.config.beam_width, tf.float32).clone(
                    cell_state=self.encoder_state_tiled),
                beam_width=self.config.beam_width,
                output_layer=core_layers.Dense(len(self.word2idx), _reuse=True),
                length_penalty_weight=0.0)
            self.predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=self.predicting_decoder,
                impute_finished=False,
                maximum_iterations=2 * tf.reduce_max(self.label_length - 1),
                scope=scope)
            with tf.variable_scope('pre_result'):
                # self.predicting_ids = predicting_decoder_output.sample_id
                self.predicting_ids = self.predicting_decoder_output.predicted_ids[:, :, 0]

    def lstm_cell(self, reuse=False):
        return tf.nn.rnn_cell.LSTMCell(self.config.hidden_size, initializer=tf.orthogonal_initializer(), reuse=reuse)

    def gru_cell(self, reuse=False):
        cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size, kernel_initializer=tf.orthogonal_initializer(), reuse=reuse)
        if self.is_training():
            dropout_cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.config.gru_dropout_keep_prob)
            return dropout_cell
        else:
            return cell

    def build_train(self, clip_gradients, learning_rate):
        self.masks = tf.sequence_mask(self.label_length - 1, tf.reduce_max(self.label_length - 1), dtype=tf.float32)   # [?, ?] 动态的掩码
        print(tf.trainable_variables())
        self.l2_losses  = [self.train_config.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'kernel' in v.name]
        self.loss = tf.contrib.seq2seq.sequence_loss(
            logits=self.training_logits, targets=self.processed_decoder_output(), weights=self.masks) + tf.add_n(self.l2_losses)
        tf.summary.scalar('loss', self.loss, collections=['train'])
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            for g in gradients:
                tf.summary.histogram(g.name, g)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_gradients)
            self.train_op = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(clipped_gradients, params))

    def run(self):
        if self.mode == 'train':
            _, loss = self.sess.run([self.train_op, self.loss])
            return loss
        elif self.mode == 'eval':
            pred, loss, label = self.sess.run([self.predicting_ids, self.loss, self.label_seqs])
            return pred, loss, label


    def infer(self, idx2word):
        self.train_flag = True
        idx2word[-1] = '-1'
        out_indices, y = self.sess.run([self.predicting_ids, self.Y])
        for j in range(len(y)):
            print('{}'.format(' '.join([idx2word[i] for i in out_indices[j]])))
            print('{}'.format(' '.join([idx2word[i] for i in y[j]])))

    def merged_summary(self):
        if self.is_training():
            summary = self.sess.run(self.train_summary)
        else:
            summary = self.sess.run(self.eval_summary)
        return summary

    def processed_decoder_input(self):
        return tf.strided_slice(self.label_seqs, [0, 0], [self.config.batch_size, -1], [1, 1])  # remove last char

    def processed_decoder_output(self):
        return tf.strided_slice(self.label_seqs, [0, 1], [self.config.batch_size, tf.shape(self.label_seqs)[1]], [1, 1])  # remove first char


if __name__ == '__main__':
    pass
