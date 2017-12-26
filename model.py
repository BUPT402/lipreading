from tensorflow.python.layers import core as core_layers
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from statistic import cer_s
from input import var_len_train_batch_generator, var_len_val_batch_generator
import datetime

NUM_VAL_SAMPLE = 3151


class Lipreading:
    def __init__(self, data_dir, depth, img_height, img_width, word2idx, batch_size, beam_width=5, keep_prob=0.1,
                 img_ch=3,
                 embedding_dim=256, hidden_size=512, n_layers=2, grad_clip=5, force_teaching_ratio=0.8, num_threads=4,
                 sess=tf.Session()):
        self.force_teaching_ratio = force_teaching_ratio
        self.depths = depth
        self.image_ch = img_ch
        self.img_height = img_height
        self.img_width = img_width
        self.word2idx = word2idx
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.beam_width = beam_width
        self.grad_clip = grad_clip
        self.sess = sess
        self.data_dir = data_dir
        self.train_flag = True
        self.num_threads = num_threads

        self.build_graph()
        self.summary_writer = tf.summary.FileWriter(
            'logs_all/log' + datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S'),
            graph=self.sess.graph)

    def build_graph(self):
        with tf.name_scope('input_layer'):
            self.add_input_layer()
        with tf.name_scope('encode'):
            self.add_encode_layer()
        with tf.variable_scope('decode'):
            self.add_decoder_for_training()
        with tf.variable_scope('decode', reuse=True):
            self.add_decoder_for_inference()
        self.add_backward_path()
        self.summary_op = tf.summary.merge_all()

    def add_input_layer(self):
        if self.train_flag:
            with tf.name_scope('input'):
                self.X, self.Y, self.Y_seq_len = var_len_train_batch_generator(self.data_dir, self.batch_size,
                                                                               self.num_threads)
        else:
            with tf.name_scope('input'):
                self.X, self.Y, self.Y_seq_len = var_len_val_batch_generator(self.data_dir, self.batch_size,
                                                                             self.num_threads)

    def add_encode_layer(self):
        with tf.name_scope('conv1'):
            conv1 = tf.layers.conv3d(self.X, 32, [3, 5, 5], [1, 2, 2], padding='same',
                                     use_bias=True, kernel_initializer=tf.truncated_normal_initializer, name='conv1')
            batch1 = tf.layers.batch_normalization(conv1, axis=-1)
            relu1 = tf.nn.relu(batch1)
            drop1 = tf.nn.dropout(relu1, self.keep_prob)
            maxp1 = tf.layers.max_pooling3d(drop1, [1, 2, 2], [1, 2, 2], padding='valid', name='maxpooling1')
            tf.summary.histogram('conv1', conv1)
            tf.summary.histogram('activations_1', maxp1)

        with tf.name_scope('conv2'):
            conv2 = tf.layers.conv3d(maxp1, 64, [3, 5, 5], [1, 1, 1], padding='same',
                                     use_bias=True, kernel_initializer=tf.truncated_normal_initializer, name='conv2')
            batch2 = tf.layers.batch_normalization(conv2, axis=-1)
            relu2 = tf.nn.relu(batch2)
            drop2 = tf.nn.dropout(relu2, self.keep_prob)
            maxp2 = tf.layers.max_pooling3d(drop2, [1, 2, 2], [1, 2, 2], padding='valid', name='maxpooling2')
            tf.summary.histogram('conv2', conv2)
            tf.summary.histogram('activations_2', maxp2)

        with tf.name_scope('conv3'):
            conv3 = tf.layers.conv3d(maxp2, 96, [3, 3, 3], [1, 1, 1], padding='same',
                                     use_bias=False, kernel_initializer=tf.truncated_normal_initializer, name='conv3')
            batch3 = tf.layers.batch_normalization(conv3, axis=-1)
            relu3 = tf.nn.relu(batch3)
            drop3 = tf.nn.dropout(relu3, self.keep_prob)
            maxp3 = tf.layers.max_pooling3d(drop3, [1, 2, 2], [1, 2, 2], padding='valid', name='maxpooling2')
            tf.summary.histogram('conv3', conv3)
            tf.summary.histogram('activations_3', maxp3)
            resh = tf.reshape(maxp3, [-1, 250, 8 * 5 * 96])

        with tf.name_scope('GRU'):
            cells_fw = [tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer),
                        tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer)]
            cells_bw = [tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer),
                        tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer)]
            encode_out, enc_fw_state, enc_bw_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw,
                                                                                                    cells_bw, resh,
                                                                                                    dtype=tf.float32)
            self.encoder_out = encode_out
            tf.summary.histogram('encoder_out', self.encoder_out)
            state_0 = tf.concat([enc_fw_state[0], enc_bw_state[0]], 1)
            state_1 = tf.concat([enc_fw_state[1], enc_bw_state[1]], 1)
            self.encoder_state = (state_0, state_1)
            tf.summary.histogram('encoder_state', self.encoder_state)

    def lstm_cell(self, reuse=False):
        return tf.nn.rnn_cell.LSTMCell(self.hidden_size, initializer=tf.orthogonal_initializer(), reuse=reuse)

    def gru_cell(self, reuse=False):
        return tf.nn.rnn_cell.GRUCell(self.hidden_size, kernel_initializer=tf.orthogonal_initializer(), reuse=reuse)

    def add_attention_for_training(self):
        # attention机制使用的是LuongAttention, num_units表示attention机制的深度，memory通常是RNN encoder的输入
        # atten_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.hidden_size, memory=self.encoder_out)
        # cell an instance of RNNcell atte_layer_size代表attention layer输出层的大小，if None表示无attention机制，直接将encode输出输入到decode中
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units=self.hidden_size,
            memory=self.encoder_out)
        self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell=tf.nn.rnn_cell.MultiRNNCell([self.gru_cell() for _ in range(self.n_layers)]),
            attention_mechanism=attention_mechanism,
            attention_layer_size=self.hidden_size)

    def add_decoder_for_training(self):
        self.add_attention_for_training()
        decoder_embedding = tf.get_variable('decode_embedding', [len(self.word2idx), self.embedding_dim],
                                            dtype=tf.float32)
        # inputs为实际的label, sequence_length为当前batch中每个序列的长度 ，timemajor=false时,[batch_size,sequence_length,embedding_size]
        # print("-------",self.processed_decoder_input()[0])
        training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs=tf.nn.embedding_lookup(decoder_embedding, self.processed_decoder_input()),
            sequence_length=self.Y_seq_len - 1,
            embedding=decoder_embedding,
            sampling_probability=1 - self.force_teaching_ratio,
            time_major=False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            helper=training_helper,
            initial_state=self.decoder_cell.zero_state(self.batch_size, tf.float32).clone(
                cell_state=self.encoder_state),
            output_layer=core_layers.Dense(len(self.word2idx)))
        # decoder表示一个decoder实例 ，maxinum_interations表示为最大解码步长，默认为None解码至结束，return(final_outputs,final_state
        # final_sequence_lengths)
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                          impute_finished=True,
                                                                          maximum_iterations=tf.reduce_max(
                                                                              self.Y_seq_len - 1))
        # print('train_decoder_output:', training_decoder_output)
        # 训练结果
        self.training_logits = training_decoder_output.rnn_output  # [10, ?, 1541]
        tf.summary.histogram('training_logits', self.training_logits)

    def add_attention_for_inference(self):
        self.encoder_out_tiled = tf.contrib.seq2seq.tile_batch(self.encoder_out, self.beam_width)
        self.encoder_state_tiled = tf.contrib.seq2seq.tile_batch(self.encoder_state, self.beam_width)

        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.hidden_size,
                                                                memory=self.encoder_out_tiled)
        self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell=tf.nn.rnn_cell.MultiRNNCell([self.gru_cell(reuse=True) for _ in range(self.n_layers)]),
            attention_mechanism=attention_mechanism, attention_layer_size=self.hidden_size)

    def add_decoder_for_inference(self):
        self.add_attention_for_inference()
        predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=self.decoder_cell, embedding=tf.get_variable('decode_embedding'),
            start_tokens=tf.tile(tf.constant([self.word2idx['<BOS>']], dtype=tf.int32), [self.batch_size]),
            end_token=self.word2idx['<EOS>'],
            initial_state=self.decoder_cell.zero_state(self.batch_size * self.beam_width, tf.float32).clone(
                cell_state=self.encoder_state_tiled),
            beam_width=self.beam_width,
            output_layer=core_layers.Dense(len(self.word2idx), _reuse=True),
            length_penalty_weight=0.0)
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=predicting_decoder,
            impute_finished=False,
            maximum_iterations=2 * tf.reduce_max(self.Y_seq_len - 1))
        self.predicting_ids = predicting_decoder_output.predicted_ids[:, :, 0]

    def add_backward_path(self):
        masks = tf.sequence_mask(self.Y_seq_len - 1, tf.reduce_max(self.Y_seq_len - 1),
                                 dtype=tf.float32)  # [?, ?] 动态的掩码
        self.loss = tf.contrib.seq2seq.sequence_loss(
            logits=self.training_logits, targets=self.processed_decoder_output(), weights=masks)
        tf.summary.scalar('loss', self.loss)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
            self.train_op = tf.train.AdamOptimizer().apply_gradients(zip(clipped_gradients, params))

    def train(self):
        self.train_flag = True
        self.keep_prob = 0.5
        _, loss = self.sess.run([self.train_op, self.loss])

        return loss

    def eval(self, idx2word):
        self.train_flag = False
        self.keep_prob = 1
        if NUM_VAL_SAMPLE % self.batch_size == 0:
            num_iteration = NUM_VAL_SAMPLE // self.batch_size
        else:
            num_iteration = NUM_VAL_SAMPLE // self.batch_size + 1

        val_pairs = []
        for i in tqdm(range(num_iteration)):
            out_indices, y = self.sess.run([self.predicting_ids, self.Y])
            for j in range(len(y)):
                unpadded_out = None
                if 1 in out_indices[j]:
                    idx_1 = np.where(out_indices[j] == 1)[0][0]
                    unpadded_out = out_indices[j][:idx_1]
                else:
                    unpadded_out = out_indices[j]
                idx_1 = np.where(y[j] == 1)[0][0]
                unpadded_y = y[j][1:idx_1]
                predic = ''.join([idx2word[k] for k in unpadded_out])
                label = ''.join([idx2word[i] for i in unpadded_y])
                val_pairs.append((predic, label))
        count, cer = cer_s(val_pairs)
        tf.summary.scalar('cer', cer)
        return cer

    def infer(self, idx2word):
        self.train_flag = True
        self.keep_prob = 1
        idx2word[-1] = '-1'
        out_indices, y = self.sess.run([self.predicting_ids, self.Y])
        for j in range(len(y)):
            print('{}'.format(' '.join([idx2word[i] for i in out_indices[j]])))
            print('{}'.format(' '.join([idx2word[i] for i in y[j]])))

    def merged_summary(self):
        summary = self.sess.run(self.summary_op)
        return summary

    def processed_decoder_input(self):
        return tf.strided_slice(self.Y, [0, 0], [self.batch_size, -1], [1, 1])  # remove last char

    def processed_decoder_output(self):
        return tf.strided_slice(self.Y, [0, 1], [self.batch_size, tf.shape(self.Y)[1]], [1, 1])  # remove first char


if __name__ == '__main__':
    pass
