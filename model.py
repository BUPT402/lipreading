# encoding = utf-8
from tensorflow.python.layers import core as core_layers
import tensorflow as tf
import  numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

class Image2Seq:
    def __init__(self, depth, img_height, img_width, word2idx,  beam_width, keep_prob, img_ch=3,
                 embedding_dim=70,
                 hidden_size=512,
                 batch_size=30,
                 n_layers=4,
                 grad_clip=5):
        self.force_teaching_ratio = 0.8
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
        
        self.build_graph()

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

    def add_input_layer(self):
        with tf.name_scope('input'):
            # self.video = tf.placeholder(tf.float32, [None, self.depths, self.img_width, self.img_height, self.image_ch])
            # self.video_mask = tf.placeholder(tf.int32, [None, self.depths])
            # self.caption = tf.placeholder(tf.int32, [None, 25, 773])
            # self.caption_mask = tf.placeholder(tf.int32, [None, 25])
            self.X = tf.placeholder(tf.float32, [None, self.depths, self.img_height, self.img_width, self.image_ch])
            self.Y = tf.placeholder(tf.int32, [None, None])
            # self.X=None
            # self.Y=None
            self.Y_seq_len = tf.placeholder(tf.int32, [None])
            self.train_flag = tf.placeholder(tf.bool)

    def add_encode_layer(self):
        with tf.name_scope('conv1'):
            conv1 = tf.layers.conv3d(self.X, 32, [3, 5, 5], [1, 2, 2], padding='same',
                                     use_bias=True, kernel_initializer=tf.truncated_normal_initializer, name='conv1')
            batch1 = tf.layers.batch_normalization(conv1, axis=-1)
            relu1 = tf.nn.relu(batch1)
            drop1 = tf.nn.dropout(relu1, self.keep_prob)
            maxp1 = tf.layers.max_pooling3d(drop1, [1, 2, 2], [1, 2, 2], padding='valid', name='maxpooling1')
            print(maxp1)

        with tf.name_scope('conv2'):
            conv2 = tf.layers.conv3d(maxp1, 64, [3, 5, 5], [1, 1, 1], padding='same',
                                     use_bias=True, kernel_initializer=tf.truncated_normal_initializer, name='conv2')
            batch2 = tf.layers.batch_normalization(conv2, axis=2)
            relu2 = tf.nn.relu(batch2)
            drop2 = tf.nn.dropout(relu2, self.keep_prob)
            maxp2 = tf.layers.max_pooling3d(drop2, [1, 2, 2], [1, 2, 2], padding='valid', name='maxpooling2')
            print(maxp2)

        with tf.name_scope('conv3'):
            conv3 = tf.layers.conv3d(maxp2, 96, [3, 3, 3], [1, 1, 1], padding='same',
                                     use_bias=False, kernel_initializer=tf.truncated_normal_initializer, name='conv3')
            batch3 = tf.layers.batch_normalization(conv3, axis=-1)
            relu3 = tf.nn.relu(batch3)
            drop3 = tf.nn.dropout(relu3, self.keep_prob)
            maxp3 = tf.layers.max_pooling3d(drop3, [1, 2, 2], [1, 2, 2], padding='valid', name='maxpooling2')
            print(maxp3)
            resh = tf.reshape(maxp3, [-1,  250, 8 * 5 * 96])

        with tf.name_scope('GRU'):
            cells_fw = [tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer),
                        tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer)]
            cells_bw = [tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer),
                        tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer)]
            # encode_out=[batch_size, max_time...]
            encode_out, enc_fw_state, enc_bw_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw,
                                                                    cells_bw, resh, dtype=tf.float32)
        #     # self.encoder_out = encode_out  # (?, 250, 512)
        #     # self.encoder_state = tf.concat([enc_fw_state[1], enc_bw_state[1]], 1)  # (?, 512)
        #     # print(self.encoder_out.shape)
            self.encoder_out = encode_out
            proj = tf.concat([enc_fw_state[1], enc_bw_state[1]], 1)
            self.encoder_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(c=proj, h=proj) for _ in range(self.n_layers)])

    def lstm_cell(self, reuse=False):
        return tf.nn.rnn_cell.LSTMCell(self.hidden_size, initializer=tf.orthogonal_initializer(), reuse=reuse)

    def add_attention_for_training(self):
        # # attention机制使用的是LuongAttention, num_units表示attention机制的深度，memory通常是RNN encoder的输入
        # atten_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.hidden_size, memory=self.encoder_out)
        # # cell an instance of RNNcell atte_layer_size代表attention layer输出层的大小，if None表示无attention机制，直接将encode输出输入到decode中
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units=self.hidden_size,
            memory=self.encoder_out)
        self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell=tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.n_layers)]),
            # cell=tf.nn.rnn_cell.GRUCell(512),
            attention_mechanism=attention_mechanism,
            attention_layer_size=self.hidden_size)

    def add_decoder_for_training(self):
        self.add_attention_for_training()
        decoder_embedding = tf.get_variable('decode_embedding', [len(self.word2idx), self.embedding_dim],
                                            dtype=tf.float32)
        # print(decoder_embedding)
        # inputs为实际的label, sequence_length为当前batch中每个序列的长度 ，timemajor=false时,[batch_size,sequence_length,embedding_size]
        # print("-------",self.processed_decoder_input()[0])
        training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs=tf.nn.embedding_lookup(decoder_embedding, self.processed_decoder_input()),
            sequence_length=self.Y_seq_len - 1,
            embedding=decoder_embedding,
            sampling_probability=1 - self.force_teaching_ratio,
            time_major=False)
        # initial_state 直接将encoder的final_state作为该参数的输入即可
        # print('self.encoder_state', self.encoder_state)
        # print("self.decoder_cell", self.decoder_cell)
        # print(self.decoder_cell)
        # training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.decoder_cell, helper=training_helper,
        #                                                    initial_state=
        #                                                    self.decoder_cell.zero_state(self.batch_size,
        #                                                                                 tf.float32).clone(
        #                                                        cell_state=self.encoder_state))
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
        # 训练结果
        self.training_logits = training_decoder_output.rnn_output

    def add_attention_for_inference(self):
        self.encoder_out_tiled = tf.contrib.seq2seq.tile_batch(self.encoder_out, self.beam_width)
        self.encoder_state_tiled = tf.contrib.seq2seq.tile_batch(self.encoder_state, self.beam_width)

        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.hidden_size,
                                                                memory=self.encoder_out_tiled)
        self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
             cell=tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell(reuse=True) for _ in range(self.n_layers)]),
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
        masks = tf.sequence_mask(self.Y_seq_len - 1, tf.reduce_max(self.Y_seq_len - 1), dtype=tf.float32)
        self.loss = tf.contrib.seq2seq.sequence_loss(
            logits=self.training_logits, targets=self.processed_decoder_output(), weights=masks)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
            self.train_op = tf.train.AdamOptimizer().apply_gradients(zip(clipped_gradients, params))

    def partial_fit(self, images, captions, lengths):
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        tf.train.start_queue_runners(sess=self.sess)
        images, captions, lengths = self.sess.run([images, captions, lengths])
        _, loss = self.sess.run([self.train_op, self.loss],
                                {self.X: images, self.Y: captions, self.Y_seq_len: lengths, self.train_flag: True})
        # _, loss = self.sess.run([self.train_op, self.loss],
        #                         [images, captions, lengths, True])

        return loss

    def infer(self, image, idx2word):
        idx2word[-1] = '-1'
        out_indices = self.sess.run(self.predicting_ids,
                                    {self.X: image, self.Y_seq_len: [20], self.train_flag: False})[0]
        print('{}'.format(' '.join([idx2word[i] for i in out_indices])))

    def processed_decoder_input(self):

        return tf.strided_slice(self.Y, [0, 0], [self.batch_size, -1], [1, 1])  # remove last char

    def processed_decoder_output(self):
        return tf.strided_slice(self.Y, [0, 1], [self.batch_size, tf.shape(self.Y)[1]], [1, 1])  # remove first char


if __name__ == '__main__':
    pass