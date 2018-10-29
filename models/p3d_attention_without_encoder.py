from tensorflow.python.layers import core as core_layers
import tensorflow as tf


class Lipreading(object):

    def __init__(self, model_config, iterator, train_config, word2idx, lamda=0.5):

        self.config = model_config
        self.train_config = train_config
        self.is_training = tf.placeholder_with_default(True, [])
        self.data_format = "channels_last"

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = tf.train.exponential_decay(train_config.learning_rate,
                                                        self.global_step, 2000, 0.5, staircase=True,
                                                        name='learning_rate')

        self.dropout_prob = tf.placeholder_with_default(1.0, [])

        self.iterator = iterator

        self.word2idx = word2idx
        self.num_class = len(self.word2idx) + 1

        self.lamda = lamda

        self.build_graph()


    def build_graph(self):
            self.build_inputs()
            self.build_p3d()
            self.build_decode_for_train()
            self.build_decode_for_infer()
            self.build_train()
            self.merged_summary = tf.summary.merge_all('train')

    def build_inputs(self):
        with tf.name_scope('input'), tf.device('/cpu: 0'):
            self.image_seqs, self.tgt_in, self.tgt_out, self.labels, self.label_length = self.iterator.get_next()
            self.image_length = tf.placeholder(shape=[], dtype=tf.int32, name='img_len')

    def build_p3d(self, data_format="channels_last"):
        def batch_norm_relu(inputs, is_training, data_format):
            """Performs a batch normalization followed by a ReLU."""
            inputs = tf.layers.batch_normalization(
                inputs=inputs, axis=1 if data_format == 'channels_first' else -1,
                center=True, scale=True, training=is_training, fused=True)
            inputs = tf.nn.relu(inputs)
            return inputs

        def fixed_padding(inputs, kernel_size, data_format):

            pad_total = kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg

            if data_format == 'channels_first':
                padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                                [pad_beg, pad_end], [pad_beg, pad_end]])
            else:
                padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                                [pad_beg, pad_end], [0, 0]])
            return padded_inputs

        def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
            """Strided 2-D convolution with explicit padding."""
            # The padding is consistent and is based only on `kernel_size`, not on the
            # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
            if strides > 1:
                inputs = fixed_padding(inputs, kernel_size, data_format)

            return tf.layers.conv2d(
                inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
                kernel_initializer=tf.variance_scaling_initializer(),
                data_format=data_format)

        def building_block(inputs, filters, is_training, projection_shortcut, strides,
                           data_format):

            shortcut = inputs
            inputs = batch_norm_relu(inputs, is_training, data_format)

            # The projection shortcut should come after the first batch norm and ReLU
            # since it performs a 1x1 convolution.
            if projection_shortcut is not None:
                shortcut = projection_shortcut(inputs)

            inputs = conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=3, strides=strides,
                data_format=data_format)

            inputs = batch_norm_relu(inputs, is_training, data_format)
            inputs = conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=3, strides=1,
                data_format=data_format)

            return inputs + shortcut

        def bottleneck_block(inputs, filters, is_training, projection_shortcut,
                             strides, data_format):
            shortcut = inputs
            inputs = batch_norm_relu(inputs, is_training, data_format)

            # The projection shortcut should come after the first batch norm and ReLU
            # since it performs a 1x1 convolution.
            if projection_shortcut is not None:
                shortcut = projection_shortcut(inputs)

            inputs = conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=1, strides=1,
                data_format=data_format)

            inputs = batch_norm_relu(inputs, is_training, data_format)
            inputs = conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=3, strides=strides,
                data_format=data_format)

            inputs = batch_norm_relu(inputs, is_training, data_format)
            inputs = conv2d_fixed_padding(
                inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
                data_format=data_format)

            return inputs + shortcut

        def block_layer(inputs, filters, block_fn, blocks, strides, is_training, name,
                        data_format):
            # Bottleneck blocks end with 4x the number of filters as they start with
            filters_out = 4 * filters if block_fn is bottleneck_block else filters

            def projection_shortcut(inputs):
                return conv2d_fixed_padding(
                    inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
                    data_format=data_format)

            # Only the first block per block_layer uses projection_shortcut and strides
            inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides,
                              data_format)

            for _ in range(1, blocks):
                inputs = block_fn(inputs, filters, is_training, None, 1, data_format)

            return tf.identity(inputs, name)

        def fixed_padding_s(inputs, kernel_size, data_format):

            pad_total = kernel_size[1] - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg

            if data_format == 'channels_first':
                padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [0, 0],
                                                [pad_beg, pad_end], [pad_beg, pad_end]])
            else:
                padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end],
                                                [pad_beg, pad_end], [0, 0]])
            return padded_inputs

        def fixed_padding_t(inputs, kernel_size, data_format):

            pad_total = kernel_size[0] - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg

            if data_format == 'channels_first':
                padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end],
                                                [0, 0], [0, 0]])
            else:
                padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [0, 0],
                                                [0, 0], [0, 0]])
            return padded_inputs

        def conv3d_fixed_padding_s(inputs, filters, kernel_size, strides, data_format):
            if strides[1] > 1:
                inputs = fixed_padding_s(inputs, kernel_size, data_format)

            return tf.layers.conv3d(
                inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                padding=('SAME' if strides[1] == 1 else 'VALID'), use_bias=False,
                kernel_initializer=tf.variance_scaling_initializer(),
                data_format=data_format)

        def conv3d_fixed_padding_t(inputs, filters, kernel_size, strides, data_format):
            if strides[0] > 1:
                inputs = fixed_padding_t(inputs, kernel_size, data_format)

            return tf.layers.conv3d(
                inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                padding=('SAME' if strides[0] == 1 else 'VALID'), use_bias=False,
                kernel_initializer=tf.variance_scaling_initializer(),
                data_format=data_format)

        def bottleneck_block_3d(inputs, filters, is_training, projection_shortcut,
                                strides, data_format):
            shortcut = inputs
            inputs = batch_norm_relu(inputs, is_training, data_format)

            # The projection shortcut should come after the first batch norm and ReLU
            # since it performs a 1x1 convolution.
            if projection_shortcut is not None:
                shortcut = projection_shortcut(inputs)

            inputs = conv3d_fixed_padding_s(
                inputs=inputs, filters=filters, kernel_size=(1, 1, 1), strides=strides,
                data_format=data_format)

            inputs = batch_norm_relu(inputs, is_training, data_format)
            inputs = conv3d_fixed_padding_s(
                inputs=inputs, filters=filters, kernel_size=(1, 3, 3), strides=(1, 1, 1),
                data_format=data_format)

            inputs = batch_norm_relu(inputs, is_training, data_format)
            inputs = conv3d_fixed_padding_t(
                inputs=inputs, filters=filters, kernel_size=(3, 1, 1), strides=(1, 1, 1),
                data_format=data_format)

            inputs = batch_norm_relu(inputs, is_training, data_format)
            inputs = conv3d_fixed_padding_s(
                inputs=inputs, filters=4 * filters, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                data_format=data_format)

            return inputs + shortcut

        def block_layer_3d(inputs, filters, block_fn, blocks, strides, is_training, name,
                           data_format):
            # Bottleneck blocks end with 4x the number of filters as they start with
            filters_out = 4 * filters if block_fn is bottleneck_block_3d else filters

            def projection_shortcut(inputs):
                return conv3d_fixed_padding_s(
                    inputs=inputs, filters=filters_out, kernel_size=(1, 1, 1), strides=strides,
                    data_format=data_format)

            # Only the first block per block_layer uses projection_shortcut and strides
            inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides,
                              data_format)

            for _ in range(1, blocks):
                inputs = block_fn(inputs, filters, is_training, None, (1, 1, 1), data_format)

            return tf.identity(inputs, name)

        with tf.variable_scope("resnet"):
            block_fn = bottleneck_block_3d
            inputs = self.image_seqs
            is_training = self.is_training

            # res1
            inputs = conv3d_fixed_padding_s(inputs, 64, (1, 3, 3), (1, 2, 2), data_format)
            inputs = batch_norm_relu(inputs, is_training, data_format)
            inputs = tf.layers.max_pooling3d(inputs, (2, 3, 3), (2, 2, 2), padding="valid")

            # res2-4
            inputs = block_layer_3d(
                inputs=inputs, filters=64, block_fn=block_fn, blocks=3,
                strides=(1, 1, 1), is_training=is_training, name='block_layer1',
                data_format=data_format)
            inputs = block_layer_3d(
                inputs=inputs, filters=128, block_fn=block_fn, blocks=4,
                strides=(1, 2, 2), is_training=is_training, name='block_layer2',
                data_format=data_format)
            inputs = block_layer_3d(
                inputs=inputs, filters=256, block_fn=block_fn, blocks=6,
                strides=(1, 2, 2), is_training=is_training, name='block_layer3',
                data_format=data_format)

            inputs = block_layer_3d(
                inputs=inputs, filters=512, block_fn=block_fn, blocks=3,
                strides=(1, 2, 2), is_training=is_training, name='block_layer4',
                data_format=data_format)
            inputs = batch_norm_relu(inputs, is_training, data_format)
            inputs = tf.layers.average_pooling3d(
                inputs=inputs, pool_size=(1, 4, 4), strides=(1, 1, 1), padding='VALID',
                data_format=data_format)
            inputs = tf.identity(inputs, 'final_avg_pool')  # img_length / 2
            self.res_out = tf.reshape(inputs, [-1, self.config.image_depth // 2, 2048])  # [14,1,1,2048]
            tf.summary.histogram("res_out", self.res_out, collections=['train'])


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
                num_units=self.config.num_units * 2,
                memory=self.res_out)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=tf.nn.rnn_cell.MultiRNNCell([self.gru_cell(self.config.num_units * 2) for _ in range(self.config.num_layers)]),
                attention_mechanism=attention_mechanism,
                attention_layer_size=self.config.num_units * 2)

            # inputs为实际的label, sequence_length为当前batch中每个序列的长度 ，timemajor=false时,[batch_size,sequence_length,embedding_size]
            # print("-------",self.processed_decoder_input()[0])
            training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                inputs=seq_embedding,
                sequence_length=self.label_length,
                embedding=decode_embedding,
                sampling_probability=1-self.config.force_teaching_ratio,
                time_major=False,
                name='traininig_helper')
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=training_helper,
                initial_state=decoder_cell.zero_state(self.config.batch_size, tf.float32),
                output_layer=core_layers.Dense(len(self.word2idx), use_bias=False))
            # decoder表示一个decoder实例 ，maxinum_interations表示为最大解码步长，默认为None解码至结束，return(final_outputs,final_state
            # final_sequence_lengths)
            #max_iteration应该改一下
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

    def gru_cell(self, num_units, reuse=False):
        cell = tf.nn.rnn_cell.GRUCell(num_units, reuse=reuse)
        dropout_cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.dropout_prob, self.dropout_prob)
        return dropout_cell

    def build_decode_for_infer(self):
        with tf.variable_scope('decoder', reuse=True) as scope:
            encoder_out_tiled = tf.contrib.seq2seq.tile_batch(self.res_out, self.config.beam_width)

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.config.num_units * 2,
                                                                    memory=encoder_out_tiled)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=tf.nn.rnn_cell.MultiRNNCell([self.gru_cell(self.config.num_units * 2, reuse=True) for _ in range(self.config.num_layers)]),
                attention_mechanism=attention_mechanism, attention_layer_size=self.config.num_units * 2)

            predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell, embedding=tf.get_variable('decode_embedding'),
                start_tokens=tf.tile(tf.constant([self.word2idx['<BOS>']], dtype=tf.int32), [self.config.batch_size]),
                end_token=self.word2idx['<EOS>'],
                initial_state=decoder_cell.zero_state(self.config.batch_size * self.config.beam_width, tf.float32),
                beam_width=self.config.beam_width,
                output_layer=core_layers.Dense(len(self.word2idx), use_bias=False, _reuse=True),
                length_penalty_weight=0.0)
            self.predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=predicting_decoder,
                impute_finished=False,
                maximum_iterations=2 * tf.reduce_max(self.label_length),
                scope=scope)
            with tf.variable_scope('pre_result'):
                self.predicting_ids = self.predicting_decoder_output.predicted_ids[:, :, 0]


    def build_train(self):
        masks = tf.sequence_mask(self.label_length, tf.reduce_max(self.label_length), dtype=tf.float32)   # [?, ?] 动态的掩码
        self.loss = tf.contrib.seq2seq.sequence_loss(
            logits=self.training_logits, targets=self.tgt_out, weights=masks)

        tf.summary.scalar('loss', self.loss, collections=['train'])
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            params = tf.trainable_variables()

            train_params = []
            for param in params:
                if "conv3d" not in param.name or "resnet" not in param.name:
                    train_params.append(param)

            gradients = tf.gradients(self.loss, train_params)

            for grad in gradients:
                tf.summary.histogram(grad.name, grad, collections=['train'])
            clipped_gradients, grad_norm = tf.clip_by_global_norm(gradients, self.train_config.max_gradient_norm)
            tf.summary.scalar("grad_norm", grad_norm, collections=['train'])
            tf.summary.scalar("learning_rate", self.train_config.learning_rate, collections=['train'])
            # self.train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.9).apply_gradients(
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.train_config.learning_rate).apply_gradients(
                zip(clipped_gradients, train_params), global_step=self.global_step)

    def train(self, sess):
        _, loss = sess.run([self.train_op, self.loss], feed_dict={self.is_training: True,
                                                                  self.dropout_prob: self.config.dropout_keep_prob})
        return loss

    def eval(self, sess):
        pred, loss, label = sess.run([self.predicting_ids, self.loss, self.tgt_out],
                                          feed_dict={self.is_training: False})
        return pred, loss, label

    def merge(self, sess):
        summary = sess.run(self.merged_summary)
        return summary

if __name__ == '__main__':
    pass
