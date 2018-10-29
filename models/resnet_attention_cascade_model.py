from tensorflow.python.layers import core as core_layers
import tensorflow as tf


class Lipreading(object):

    def __init__(self, model_config, iterator, train_config, word2idx, lamda=0.5, attention='lunong'):

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

        self.attention = attention

        self.build_graph()


    def build_graph(self):
            self.build_inputs()
            self.build_conv3d()
            self.build_resnet()
            self.build_encoder()
            self.build_attention()
            self.build_infer()
            self.build_train()
            self.merged_summary = tf.summary.merge_all('train')

    def build_inputs(self):
        with tf.name_scope('input'), tf.device('/cpu: 0'):
            self.image_seqs, self.tgt_in, self.tgt_out, self.labels, self.label_length = self.iterator.get_next()
            self.ctc_label_length = self.label_length - 1

    def build_conv3d(self):
        with tf.variable_scope('conv3d') as scope:
            with tf.variable_scope('conv1'):
                conv1 = tf.layers.conv3d(self.image_seqs, 32, [5, 5, 5], [1, 2, 2], padding='same',
                                         use_bias=True, name='conv1')
                batch1 = tf.layers.batch_normalization(conv1, axis=-1, training=self.is_training, name='bn1')
                relu1 = tf.nn.relu(batch1, name='relu1')
                # drop1 = tf.nn.dropout(relu1, self.dropout_prob, name='drop1')
                # maxp1 = tf.layers.max_pooling3d(relu1, [1, 2, 2], [1, 2, 2], padding='same', name='maxpooling1')
                self.video_feature = relu1
                tf.summary.histogram("video_feature", self.video_feature, collections=['train'])

    def build_resnet(self):
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

        with tf.variable_scope("resnet"):
            block_fn = building_block
            data_format = self.data_format
            # res_input = tf.reshape(self.video_feature, [-1, 23, 35, 64])
            res_input = tf.reshape(self.video_feature, [-1, 56, 56, 32])
            inputs = conv2d_fixed_padding(
                inputs=res_input, filters=64, kernel_size=3, strides=1,
                data_format=data_format)
            inputs = tf.identity(inputs, 'initial_conv')
            inputs = tf.layers.max_pooling2d(
                inputs=inputs, pool_size=3, strides=2, padding='SAME',
                data_format=data_format)
            inputs = tf.identity(inputs, 'initial_max_pool')
            inputs = block_layer(
                inputs=inputs, filters=64, block_fn=block_fn, blocks=3,
                strides=1, is_training=self.is_training, name='block_layer1',
                data_format=data_format)
            inputs = block_layer(
                inputs=inputs, filters=128, block_fn=block_fn, blocks=4,
                strides=2, is_training=self.is_training, name='block_layer2',
                data_format=data_format)
            inputs = block_layer(
                inputs=inputs, filters=256, block_fn=block_fn, blocks=6,
                strides=2, is_training=self.is_training, name='block_layer3',
                data_format=data_format)
            inputs = block_layer(
                inputs=inputs, filters=512, block_fn=block_fn, blocks=3,
                strides=2, is_training=self.is_training, name='block_layer4',
                data_format=data_format)

            inputs = batch_norm_relu(inputs, self.is_training, data_format)
            inputs = tf.layers.average_pooling2d(
                inputs=inputs, pool_size=4, strides=1, padding='VALID',
                data_format=data_format)
            inputs = tf.identity(inputs, 'final_avg_pool')
            #固定长度
            self.res_out = tf.reshape(inputs, [-1, self.config.image_depth, 512])
            #变长
            # self.res_out = tf.reshape(inputs, [self.config.batch_size, -1, 512])
            tf.summary.histogram("res_out", self.res_out, collections=['train'])

    def build_encoder(self):
        with tf.variable_scope("encoder") as scope:
            encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.MultiRNNCell([self.gru_cell(self.config.num_units) for _ in range(self.config.num_layers)]),
                cell_bw=tf.nn.rnn_cell.MultiRNNCell([self.gru_cell(self.config.num_units) for _ in range(self.config.num_layers)]),
                inputs=self.res_out, dtype=tf.float32, scope=scope)
            #可以实现变长
                # inputs=self.res_out, dtype=tf.float32, scope=scope, sequence_length=self.image_length)
            encoder_state = []
            for i in range(self.config.num_layers):
                encoder_state.append(tf.concat([bi_encoder_state[0][i], bi_encoder_state[1][i]], axis=-1))
            self.encoder_state = tuple(encoder_state)
            self.encoder_out = tf.concat(encoder_outputs, -1)
            tf.summary.histogram('encoder_out', self.encoder_out, collections=['train'])
            tf.summary.histogram('encoder_state', self.encoder_state, collections=['train'])

    def build_attention(self):
        with tf.variable_scope('attention') as scope:
            if self.attention == 'lunong':
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    num_units=self.config.num_units * 2,
                    memory=self.encoder_out)
            else:
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=self.config.num_units * 2,
                    memory=self.encoder_out
                )
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=tf.nn.rnn_cell.MultiRNNCell([self.gru_cell(self.config.num_units * 2) for _ in range(self.config.num_layers)]),
                attention_mechanism=attention_mechanism,
                attention_layer_size=self.config.num_units * 2)

            attention_outputs, attention_state = tf.nn.dynamic_rnn(
                cell=decoder_cell, inputs=self.encoder_out,
                dtype=tf.float32
            )
            shape = tf.shape(attention_outputs)
            batch_s, max_timesteps = shape[0], shape[1]
            outputs = tf.reshape(attention_outputs, [-1, 512])
            logits = tf.layers.dense(outputs, self.num_class)
            logits = tf.reshape(logits, [batch_s, -1, self.num_class])
            self.logits = tf.transpose(logits, (1, 0, 2))


    def build_infer(self):
        sparse_label = tf.keras.backend.ctc_label_dense_to_sparse(self.labels, self.ctc_label_length)
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(self.logits, tf.constant(self.config.image_depth,
                                                                      shape=[self.config.batch_size]),
                                                                       merge_repeated=False,
                                                                       beam_width=self.config.beam_width)
        self.prediction = tf.cast(decoded[0], tf.int32)
        self.err_rate_train = tf.reduce_mean(tf.edit_distance(self.prediction,
                                                              sparse_label,
                                                              normalize=True))
        tf.summary.scalar('ler_train', self.err_rate_train, collections=['train'])

    def gru_cell(self, num_units, reuse=False):
        cell = tf.nn.rnn_cell.GRUCell(num_units, reuse=reuse)
        dropout_cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.dropout_prob, self.dropout_prob)
        return dropout_cell

    def build_train(self):
        sparse_label = tf.keras.backend.ctc_label_dense_to_sparse(self.labels, self.ctc_label_length)
        ctc_loss = tf.nn.ctc_loss(labels=sparse_label, inputs=self.logits,
                              sequence_length=tf.cast(tf.constant(self.config.image_depth,
                                                                      shape=[self.config.batch_size]), tf.int32),
                              ctc_merge_repeated=False,
                              ignore_longer_outputs_than_inputs=True)
        ctc_loss = tf.reduce_mean(ctc_loss, name="ctc_loss_mean")

        self.loss = ctc_loss
        tf.summary.scalar('loss', self.loss, collections=['train'])
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            params = tf.trainable_variables()

            train_params = []
            for param in params:
                if "conv3d" not in param.name or "resnet" not in param.name:
                    train_params.append(param)

            gradients = tf.gradients(self.loss, params)

            for grad in gradients:
                tf.summary.histogram(grad.name, grad, collections=['train'])
            clipped_gradients, grad_norm = tf.clip_by_global_norm(gradients, self.train_config.max_gradient_norm)
            tf.summary.scalar("grad_norm", grad_norm, collections=['train'])
            tf.summary.scalar("learning_rate", self.train_config.learning_rate, collections=['train'])
            # self.train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.9).apply_gradients(
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.train_config.learning_rate).apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)



    def train(self, sess):
        _, loss, length = sess.run([self.train_op, self.loss, self.label_length], feed_dict={self.is_training: True,
                                                                  self.dropout_prob: self.config.dropout_keep_prob})
        return loss

    def eval(self, sess):
        pred, loss, err = sess.run([self.prediction, self.loss, self.err_rate_train],
                                     feed_dict={self.is_training: False})
        return pred, loss, err

    def merge(self, sess):
        summary = sess.run(self.merged_summary)
        return summary

if __name__ == '__main__':
    pass
