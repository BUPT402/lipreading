from tensorflow.python.layers import core as core_layers
import tensorflow as tf
import numpy as np


class Lipreading(object):
    def __init__(self, model_config, iterator, train_config, word2idx):

        self.config = model_config
        self.train_config = train_config
        self.is_training = tf.placeholder_with_default(True, [])
        self.data_format = "channels_last"

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = tf.train.exponential_decay(train_config.learning_rate,
                                                        self.global_step, 10000, 0.8, staircase=True,
                                                        name='learning_rate')

        self.dropout_prob = tf.placeholder_with_default(1.0, [])

        self.iterator = iterator

        self.word2idx = word2idx
        self.num_class = len(self.word2idx) + 1

        self.build_graph()

    def build_graph(self):
        self.build_inputs()
        self.build_p3d()
        self.build_bigru()
        self.build_train()
        self.build_infer()
        self.merged_summary = tf.summary.merge_all('train')

    def build_inputs(self):
        with tf.name_scope('input'), tf.device('/cpu: 0'):
            self.image_seqs, self.tgt_in, self.tgt_out, self.labels, self.label_length = self.iterator.get_next()
            self.ctc_label_length = self.label_length - 1
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

    def build_bigru(self):
        with tf.variable_scope("bi_gru") as scope:
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.MultiRNNCell(
                    [self.gru_cell(self.config.num_units) for _ in range(self.config.num_layers)]),
                cell_bw=tf.nn.rnn_cell.MultiRNNCell(
                    [self.gru_cell(self.config.num_units) for _ in range(self.config.num_layers)]),
                inputs=self.res_out, dtype=tf.float32, scope=scope)
            outputs = tf.concat(outputs, -1)
            shape = tf.shape(outputs)
            batch_s, max_timesteps = shape[0], shape[1]
            outputs = tf.reshape(outputs, [-1, 512])
            logits = tf.layers.dense(outputs, self.num_class)
            logits = tf.reshape(logits, [batch_s, -1, self.num_class])
            self.logits = tf.transpose(logits, (1, 0, 2))

    def build_infer(self):
        sparse_label = tf.keras.backend.ctc_label_dense_to_sparse(self.labels, self.ctc_label_length)
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(self.logits, self.ctc_label_length,
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
        loss = tf.nn.ctc_loss(labels=sparse_label, inputs=self.logits,
                              sequence_length=tf.cast(self.ctc_label_length, tf.int32),
                              ctc_merge_repeated=False,
                              ignore_longer_outputs_than_inputs=True)
        self.loss = tf.reduce_mean(loss, name="ctc_loss_mean")
        tf.summary.scalar('loss', self.loss, collections=['train'])

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            params = tf.trainable_variables()

            train_params = []
            for param in params:
                if "resnet" in param.name:
                    train_params.append(param)

            gradients = tf.gradients(self.loss, train_params)
            for grad in gradients:
                tf.summary.histogram(grad.name, grad, collections=['train'])

            clipped_gradients, grad_norm = tf.clip_by_global_norm(gradients, self.train_config.max_gradient_norm)
            tf.summary.scalar("grad_norm", grad_norm, collections=['train'])

            # tf.summary.scalar("learning_rate", self.train_config.learning_rate, collections=['train'])
            # self.train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.99).apply_gradients(
            self.train_op = tf.train.AdamOptimizer(self.train_config.learning_rate).apply_gradients(
                zip(clipped_gradients, train_params), global_step=self.global_step)

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
