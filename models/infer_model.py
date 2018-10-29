from tensorflow.python.layers import core as core_layers
import tensorflow as tf


class Lipreading(object):
    def __init__(self, model_config, word2idx, batch_size=1):

        self.config = model_config
        self.data_format = "channels_last"
        self.batch_size = batch_size
        self.word2idx = word2idx
        # self.image_length = tf.placeholder_with_default(77, [], name='img_len')
        self.build_graph()

    def build_graph(self):
        self.build_inputs()
        self.build_conv3d()
        self.build_resnet()
        self.build_encoder()
        self.build_decode_for_infer()

    def build_inputs(self):
        self.image_seqs = tf.placeholder(shape=[None, None, 112, 112, 3], dtype=tf.float32, name='img')
        self.image_length = tf.placeholder(shape=[], dtype=tf.int32, name='img_len')
        self.label_length = [30]

    def build_conv3d(self):
        with tf.variable_scope('conv3d') as scope:
            with tf.variable_scope('conv1'):
                conv1 = tf.layers.conv3d(self.image_seqs, 32, [5, 5, 5], [1, 2, 2], padding='same',
                                         use_bias=True, name='conv1')
                batch1 = tf.layers.batch_normalization(conv1, axis=-1, training=False, name='bn1')
                relu1 = tf.nn.relu(batch1, name='relu1')
                self.video_feature = relu1
                video_vis = relu1[:, :, :, :, 1:2]
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
            inputs_vis = inputs[:, :, :, 2]
            inputs = tf.identity(inputs, 'initial_conv')
            inputs = tf.layers.max_pooling2d(
                inputs=inputs, pool_size=3, strides=2, padding='SAME',
                data_format=data_format)
            inputs = tf.identity(inputs, 'initial_max_pool')
            inputs = block_layer(
                inputs=inputs, filters=64, block_fn=block_fn, blocks=3,
                strides=1, is_training=False, name='block_layer1',
                data_format=data_format)
            inputs_vis = inputs[:, :, :, 2]
            inputs = block_layer(
                inputs=inputs, filters=128, block_fn=block_fn, blocks=4,
                strides=2, is_training=False, name='block_layer2',
                data_format=data_format)
            inputs_vis = inputs[:, :, :, 2]
            inputs = block_layer(
                inputs=inputs, filters=256, block_fn=block_fn, blocks=6,
                strides=2, is_training=False, name='block_layer3',
                data_format=data_format)
            inputs_vis = inputs[:, :, :, 2]
            inputs = block_layer(
                inputs=inputs, filters=512, block_fn=block_fn, blocks=3,
                strides=2, is_training=False, name='block_layer4',
                data_format=data_format)

            inputs = batch_norm_relu(inputs, False, data_format)
            inputs = tf.layers.average_pooling2d(
                inputs=inputs, pool_size=4, strides=1, padding='VALID',
                data_format=data_format)
            inputs = tf.identity(inputs, 'final_avg_pool')
            self.res_out = tf.reshape(inputs, [-1, self.image_length, 512])

            tf.summary.histogram("res_out", self.res_out, collections=['train'])

    def build_encoder(self):
        with tf.variable_scope("encoder") as scope:
            encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.MultiRNNCell(
                    [self.gru_cell(self.config.num_units) for _ in range(self.config.num_layers)]),
                cell_bw=tf.nn.rnn_cell.MultiRNNCell(
                    [self.gru_cell(self.config.num_units) for _ in range(self.config.num_layers)]),
                inputs=self.res_out, dtype=tf.float32, scope=scope)
            encoder_state = []
            for i in range(self.config.num_layers):
                encoder_state.append(tf.concat([bi_encoder_state[0][i], bi_encoder_state[1][i]], axis=-1))
            self.encoder_state = tuple(encoder_state)
            self.encoder_out = tf.concat(encoder_outputs, -1)
            encoder_vis = self.encoder_out[:, :, 100:150]
            tf.summary.histogram('encoder_out', self.encoder_out, collections=['train'])
            tf.summary.histogram('encoder_state', self.encoder_state, collections=['train'])

    def gru_cell(self, num_units, reuse=tf.AUTO_REUSE):
        cell = tf.nn.rnn_cell.GRUCell(num_units, reuse=reuse)
        return cell

    def build_decode_for_infer(self):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE) as scope:
            decode_embedding = tf.get_variable('decode_embedding',
                                               [len(self.word2idx), self.config.embedding_size],
                                               tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
            encoder_out_tiled = tf.contrib.seq2seq.tile_batch(self.encoder_out, self.config.beam_width)
            encoder_state_tiled = tf.contrib.seq2seq.tile_batch(self.encoder_state, self.config.beam_width)

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.config.num_units * 2,
                                                                    memory=encoder_out_tiled)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=tf.nn.rnn_cell.MultiRNNCell([self.gru_cell(self.config.num_units * 2, reuse=tf.AUTO_REUSE) for _ in
                                                  range(self.config.num_layers)]),
                attention_mechanism=attention_mechanism, attention_layer_size=self.config.num_units * 2)
            predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell, embedding=tf.get_variable('decode_embedding'),
                start_tokens=tf.tile(tf.constant([2], dtype=tf.int32), [1]),
                end_token=1,
                initial_state=decoder_cell.zero_state(self.batch_size * self.config.beam_width, tf.float32).clone(
                    cell_state=encoder_state_tiled),
                beam_width=self.config.beam_width,
                output_layer=core_layers.Dense(len(self.word2idx), use_bias=False, _reuse=tf.AUTO_REUSE),
                length_penalty_weight=0.0)
            self.predicting_decoder_output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=predicting_decoder,
                impute_finished=False,
                maximum_iterations=2 * tf.reduce_max(self.label_length),
                scope=scope)
            score_vis = self.predicting_decoder_output.beam_search_decoder_output.scores
            score_vis = score_vis[:, :, 0:3]
            word_idx = self.predicting_decoder_output.beam_search_decoder_output.predicted_ids
            word_idx = word_idx[:, :, 0:3]

            with tf.variable_scope('pre_result'):
                self.predicting_ids = self.predicting_decoder_output.predicted_ids[:, :, 0]


if __name__ == '__main__':
    pass
