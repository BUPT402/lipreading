from tensorflow.python.layers import core as core_layers
import tensorflow as tf


class Lipreading(object):

    def __init__(self, model_config, iterator, train_config, word2idx):

        self.config = model_config
        self.train_config = train_config
        self.is_training = tf.placeholder_with_default(True, [])
        self.dropout_prob = tf.placeholder_with_default(1.0, [])
        self.data_format = "channels_last"

        self.iterator = iterator

        self.word2idx = word2idx

        self.initializer = tf.glorot_uniform_initializer()

        with tf.variable_scope("build_network"):
            with tf.variable_scope("decoder/output_projection"):
                self.output_layer = core_layers.Dense(len(self.word2idx), use_bias=False, name="output_projection")

        with tf.variable_scope("embeddings", dtype=tf.float32) as scope:
            with tf.variable_scope("decoder"):
                self.embedding_decoder = tf.get_variable("embedding_decoder", [len(self.word2idx), self.config.embedding_size])

        self.build_graph()


    def build_graph(self):
            self.build_inputs()
            self.build_conv3d()
            self.build_resnet()
            self.build_encoder()
            self.build_decoder()
            self.compute_loss()
            self.backprob()

    def build_inputs(self):
        with tf.name_scope('input'), tf.device('/cpu: 0'):
            self.image_seqs, self.tgt_in, self.tgt_out, self.label_length = self.iterator.get_next()


    def build_conv3d(self):
        with tf.variable_scope('conv3d') as scope:
            with tf.variable_scope('conv1'):
                conv1 = tf.layers.conv3d(self.image_seqs, 64, [5, 3, 3], [1, 2, 2], padding='same',
                                         use_bias=True, kernel_initializer=self.initializer, name='conv1')
                batch1 = tf.layers.batch_normalization(conv1, axis=-1, training=self.is_training, name='bn1')
                relu1 = tf.nn.relu(batch1, name='relu1')
                drop1 = tf.nn.dropout(relu1, self.dropout_prob, name='drop1')
                maxp1 = tf.layers.max_pooling3d(drop1, [1, 2, 2], [1, 1, 1], padding='same', name='maxpooling1')
                self.conv3d_out = maxp1

    def build_resnet(self):
        def batch_norm_relu(inputs, is_training, data_format):
            """Performs a batch normalization followed by a ReLU."""
            # We set fused=True for a significant performance boost. See
            # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
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
            """Creates one layer of blocks for the ResNet model.
            Args:
              inputs: A tensor of size [batch, channels, height_in, width_in] or
                [batch, height_in, width_in, channels] depending on data_format.
              filters: The number of filters for the first convolution of the layer.
              block_fn: The block to use within the model, either `building_block` or
                `bottleneck_block`.
              blocks: The number of blocks contained in the layer.
              strides: The stride to use for the first convolution of the layer. If
                greater than 1, this layer will ultimately downsample the input.
              is_training: Either True or False, whether we are currently training the
                model. Needed for batch norm.
              name: A string name for the tensor output of the block layer.
              data_format: The input format ('channels_last' or 'channels_first').
            Returns:
              The output tensor of the block layer.
            """
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

        block_fn = building_block
        res_input = tf.reshape(self.conv3d_out, [-1, 45, 70, 64])
        inputs = conv2d_fixed_padding(
            inputs=res_input, filters=64, kernel_size=3, strides=2,
            data_format=self.data_format)
        inputs = tf.identity(inputs, 'initial_conv')
        inputs = tf.layers.max_pooling2d(
            inputs=inputs, pool_size=3, strides=2, padding='SAME',
            data_format=self.data_format)
        inputs = tf.identity(inputs, 'initial_max_pool')
        inputs = block_layer(
            inputs=inputs, filters=64, block_fn=block_fn, blocks=3,
            strides=1, is_training=self.is_training, name='block_layer1',
            data_format=self.data_format)
        inputs = block_layer(
            inputs=inputs, filters=128, block_fn=block_fn, blocks=4,
            strides=2, is_training=self.is_training, name='block_layer2',
            data_format=self.data_format)
        inputs = block_layer(
            inputs=inputs, filters=256, block_fn=block_fn, blocks=6,
            strides=2, is_training=self.is_training, name='block_layer3',
            data_format=self.data_format)
        inputs = block_layer(
            inputs=inputs, filters=512, block_fn=block_fn, blocks=3,
            strides=2, is_training=self.is_training, name='block_layer4',
            data_format=self.data_format)

        inputs = batch_norm_relu(inputs, self.is_training, self.data_format)
        inputs = tf.layers.average_pooling2d(
            inputs=inputs, pool_size=2, strides=1, padding='VALID',
            data_format=self.data_format)
        inputs = tf.identity(inputs, 'final_avg_pool')
        inputs = tf.reshape(inputs,
                            [-1, 512 if block_fn is building_block else 2048])
        inputs = tf.layers.dense(inputs=inputs, units=4096)
        inputs = tf.identity(inputs, 'final_dense')
        self.res_out = tf.reshape(inputs, [-1, 77, 4096])

    def build_encoder(self):
        with tf.variable_scope("encoder") as scope:
            encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.MultiRNNCell([self.gru_cell() for _ in range(self.config.num_layers)]),
                cell_bw=tf.nn.rnn_cell.MultiRNNCell([self.gru_cell() for _ in range(self.config.num_layers)]),
                inputs=self.res_out, dtype=tf.float32, scope=scope)
            self.encoder_state = tuple([bi_encoder_state[0][1], bi_encoder_state[1][1]])
            self.encoder_outputs = tf.concat(encoder_outputs, -1)

    def build_decoder(self):
        with tf.variable_scope('decoder') as decoder_scope:
            memory = self.encoder_outputs
            batch_size = self.config.batch_size

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.config.num_units, memory)
            decode_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=tf.nn.rnn_cell.MultiRNNCell([self.gru_cell() for _ in range(self.config.num_layers)]),
                attention_mechanism=attention_mechanism,
                attention_layer_size=self.config.num_units,
                name="attention")
            decoder_initial_state = decode_cell.zero_state(batch_size, dtype=tf.float32).clone(
                cell_state=self.encoder_state)
            decoder_emb_inp = tf.nn.embedding_lookup(self.embedding_decoder, self.tgt_in)

            helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_emb_inp, self.label_length)

            my_decoder = tf.contrib.seq2seq.BasicDecoder(
                decode_cell,
                helper,
                decoder_initial_state)


            outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                my_decoder,
                maximum_iterations=tf.reduce_max(self.label_length),
                swap_memory=True,
                scope=decoder_scope)

            self.train_sample_id = outputs.sample_id
            self.train_logits = self.output_layer(outputs.rnn_output)


        with tf.variable_scope('decoder', reuse=True) as decoder_scope:
            memory_tiled = tf.contrib.seq2seq.tile_batch(memory, self.config.beam_width)
            encoder_state_tiled = tf.contrib.seq2seq.tile_batch(self.encoder_state, self.config.beam_width)
            batch_size_tiled = self.config.batch_size * self.config.beam_width
            infer_attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.config.num_units, memory_tiled)
            infer_decode_cell = tf.contrib.seq2seq.AttentionWrapper(
                tf.nn.rnn_cell.MultiRNNCell([self.gru_cell(reuse=True) for _ in range(self.config.num_layers)]),
                infer_attention_mechanism,
                attention_layer_size=self.config.num_units,
                name="attention")
            infer_decoder_initial_state = decode_cell.zero_state(batch_size_tiled, dtype=tf.float32).clone(
                cell_state=encoder_state_tiled)

            tgt_bos_id = self.word2idx['<BOS>']
            tgt_eos_id = self.word2idx['<EOS>']

            beam_width = self.config.beam_width
            start_tokens = tf.fill([self.config.batch_size], tgt_bos_id)
            end_token = tgt_eos_id

            infer_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=infer_decode_cell,
                embedding=self.embedding_decoder,
                start_tokens=start_tokens,
                end_token=end_token,
                initial_state=infer_decoder_initial_state,
                beam_width=beam_width,
                output_layer=self.output_layer)

            infer_outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                infer_decoder,
                maximum_iterations=tf.reduce_max(self.label_length) * 2,
                swap_memory=True,
                scope=decoder_scope)

            self.infer_pred_id = tf.transpose(infer_outputs.predicted_ids, [0, 2, 1])

    def gru_cell(self, reuse=False):
        cell = tf.nn.rnn_cell.GRUCell(self.config.num_units, reuse=reuse)
        dropout_cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.dropout_prob)
        return dropout_cell

    def get_max_time(self, tensor):
        time_axis = 1
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

    def compute_loss(self):
        target_output = self.tgt_out
        max_time = self.get_max_time(target_output)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=self.train_logits)
        target_weights = tf.sequence_mask(self.label_length, max_time, dtype=self.train_logits.dtype)

        self.eval_loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(self.config.batch_size)
        # self.eval_loss = tf.contrib.seq2seq.sequence_loss(
        #     logits=self.train_logits, targets=self.tgt_out, weights=target_weights)
        self.train_loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(self.config.batch_size)

    def backprob(self):
        params = tf.trainable_variables()
        params_summary = []
        for param in params:
            if 'kernel' in param.name:
                params_summary.append(tf.summary.histogram(param.name, param))
        opt = tf.train.AdamOptimizer(self.train_config.learning_rate)

        gradients = tf.gradients(self.train_loss, params, colocate_gradients_with_ops=True)

        clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, self.train_config.max_gradient_norm)
        gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
        gradient_norm_summary.append(tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

        self.update = opt.apply_gradients(zip(clipped_gradients, params))
        self.train_summary = tf.summary.merge([tf.summary.scalar("loss", self.train_loss)] + gradient_norm_summary
                                              + params_summary
                                              + [tf.summary.histogram("encoder_outputs", self.encoder_outputs)])

        self.infer_summarry = tf.no_op()

    def train(self, sess):

        _, loss, train_summary = sess.run([self.update, self.train_loss, self.train_summary], feed_dict={self.is_training: True, self.dropout_prob: 0.5})
        return loss, train_summary

    def eval(self, sess):
        loss, pred, label = sess.run([self.eval_loss, self.infer_pred_id, self.tgt_out],
                                          feed_dict={self.is_training: False, self.dropout_prob: 1.0})
        return loss, pred, label


if __name__ == '__main__':
    pass
