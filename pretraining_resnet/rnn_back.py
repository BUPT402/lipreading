from tensorflow.python.layers import core as core_layers
import tensorflow as tf


class Lipreading(object):

    def __init__(self, model_config, iterator, train_config, word2idx, dropout=0.5):

        self.config = model_config
        self.train_config = train_config
        self.is_training = tf.placeholder_with_default(True, [])
        self.data_format = "channels_last"
        self.num_classes = 500

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(train_config.learning_rate,
                                                        self.global_step, 15000, 0.5, staircase=True)

        self.dropout_prob = dropout

        self.iterator = iterator

        self.word2idx = word2idx

        self.build_graph()


    def build_graph(self):
            self.build_inputs()
            self.build_conv3d()
            self.build_resnet()
            self.build_classifier()
            self.build_train()
            self.build_prediction()
            self.merged_summary = tf.summary.merge_all('train')

    def build_inputs(self):
        with tf.name_scope('input'), tf.device('/cpu: 0'):
            self.image_seqs, self.labels = self.iterator.get_next()

    def build_conv3d(self):
        with tf.variable_scope('conv3d') as scope:
            with tf.variable_scope('conv1'):
                conv1 = tf.layers.conv3d(self.image_seqs, 32, [5, 5, 5], [1, 2, 2], padding='same',
                                         use_bias=True, name='conv1')
                batch1 = tf.layers.batch_normalization(conv1, axis=-1, training=self.is_training, name='bn1')
                relu1 = tf.nn.relu(batch1, name='relu1')
                # drop1 = tf.nn.dropout(relu1, self.dropout_prob, name='drop1')
                # maxp1 = tf.layers.max_pooling3d(drop1, [1, 2, 2], [1, 2, 2], padding='same', name='maxpooling1')
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
            self.res_out = tf.reshape(inputs, [-1, self.config.image_depth, 512])
            tf.summary.histogram("res_out", self.res_out, collections=['train'])

    def gru_cell(self, num_units, reuse=False):
        cell = tf.nn.rnn_cell.GRUCell(num_units, reuse=reuse)
        dropout_cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.dropout_prob, self.dropout_prob)
        return dropout_cell

    def build_classifier(self):
        with tf.variable_scope("bi_gru_backend") as scope, tf.name_scope("bi_gru_backend"):
            backend_outputs, bi_gru_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.MultiRNNCell(
                    [self.gru_cell(self.config.num_units) for _ in range(self.config.num_layers)]),
                cell_bw=tf.nn.rnn_cell.MultiRNNCell(
                    [self.gru_cell(self.config.num_units) for _ in range(self.config.num_layers)]),
                inputs=self.res_out, dtype=tf.float32, scope=scope)
            backend_state = tf.concat([bi_gru_state[0][0], bi_gru_state[1][0], bi_gru_state[0][1], bi_gru_state[1][1]], axis=-1)
            self.backend_state = backend_state
        with tf.variable_scope("fc") as scope:
            self.logits = tf.layers.dense(self.backend_state, self.num_classes)

    def build_train(self):
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.logits)
        tf.summary.scalar('loss', self.loss, collections=['train'])
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            params = tf.trainable_variables()
            train_params = []
            for param in params:
                if "bi_gru_backend" in param.name:
                    train_params.append(param)
            gradients = tf.gradients(self.loss, train_params)
            for grad in gradients:
                tf.summary.histogram(grad.name, grad, collections=['train'])
            clipped_gradients, grad_norm = tf.clip_by_global_norm(gradients, self.train_config.max_gradient_norm)
            print(clipped_gradients)
            tf.summary.scalar("grad_norm", grad_norm, collections=['train'])
            tf.summary.scalar("learning_rate", self.learning_rate, collections=['train'])
            self.train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.9).apply_gradients(
                zip(clipped_gradients, train_params), global_step=self.global_step)

    def build_prediction(self):
        correct = tf.nn.in_top_k(self.logits, self.labels, 1)
        correct = tf.cast(correct, tf.float16)
        self.accuracy = tf.reduce_mean(correct)

    def train(self, sess):
        _, loss = sess.run([self.train_op, self.loss])
        return loss

    def eval(self, sess):
        pred = sess.run([self.accuracy], feed_dict={self.is_training: False})
        return pred

    def merge(self, sess):
        summary = sess.run(self.merged_summary)
        return summary

if __name__ == '__main__':
    pass
