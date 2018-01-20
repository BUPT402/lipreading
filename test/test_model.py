from tensorflow.python.layers import core as core_layers
import tensorflow as tf


class Lipreading(object):

    def __init__(self, model_config, iterator, train_config, word2idx):

        self.config = model_config
        self.train_config = train_config
        self.is_training = tf.placeholder_with_default(True, [])
        self.dropout_prob = tf.placeholder_with_default(1.0, [])

        self.iterator = iterator

        self.word2idx = word2idx

        self.initializer = tf.contrib.layers.xavier_initializer()

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
                conv1 = tf.layers.conv3d(self.image_seqs, 32, [5, 5, 5], [1, 2, 2], padding='same',
                                         use_bias=True, kernel_initializer=self.initializer, name='conv1')
                batch1 = tf.layers.batch_normalization(conv1, axis=-1, training=self.is_training, name='bn1')
                relu1 = tf.nn.relu(batch1, name='relu1')
                drop1 = tf.nn.dropout(relu1, self.dropout_prob, name='drop1')
                maxp1 = tf.layers.max_pooling3d(drop1, [1, 2, 2], [1, 2, 2], padding='valid', name='maxpooling1')
                self.video_feature = tf.reshape(maxp1, [-1, 77, 22 * 35 * 32])

    def build_encoder(self):
        with tf.variable_scope("encoder") as scope:
            encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.MultiRNNCell([self.gru_cell() for _ in range(self.config.num_layers)]),
                cell_bw=tf.nn.rnn_cell.MultiRNNCell([self.gru_cell() for _ in range(self.config.num_layers)]),
                inputs=self.video_feature, dtype=tf.float32, scope=scope)
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
        cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size, kernel_initializer=tf.orthogonal_initializer(), reuse=reuse)
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
        opt = tf.train.AdamOptimizer(self.train_config.learning_rate)

        gradients = tf.gradients(self.train_loss, params, colocate_gradients_with_ops=True)

        clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, self.train_config.max_gradient_norm)
        gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
        gradient_norm_summary.append(tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

        self.update = opt.apply_gradients(zip(clipped_gradients, params))
        self.train_summary = tf.summary.merge([tf.summary.scalar("train_loss", self.train_loss)] + gradient_norm_summary)
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
