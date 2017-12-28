from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class ModelConfig(object):

    def __init__(self):
        #tfrecords的目录，在train和evaluation中必须有
        self.input_file = None

        self.image_format = 'png'

        #队列中的容量
        self.queue_capcity = 200

        #队列最短长度
        self.shuffle_min_after_dequeue = 100

        self.num_threads = 4

        self.label_length_name = 'label_length'
        self.frames_name = 'frames'
        self.label_name = 'labels'

        self.batch_size = 10

        self.image_wieght = 90
        self.image_height = 140
        self.image_depth = 250
        self.image_channel = 3

        self.initializer_scale = 0.08

        self.beam_width = 5

        self.embedding_size = 512
        self.hidden_size = 512
        self.num_layers = 2

        self.force_teaching_ratio = 0.8

        self.conv_dropout_keep_prob = 0.5


class TrainingConfig(object):

    def __init__(self):
        self.num_examples_per_epoch = 28363

        self.initial_learning_rate = 0.001
        self.learning_rate_decay = 0.9
        self.num_epoch_per_decay = 8.0

        self.clip_gradients = 5.0
