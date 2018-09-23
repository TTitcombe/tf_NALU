import numpy as np
import tensorflow as tf

from nalu import NAC_cell, NALU_cell, NALU

class model():
    '''
    A class to store the learning hyperparameters of the NALU network
    e.g. learning rate, learning algorithm, decay rate etc.
    '''
    def __init__(self, input_dim, output_dim, hidden_dim, model_type, hyper = {}):
        self.x = tf.placeholder(tf.float32, [None, input_dim], name='input')
        self.y = tf.placeholder(tf.float32, [None, output_dim], name='ouput')

        if model_type == 'nalu':
            _model = NALU(input_dim, output_dim, hidden_dim)
        elif model_type == 'nalu_single':
            _model = NALU_cell(input_dim, output_dim)
        elif model_type == 'nac':
            _model = NAC_cell(input_dim, output_dim)
        else:
            raise NotImplementedError("Need to do MLP")

        self.model = _model

        self.global_step = tf.Variable(0, trainable=False)
        optim = hyper.get('optim', 'rms')
        decay = hyper.get('decay', None)

        if decay is not None:
            start_lr = hyper.get('lr', 0.001)
            self.lr = tf.train.exponential_decay(start_lr, self.global_step,
                                                       decay_steps=1000, decay_rate=decay, staircase=True)
        else:
            self.lr = hyper.get('lr', 0.001)

        if optim.lower() == 'adam':
            self.optim = tf.train.AdamOptimizer(self.lr)
        elif optim.lower() == 'gd':
            self.optim = tf.train.GradientDescentOptimizer(self.lr)
        elif optim.lower() == 'rms':
            self.optim = tf.train.RMSPropOptimizer(self.lr)
        else:
            raise NotImplementedError("Learning algorithm not recognised")

        self.y_hat = self.model(self.x)
        self.square = tf.square(tf.subtract(self.y_hat, self.y), name='square_diffs')
        self.loss = tf.reduce_mean(self.square, name='loss')
        self.optimise = self.optim.minimize(self.loss)
