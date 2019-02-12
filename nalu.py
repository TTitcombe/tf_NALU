import numpy as np
import tensorflow as tf

from base_cell import BaseCell

class NALU(BaseCell):
    def __init__(self, input_dim, hidden_dim, output_dim, hyper={}):
        super(NALU,self).__init__(input_dim, hidden_dim, output_dim, hyper)

        self.build(input_dim, hidden_dim, output_dim)

        self.loss = tf.losses.mean_squared_error(self.y_hat, self.y)
        self.optimise = self.optim.minimize(self.loss)

        self._Sess = tf.Session()
        init = tf.global_variables_initializer()
        self._Sess.run(init)

    def build(self, input_dim, hidden_dim, output_dim):
        initialiser = tf.truncated_normal_initializer()

        with tf.variable_scope("NALU_1"):
            W_hat = tf.get_variable("W_hat", [input_dim, hidden_dim], initializer=initialiser)
            M_hat = tf.get_variable("M_hat", [input_dim, hidden_dim], initializer=initialiser)
            W = tf.multiply(tf.nn.tanh(W_hat), tf.nn.sigmoid(M_hat), name='W')

            G = tf.get_variable('G', [input_dim, hidden_dim], initializer=initialiser)
            g = tf.nn.sigmoid(tf.matmul(self.x, G), name='g_weighting')

            a = tf.matmul(self.x, W, name='a')

            log_x = tf.log(self.x + 1e-7)
            m = tf.matmul(log_x, W, name='NAC_log_output')
            m = tf.exp(m, name='m')

            output = tf.add(tf.multiply(g,a), tf.multiply((1-g),m), name='NALU_output')
        with tf.variable_scope("NALU_2"):
            W_hat = tf.get_variable("W_hat", [hidden_dim, output_dim], initializer=initialiser)
            M_hat = tf.get_variable("M_hat", [hidden_dim, output_dim], initializer=initialiser)
            W = tf.multiply(tf.nn.tanh(W_hat), tf.nn.sigmoid(M_hat), name='W')

            G = tf.get_variable('G', [hidden_dim, output_dim], initializer=initialiser)
            g = tf.nn.sigmoid(tf.matmul(output, G), name='g_weighting')

            a = tf.matmul(output, W, name='a')

            log_x = tf.log(tf.abs(output) + 1e-7)
            m = tf.matmul(log_x, W, name='NAC_log_output')
            m = tf.exp(m, name='m')

            output = tf.add(tf.multiply(g,a), tf.multiply((1-g),m), name='NALU_output')
        self.y_hat = output
