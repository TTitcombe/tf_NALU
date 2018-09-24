import numpy as np
import tensorflow as tf

class MLP():
    def __init__(self, input_dim, output_dim, hidden_dim = [], act_func='relu'):
        self.__n_layers = len(hidden_dim)

        if act_func == 'relu':
            self.act_func = tf.nn.relu
        elif act_func == 'leaky':
            self.act_func = tf.nn.leaky_relu
        elif act_func == 'sigmoid':
            self.act_func = tf.sigmoid
        elif act_func == 'tanh':
            self.act_func = tf.tanh

        initialiser = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('MLP'):
            for i, dim in enumerate(hidden_dim):
                if i == 0:
                    inp_dim = input_dim
                else:
                    inp_dim = hidden_dim[i-1]
                W = tf.get_variable("W{}".format(i), [inp_dim, dim], initializer=initialiser)
                b = tf.get_variable('b{}'.format(i), [dim,], initializer=tf.zeros_initializer())
            W = tf.get_variable("Wout", [hidden_dim[-1], output_dim], initializer=initialiser)
            b = tf.get_variable('bout', [output_dim,], initializer=tf.zeros_initializer())

    def __call__(self, x_input):
        with tf.variable_scope('MLP', reuse=True):
            for i in range(self.__n_layers):
                W = tf.get_variable("W{}".format(i))
                b = tf.get_variable("b{}".format(i))
                x_input = tf.add(tf.matmul(x_input, W), b)
                #x_input = self.act_func(x_input)

            W = tf.get_variable("Wout")
            b = tf.get_variable("bout")
            x_input = tf.add(tf.matmul(x_input, W), b)
            x_input = self.act_func(x_input)

        return x_input
