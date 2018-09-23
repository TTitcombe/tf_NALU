import numpy as np
import tensorflow as tf

class NAC_cell():
    '''
    A Basic Neural Accumulator Cell
    '''
    def __init__(self, input_dim, output_dim, scope_name='NAC_1'):
        '''
        scope_name is to avoid naming errors if multiple NACs are defined in the
        same model
        '''
        self.scope_name = scope_name

        initialiser = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(scope_name):
            W_hat = tf.get_variable("W_hat", [input_dim, output_dim], initializer=initialiser)
            M_hat = tf.get_variable("M_hat", [input_dim, output_dim], initializer=initialiser)

    def __call__(self, x_input):
        with tf.variable_scope(self.scope_name, reuse=True):
            W_hat = tf.get_variable("W_hat")
            M_hat = tf.get_variable("M_hat")

        W = tf.multiply(tf.nn.tanh(W_hat), tf.nn.sigmoid(M_hat), name='W')
        output = tf.matmul(x_input, W, name='NAC_output')

        return output


class NALU_cell():
    '''
    A single Neural Arithmetic Logic Unit
    '''

    def __init__(self, input_dim, output_dim, scope_n = '0'):
        '''
        scope_n defines NALU number within the model, to avoid naming errors
        '''
        self.scope_n = scope_n
        self.eps = 10e-6

        initialiser = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('NALU_{}'.format(scope_n)):
            self.nac = NAC_cell(input_dim, output_dim)

            G = tf.get_variable('G', [input_dim, output_dim], initializer=initialiser)
            b_g = tf.get_variable('b_g', [output_dim,], initializer=tf.zeros_initializer())

    def __call__(self, x_input):

        #normal space
        with tf.variable_scope('NALU_{}'.format(self.scope_n), reuse=tf.AUTO_REUSE):
            a = self.nac(x_input)
            a = tf.identity(a, name='a')

            #log space
            log_x = tf.log(tf.abs(x_input) + self.eps)
            m = self.nac(log_x)
            m = tf.exp(m, name='m')

            G = tf.get_variable('G')
            g = tf.nn.sigmoid(tf.matmul(x_input, G), name='g')

            output = tf.add(tf.multiply(g,a),tf.multiply((1-g), m), name='nalu_output')

        return output

class NALU():
    '''
    A model comprised of multiple (consecutive) NALU modules.
    '''
    def __init__(self, input_dim, output_dim, hidden_dim=[]):
        '''
        inputs:
            input_dim | int, number of dimensions on the x data
            output_dim | int, number of dimensions on the y data
            hidden_dim | list of ints (or empty list), hidden dimensions of
                        NALU modules in this model
                        Number of NALU modules = length of hidden_dim + 1
        '''
        hid_dim_type = type(hidden_dim) is list or hidden_dim is None
        assert hid_dim_type, "hidden_dim must be None or a list"

        self._layers = []
        if len(hidden_dim) > 0:
            for i, dim in enumerate(hidden_dim):
                if i == 0:
                    old_dim = input_dim
                else:
                    old_dim = hidden_dim[i-1]
                self._layers.append(NALU_cell(old_dim, dim, scope_n=str(i)))
            input_dim = hidden_dim[-1]

        self._layers.append(
        NALU_cell(input_dim, output_dim, scope_n=str(len(hidden_dim)))
        )

    def __call__(self, x_input):
        for k, cell in enumerate(self._layers):
            x_input = cell(x_input)

        return x_input
