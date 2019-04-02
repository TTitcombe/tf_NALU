from __future__ import absolute_import

import tensorflow as tf

from .nac import NACLayer


class NALULayer(NACLayer):
    """
    A single NAC cell
    """
    def __init__(self, output_units):
        super(NALULayer, self).__init__(output_units)
        self.output_units = output_units

    def build(self, input_shape):
        super(NALULayer, self).build(input_shape)
        self.G = self.add_variable("G", [int(input_shape[-1]), int(self.output_units)])

    def call(self, data):
        W = tf.multiply(tf.tanh(self.W_hat), tf.sigmoid(self.M_hat))

        g = tf.sigmoid(tf.matmul(data, self.G))
        a = tf.matmul(data, W)

        log_x = tf.log(tf.abs(data) + 1e-7)
        m = tf.exp(tf.matmul(log_x, W))

        return tf.add(tf.multiply(g, a), tf.multiply((1-g), m))


class NALU(tf.keras.Model):
    def __init__(self, output_dim, hidden_dim=[]):
        super(NALU, self).__init__()
        hidden_dim.append(output_dim)
        self._layers = []
        for dim in hidden_dim:
            self._layers.append(NALULayer(dim))

    def call(self, data):
        for layer in self._layers:
            data = layer(data)
        return data
