import tensorflow as tf


class NACLayer(tf.keras.layers.Layer):
    """
    A single NAC cell
    """
    def __init__(self, output_units):
        super(NACLayer, self).__init__()
        self.output_units = output_units

    def build(self, input_shape):
        self.W_hat = self.add_variable("W_hat", [input_shape[-1], self.output_units])
        self.M_hat = self.add_variable("M_hat", [input_shape[-1], self.output_units])

    def call(self, data):
        W = tf.multiply(tf.tanh(self.W_hat), tf.sigmoid(self.M_hat))
        return tf.matmul(data, W)


class NAC(tf.keras.Model):
    def __init__(self, output_dim, hidden_dim=[]):
        super(NAC, self).__init__()
        hidden_dim.append(output_dim)
        self._layers = []
        for dim in hidden_dim:
            self._layers.append(NACLayer(dim))

    def call(self, data):
        for layer in self._layers:
            data = layer(data)
        return data
