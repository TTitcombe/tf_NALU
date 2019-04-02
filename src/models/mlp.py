import tensorflow as tf


class MLP(tf.keras.Model):
    ACT_FUNCS = {'relu': tf.nn.relu,
                 'relu6': tf.nn.relu6,
                 'elu': tf.nn.elu,
                 'leaky': tf.nn.leaky_relu,
                 'sigmoid': tf.sigmoid,
                 'tanh': tf.tanh,
                 'softplus': tf.nn.softplus,
                 'None': lambda x: x}

    def __init__(self, input_dim, output_dim, hidden_dim=[], act_func=None):
        super(MLP, self).__init__()
        self._layers = []
        hidden_dim.append(output_dim)
        for dim in hidden_dim:
            self._layers.append(tf.keras.layers.Dense(dim, input_shape=(input_dim,)))
            input_dim = dim
        self.act_func = self._retrieve_act_func(act_func)

    def call(self, data):
        for layer in self._layers:
            data = layer(data)
            data = self.act_func(data)
        return data

    @staticmethod
    def _retrieve_act_func(act_func):
        try:
            return MLP.ACT_FUNCS[act_func]
        except KeyError:
            return MLP.ACT_FUNCS["None"]