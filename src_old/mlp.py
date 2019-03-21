import numpy as np
import tensorflow as tf

from .base_cell import BaseCell


class MLP(BaseCell):
    ACT_FUNCS = {'relu': tf.nn.relu,
                 'relu6': tf.nn.relu6,
                 'elu': tf.nn.elu,
                 'leaky': tf.nn.leaky_relu,
                 'sigmoid': tf.sigmoid,
                 'tanh': tf.tanh,
                 'softplus': tf.nn.softplus,
                 'None': lambda x: x}

    def __init__(self, input_dim, hidden_dim, output_dim, hyper={}, act_func="relu"):
        super(MLP, self).__init__(input_dim, hidden_dim, output_dim, hyper)

        try:
            function = MLP.ACT_FUNCS[act_func]
        except KeyError:
            function = MLP.ACT_FUNCS["None"]
        self.act_func = function

        self.build(input_dim, hidden_dim, output_dim)

        self.loss = tf.losses.mean_squared_error(self.y_hat, self.y)
        self.optimise = self.optim.minimize(self.loss)

        self._Sess = tf.Session()
        init = tf.global_variables_initializer()
        self._Sess.run(init)

    def build(self, input_dim, hidden_dim, output_dim):
        initialiser = tf.truncated_normal_initializer()

        with tf.variable_scope("MLP_1"):
            W = tf.get_variable("W", [input_dim, hidden_dim], initializer=initialiser)
            b = tf.get_variable("b", initializer = np.zeros((hidden_dim,), dtype=np.float32))
            output = tf.add(tf.matmul(self.x, W), b, name="MLP_output")
            output = self.act_func(output)

        with tf.variable_scope("MLP_2"):
            W = tf.get_variable("W", [hidden_dim, output_dim], initializer=initialiser)
            b = tf.get_variable("b", initializer=np.zeros((output_dim,), dtype=np.float32))
            output = tf.add(tf.matmul(output, W), b, name="MLP_output")
            output = self.act_func(output)

        self.y_hat = output
