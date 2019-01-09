import numpy as np
import tensorflow as tf

from base_cell import BaseCell
from utils import (operations, calcYFromOperation, gridSearch)

class NAC(BaseCell):
    def __init__(self, input_dim, hidden_dim, output_dim, hyper={}):
        super(NAC, self).__init__(input_dim, hidden_dim, output_dim, hyper)

        self.build(input_dim, hidden_dim, output_dim)

        self.error = tf.reduce_mean(tf.abs((self.y_hat - self.y)/self.y), name='mean_abs_error')
        self.square = tf.square(self.y_hat - self.y, name='square_diffs')
        self.loss = tf.reduce_mean(self.square, name='loss')
        self.optimise = self.optim.minimize(self.loss)

        self._Sess = tf.Session()
        init = tf.global_variables_initializer()
        self._Sess.run(init)

    def build(self, input_dim, hidden_dim, output_dim):
        initialiser = tf.truncated_normal_initializer()

        with tf.variable_scope("NAC_1"):
            W_hat = tf.get_variable("W_hat", [input_dim, hidden_dim], initializer=initialiser)
            M_hat = tf.get_variable("M_hat", [input_dim, hidden_dim], initializer=initialiser)

            W = tf.multiply(tf.nn.tanh(W_hat), tf.nn.sigmoid(M_hat), name='W')
            output = tf.matmul(self.x, W, name='NAC_output')

        with tf.variable_scope("NAC_2"):
            W_hat = tf.get_variable("W_hat", [hidden_dim, output_dim], initializer=initialiser)
            M_hat = tf.get_variable("M_hat", [hidden_dim, output_dim], initializer=initialiser)

            W = tf.multiply(tf.nn.tanh(W_hat), tf.nn.sigmoid(M_hat), name='W')
            output = tf.matmul(output, W, name='NAC_output')

        self.y_hat = output

if __name__ == "__main__":
    print("Testing NAC cell...")
    results_dir = 'results/'
    filename = "Interpolation.txt"

    with open(results_dir + filename, "a") as f:
        f.write("\nNAC Interpolation")

    np.random.seed(42)
    x = np.random.uniform(10, 15, size=(10000,100))
    x_test = np.random.uniform(10, 15, size=(1000, 100))

    best_results = {}
    tf.logging.set_verbosity(tf.logging.ERROR)
    for op, func in operations.items():
        y = calcYFromOperation(x, op)
        y_test = calcYFromOperation(x_test, op)

        best_lr = None
        best_err = 100000000

        best_err, best_lr = gridSearch(x, y, x_test, y_test, NAC,
                                       100, 2, 1, 10000)
        best_results[op] = (best_lr, best_err)

    with open(results_dir + filename, "a") as f:
        for _op, results in best_results.items():
            print("Operation {}".format(_op))
            print("Best lr: {}".format(results[0]))
            print("Best err: {}".format(results[1]))
            print("\n")
            f.write("\nOperation: {}\n".format(_op))
            f.write("Best lr: {}\n".format(results[0]))
            f.write("Best error: {}\n".format(results[1]))

    # Add: 0.01764 (0.005)
    # Subtract: 1.2488 (0.01)
    # Mult (0.027777 0.1)
    # Div: 0.03254 (0.0005)
    # Sq: 0.045984 (0.0005)
    # Root: 0.014275 (0.0001)
