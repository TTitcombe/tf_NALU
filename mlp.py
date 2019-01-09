import numpy as np
import tensorflow as tf

from base_cell import BaseCell
from utils import (operations, calcYFromOperation, gridSearch)

class MLP(BaseCell):
    def __init__(self, input_dim, hidden_dim, output_dim, hyper={}):
        super(MLP, self).__init__(input_dim, hidden_dim, output_dim, hyper)

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

        with tf.variable_scope("MLP_1"):
            W = tf.get_variable("W", [input_dim, hidden_dim], initializer=initialiser)
            output = tf.matmul(self.x, W, name='MLP_output')

        with tf.variable_scope("MLP_2"):
            W = tf.get_variable("W", [hidden_dim, output_dim], initializer=initialiser)
            output = tf.matmul(output, W, name='MLP_output')

        self.y_hat = output

if __name__ == "__main__":
    print("Testing MLP...")
    results_dir = 'results/'
    filename = "Interpolation.txt"

    with open(results_dir + filename, "a") as f:
        f.write("\nMLP Interpolation")

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

        best_err, best_lr = gridSearch(x, y, x_test, y_test, MLP,
                                       100, 2, 1, 10000)
        best_results[op] = (best_lr, best_err)

    with open(results_dir + filename, "a") as f:
        for _op, results in best_results.items():
            print("Operation {}".format(_op))
            print("Best lr: {}".format(results[0]))
            print("Best err: {}".format(results[1]))
            print("\n")
            f.write("\nOperation: {}\n".format(_op))
            f.write("\nBest lr: {}\n".format(results[0]))
            f.write("\nBest error: {}\n".format(results[1]))

    # Add: 0.01857 (0.001)
    # Sub: 1.508765 (0.0005)
    # Mult: 0.02868 (0.05)
    # Divide: 0.05604 (0.005)
    # Square: 0.044806 (0.005)
    # Root: 0.021807 (0.0005)
