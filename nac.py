import numpy as np
import tensorflow as tf

from base_cell import BaseCell
from utils import (operations, calcYFromOperation, gridSearch)

class NAC(BaseCell):
    def __init__(self, input_dim, hidden_dim, output_dim, hyper={}, layers=2):
        super(NAC, self).__init__(input_dim, hidden_dim, output_dim, hyper)

        self.build(input_dim, hidden_dim, output_dim)

        if layers == 2:
            output = self.y_hat
        elif layers == 1:
            output = self.y_layer1
        self.error = tf.reduce_mean(tf.abs((output - self.y)/self.y), name='mean_abs_error')
        self.square = tf.square(output - self.y, name='square_diffs')
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
            self.W1 = W
            output = tf.matmul(self.x, W, name='NAC_output')
        self.y_layer1 = output

        with tf.variable_scope("NAC_2"):
            W_hat = tf.get_variable("W_hat", [hidden_dim, output_dim], initializer=initialiser)
            M_hat = tf.get_variable("M_hat", [hidden_dim, output_dim], initializer=initialiser)

            W = tf.multiply(tf.nn.tanh(W_hat), tf.nn.sigmoid(M_hat), name='W')
            self.W2 = W
            output = tf.matmul(output, W, name='NAC_output')

        self.y_hat = output

    def evalTensors(self):
        w1 = self._Sess.run(self.W1)
        w2 = self._Sess.run(self.W2)
        return w1, w2

if __name__ == "__main__":
    def calcOperations(x, x_test):
        results_dir = 'results/'
        filename = "Interpolation.txt"

        with open(results_dir + filename, "a") as f:
            f.write("\nNAC Interpolation")

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
                f.write("\nBest lr: {}\n".format(results[0]))
                f.write("\nBest error: {}\n".format(results[1]))

    def trainAndGetTensors(x, x_test, op, model):
        y = calcYFromOperation(x, op, 2, 1)
        y_test = calcYFromOperation(x_test, op, 2, 1)

        model.train(x, y, x_test, y_test, N_epochs=50000)

        w1, w2 = model.evalTensors()
        print(w1)
        print(w2)


    print("Testing NAC cell...")

    np.random.seed(42)
    x = np.random.uniform(10, 100, size=(10,2))
    x_test = np.random.uniform(10, 100, size=(1, 2))

    #calcOperations(x, x_test)

    model = NAC(2, 1, 1, layers=1)
    trainAndGetTensors(x, x_test, "add", model)

    # Add: 0.01764 (0.005)
    # Subtract: 1.2488 (0.01)
    # Mult (0.027777 0.1)
    # Div: 0.03254 (0.0005)
    # Sq: 0.045984 (0.0005)
    # Root: 0.014275 (0.0001)
