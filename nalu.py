import numpy as np
import tensorflow as tf

from base_cell import BaseCell
from utils import (operations, calcYFromOperation, gridSearch)

class NALU(BaseCell):
    def __init__(self, input_dim, hidden_dim, output_dim, hyper={}):
        super(NALU,self).__init__(input_dim, hidden_dim, output_dim, hyper)

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

        with tf.variable_scope("NALU_1"):
            W_hat = tf.get_variable("W_hat", [input_dim, hidden_dim], initializer=initialiser)
            M_hat = tf.get_variable("M_hat", [input_dim, hidden_dim], initializer=initialiser)
            W = tf.multiply(tf.nn.tanh(W_hat), tf.nn.sigmoid(M_hat), name='W')
            self.W1 = W

            G = tf.get_variable('G', [input_dim, hidden_dim], initializer=initialiser)
            g = tf.nn.sigmoid(tf.matmul(self.x, G), name='g')

            a = tf.matmul(self.x, W, name='a')

            log_x = tf.log(tf.abs(self.x) + 1e-7)
            m = tf.matmul(log_x, W, name='NAC_log_output')
            m = tf.exp(m, name='m')

            output = tf.add(tf.multiply(g,a), tf.multiply((1-g),m), name='NALU_output')

        with tf.variable_scope("NALU_2"):
            W_hat = tf.get_variable("W_hat", [hidden_dim, output_dim], initializer=initialiser)
            M_hat = tf.get_variable("M_hat", [hidden_dim, output_dim], initializer=initialiser)
            W = tf.multiply(tf.nn.tanh(W_hat), tf.nn.sigmoid(M_hat), name='W')
            self.W2 = W

            G = tf.get_variable('G', [hidden_dim, output_dim], initializer=initialiser)
            g = tf.nn.sigmoid(tf.matmul(output, G), name='g')

            a = tf.matmul(output, W, name='a')

            log_x = tf.log(tf.abs(output) + 1e-7)
            m = tf.matmul(log_x, W, name='NAC_log_output')
            m = tf.exp(m, name='m')

            output = tf.add(tf.multiply(g,a), tf.multiply((1-g),m), name='NALU_output')
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
            f.write("\nNALU Interpolation")

        best_results = {}
        tf.logging.set_verbosity(tf.logging.ERROR)
        for op, func in operations.items():
            y = calcYFromOperation(x, op)
            y_test = calcYFromOperation(x_test, op)

            best_lr = None
            best_err = 100000000

            best_err, best_lr = gridSearch(x, y, x_test, y_test, NALU,
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

    def trainAndGetTensors(x, x_test, op):
        y = calcYFromOperation(x, op)
        y_test = calcYFromOperation(x_test, op)

        model = NALU(100, 2, 1)
        model.train(x, y, x_test, y_test)

        w1, w2 = model.evalTensors()
        print(w1)
        print(w2)


    print("Testing NALU cell...")

    np.random.seed(42)
    x = np.random.uniform(10, 15, size=(10000,100))
    x_test = np.random.uniform(10, 15, size=(1000, 100))

    #calcOperations(x, x_test)
    trainAndGetTensors(x, x_test, "add")


    # Best addition: 0.01712 (0.0005)
    # Sub: -1.1595598 (0.0005)
    # Mult: 0.03124 (0.0005)
    # Divide: 0.03106 (0.005)
    # Square: 0.04506 (0.0005)
    # Root: 0.01235 (0.001)
