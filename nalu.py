import numpy as np
import tensorflow as tf

from base_cell import BaseCell
from utils import (operations, calcYFromOperation, gridSearch)

class NALU(BaseCell):
    def __init__(self, input_dim, hidden_dim, output_dim, hyper={}, layers=2):
        super(NALU,self).__init__(input_dim, hidden_dim, output_dim, hyper)

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

        with tf.variable_scope("NALU_1"):
            W_hat = tf.get_variable("W_hat", [input_dim, hidden_dim], initializer=initialiser)
            M_hat = tf.get_variable("M_hat", [input_dim, hidden_dim], initializer=initialiser)
            W = tf.multiply(tf.nn.tanh(W_hat), tf.nn.sigmoid(M_hat), name='W')
            self.W1 = W

            G = tf.get_variable('G', [input_dim, hidden_dim], initializer=initialiser)
            g = tf.nn.sigmoid(tf.matmul(self.x, G), name='g_weighting')

            a = tf.matmul(self.x, W, name='a')

            log_x = tf.log(tf.abs(self.x) + 1e-7)
            m = tf.matmul(log_x, W, name='NAC_log_output')
            m = tf.exp(m, name='m')

            output = tf.add(tf.multiply(g,a), tf.multiply((1-g),m), name='NALU_output')
        self.y_layer1 = output

        with tf.variable_scope("NALU_2"):
            W_hat = tf.get_variable("W_hat", [hidden_dim, output_dim], initializer=initialiser)
            M_hat = tf.get_variable("M_hat", [hidden_dim, output_dim], initializer=initialiser)
            W = tf.multiply(tf.nn.tanh(W_hat), tf.nn.sigmoid(M_hat), name='W')
            self.W2 = W

            G = tf.get_variable('G', [hidden_dim, output_dim], initializer=initialiser)
            g = tf.nn.sigmoid(tf.matmul(output, G), name='g_weighting')

            a = tf.matmul(output, W, name='a')

            log_x = tf.log(tf.abs(output) + 1e-7)
            m = tf.matmul(log_x, W, name='NAC_log_output')
            m = tf.exp(m, name='m')

            output = tf.add(tf.multiply(g,a), tf.multiply((1-g),m), name='NALU_output')
        self.y_hat = output

    def evalTensors(self, x):
        w1 = self._Sess.run(self.W1)
        w2 = self._Sess.run(self.W2)

        with tf.variable_scope("NALU_1", reuse=True):
            G1 = tf.get_variable("G")
        g1 = tf.get_default_graph().get_operation_by_name("NALU_1/g_weighting")
        with tf.variable_scope("NALU_2", reuse=True):
            G2 = tf.get_variable("G")
        g2 = tf.get_default_graph().get_operation_by_name("NALU_2/g_weighting")

        G1_val = self._Sess.run(G1)
        G2_val = self._Sess.run(G2)
        g1_val = self._Sess.run(g1, {self.x:x})
        g2_val = self._Sess.run(g2, {self.x:x})
        return w1, w2, G1_val, G2_val, g1_val, g2_val

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

    def trainAndGetTensors(x, x_test, op, model):
        y = calcYFromOperation(x, op, 2, 1)
        y_test = calcYFromOperation(x_test, op, 2, 1)

        model.train(x, y, x_test, y_test, N_epochs=50000)

        w1, w2, G1, G2, g1, g2 = model.evalTensors(x_test)

        return [w1, G1, g1]


    print("Testing NALU cell...")

    np.random.seed(42)
    x = np.random.uniform(10, 100, size=(10,2))
    x_test = np.random.uniform(10, 100, size=(1, 2))

    results_dir = 'results/'
    filename = "Weights_Sanity_Test.txt"

    #calcOperations(x, x_test)
    weights = {}
    with open(results_dir + filename, "a") as f:
        f.write("NALU Sanity Check\n")
    for k, _ in operations.items():
        tf.reset_default_graph()
        model = NALU(2, 1, 1, layers=1)
        weights[k] = trainAndGetTensors(x, x_test, k, model)

    with open(results_dir + filename, "a") as f:
        for k, v in weights.items():
            print("{}: {} \n {} \n {}".format(k, v[0], v[1], v[2]))
            f.write("{}\n".format(k))
            f.write("W: {}\n".format(v[0]))
            f.write("G: {}\n".format(v[1]))
            f.write("g: {}\n".format(v[2]))


    # Best addition: 0.01712 (0.0005)
    # Sub: -1.1595598 (0.0005)
    # Mult: 0.03124 (0.0005)
    # Divide: 0.03106 (0.005)
    # Square: 0.04506 (0.0005)
    # Root: 0.01235 (0.001)
