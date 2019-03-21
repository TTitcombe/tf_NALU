'''Test extrapolation failures of standard neural networks.
Replication of the experiment in section 1.1 of the paper
Neural Arithmetic Logic Unit'''
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

from .mlp import MLP


def generate_data(n, min=-5., max=5., seed=42):
    if seed is not None:
        np.random.seed(seed)
    x = np.random.uniform(min, max, size=(n,1))
    return x


def testing(x, x_E, hidden):
    results = {}
    for act_func in ['relu6','relu', 'leaky', 'elu', 'sigmoid', 'tanh']:
        results[act_func] = np.zeros((1000,))
        print('Testing {}...'.format(act_func))
        model = MLP(1, hidden, 1, act_func=act_func)
        model.train(x, x, x_E, x_E)

        _, final_output = model.validate(x_E,x_E)
        final_output = final_output[:,0]
        final_output = abs(x_E[:,0] - final_output)
        results[act_func] = final_output

        tf.reset_default_graph()

    return results, x_E[:,0]


def plot(results, x, save_name):
    for k, v in results.items():
        plt.scatter(x, v, label=k)
    plt.legend()
    plt.savefig(save_name)
    plt.show()


if __name__ == '__main__':
    x = generate_data(10000)
    x_extrapolate = generate_data(5000, min=-25., max=25.)

    save_name = os.path.join("figures", "extrapolation_testTMP.png")

    INPUT = 1
    OUTPUT = 1
    HIDDEN = 8

    results, x = testing(x, x_extrapolate, HIDDEN)
    plot(results, x, save_name)
