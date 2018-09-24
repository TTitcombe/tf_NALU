'''Test extrapolation failures of standard neural networks.
Replication of the experiment in section 1.1 of the paper
Neural Arithmetic Logic Unit'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from trainer import Trainer
from models import Model

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
        N_repeats = 10
        for _ in range(N_repeats):
            g = tf.Graph()
            with g.as_default():
                model = Model(1, 1, hidden, model_type=act_func)
                trainer = Trainer(model)
                trainer.train(x, x, x_E, x_E, 100, 10000)

            final_output = trainer._get_output(x_E)[:,0]
            final_output = abs(x_E[:,0] - final_output)
            results[act_func] += final_output

    for k, v in results.items():
        plt.scatter(x_E, v/N_repeats, label=k)
    plt.legend()
    plt.savefig('tmp.png')
    plt.show()

if __name__ == '__main__':
    x = generate_data(1000)
    x_extrapolate = generate_data(1000, min=-20., max=20.)

    INPUT = 1
    OUTPUT = 1
    HIDDEN = [8, 8, 8]

    testing(x, x_extrapolate, HIDDEN)
