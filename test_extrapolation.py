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
    for act_func in ['relu', 'leaky', 'sigmoid', 'tanh']:
        results[act_func] = np.zeros((1000,))
        print('Testing {}...'.format(act_func))
        for _ in range(100):
            g = tf.Graph()
            with g.as_default():
                model = Model(1, 1, hidden, model_type=act_func)
                trainer = Trainer(model)
                trainer.train(x, x, x_E, x_E, 100, 1000)

            final_output = trainer._get_output(x_E)[:,0]
            final_output = abs(x_E[:,0] - final_output)
            results[act_func] += final_output

    for k, v in results.items():
        plt.scatter(x_E, v/100, label=k)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    x = generate_data(1000)
    x_extrapolate = generate_data(1000, min=-20., max=20.)

    INPUT = 1
    OUTPUT = 1
    HIDDEN = [8, 8, 8]

    testing(x, x_extrapolate, HIDDEN)
