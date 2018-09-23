import numpy as np

from trainer import Trainer

def generate_data(n,z=100, min=-1., max=1., op = 'add', seed=42):
    if seed is not None:
        np.random.seed(seed)

    _a = np.random.choice(z, 10)
    _b = np.random.choice(z, 10)


    x = np.random.uniform(min, max, size=(n,z))
    a = np.sum(x[:,_a], axis=1)
    b = np.sum(x[:,_b], axis=1)

    if op == 'add':
        y = a
    elif op == 'multiply':
        y = a * b
    else:
        raise NotImplementedError("Only addition and multiplication problems")
    y = np.reshape(y, (y.shape[0],1))

    return x, y

x, y = generate_data(1000)
x_extrapolate, y_extrapolate = generate_data(100, min=-10., max=10.)

INPUT = 100
OUTPUT = 1
HIDDEN = [50]

'''print('Testing NAC...')
train_model = Trainer(INPUT, OUTPUT, HIDDEN)
train_model.train(x, y, x_extrapolate, y_extrapolate, 128, 1000)'''

print('Testing single NALU...')
nalu_model = Trainer(INPUT, OUTPUT, [50], model_type='nalu')
nalu_model.train(x, y, x_extrapolate, y_extrapolate, 128, 1000)

'''print('Testing NALU....')
nalu_model = Trainer(INPUT, OUTPUT, HIDDEN, model_type='nalu')
nalu_model.train(x, y, x_extrapolate, y_extrapolate, 128, 1000)'''
