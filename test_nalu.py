import numpy as np

from trainer import Trainer
from models import Model

operations = {
'add': lambda x, y: x + y,
'subtract': lambda x,y: x - y,
'multiply': lambda x,y: x * y,
'divide': lambda x, y: x / y,
'square': lambda x, y: x**2,
'root': lambda x, y: np.sqrt(x)
}

def generate_data(n,z=100, min=-1., max=1., op = 'add', seed=42):
    global operations

    if seed is not None:
        np.random.seed(seed)

    _a = np.random.choice(z, 10)
    _b = np.random.choice(z, 10)


    x = np.random.uniform(min, max, size=(n,z))
    a = np.sum(x[:,_a], axis=1)
    b = np.sum(x[:,_b], axis=1)

    y = operations[op](a,b)
    y = np.reshape(y, (y.shape[0],1))

    return x, y


if __name__ == '__main__':
    _op = 'add'
    x, y = generate_data(1000, op=_op)
    x_extrapolate, y_extrapolate = generate_data(100, min=-10., max=10., op=_op)

    INPUT = 100
    OUTPUT = 1
    HIDDEN = [2]

    models = ['nac', 'nalu', 'relu']
    losses = {}

    for model_type in models:
        print("Testing {}...".format(model_type))
        model = Model(INPUT, OUTPUT, HIDDEN, model_type=model_type)
        trainer = Trainer(model)
        trainer.train(x, y, x_extrapolate, y_extrapolate, 128, 1000)
        err, loss = trainer._validate(x_extrapolate, y_extrapolate)
        losses[model_type] = loss

    for k, v in losses.items():
        print(k)
        print(v)
