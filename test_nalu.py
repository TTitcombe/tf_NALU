import numpy as np
import tensorflow as tf

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

def generate_data(n,z=100, min=-1., max=1., select = 10, op = 'add', seed=42):
    global operations

    if seed is not None:
        np.random.seed(seed)

    _a = np.random.choice(z, select*2)

    x = np.random.uniform(min, max, size=(n,z))
    a = np.sum(x[:,_a[:select]], axis=1)
    b = np.sum(x[:,_a[select:]], axis=1)

    y = operations[op](a,b)
    y = np.reshape(y, (y.shape[0],1))

    return x, y

def test(models, x_train, y_train, x_Is, y_Is,
            INPUT, OUTPUT, HIDDEN, batch_size,
            N_epochs, filename):
    global results_dir
    losses = {}
    N_test_data = len(x_Is)

    for model_type in models:
        tf.reset_default_graph()
        print("Testing {}...".format(model_type))
        losses[model_type] = 0.

        if model_type == 'random':
            model = Model(INPUT, OUTPUT, HIDDEN, model_type='relu6')
            trainer = Trainer(model)
        else:
            model = Model(INPUT, OUTPUT, HIDDEN, model_type=model_type)
            trainer = Trainer(model)
            trainer.train(x, y, x_Is[0], y_Is[0], batch_size, N_epochs)

        for x_I, y_I in zip(x_Is, y_Is):
            err, loss = trainer._validate(x_I, y_I)
            losses[model_type] += loss
        losses[model_type] /= N_test_data

    with open(results_dir + filename, "w") as f:
        for k, v in losses.items():
            f.write("{}: {} \n".format(k,round(v,2)))


if __name__ == '__main__':
    results_dir = 'results/'

    _op = 'add'
    x, y = generate_data(1000, op=_op)
    x_Is = []
    y_Is = []

    N_tests = 100
    for i in range(N_tests):
        x_I, y_I = generate_data(100, op=_op)
        x_Is.append(x_I)
        y_Is.append(y_I)

    #x_E, y_E = generate_data(100, min=-10., max=10., op=_op)

    INPUT = 100
    OUTPUT = 1
    HIDDEN = [2]

    N_epochs = int(1e2)
    batch_size = 100

    models = ['nac', 'nalu', 'relu6', 'random']
    test(models, x, y, x_Is, y_Is, INPUT, OUTPUT, HIDDEN,
            batch_size, N_epochs, "interpolation.txt")
