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
'root': lambda x, y: np.sqrt(x + 1e-6)
}

def generate_data(n,z=2, min=-1., max=1., select = 1, op = 'add', seed=42):
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

def test(models, x_train, y_train, x_tests, y_tests,
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
            trainer.train(x, y, x_tests[0], y_tests[0], batch_size, N_epochs)

        for x_test, y_test in zip(x_tests, y_tests):
            err, loss = trainer._validate(x_test, y_test)
            losses[model_type] += loss
        losses[model_type] /= N_test_data

    with open(results_dir + filename, "w") as f:
        for k, v in losses.items():
            f.write("{}: {} \n".format(k,round(v,5)))


if __name__ == '__main__':
    results_dir = 'results/'

    INPUT = 2
    OUTPUT = 1
    HIDDEN = [2]
    MIN = -10.
    MAX = 10.

    N_epochs = int(1e4)
    batch_size = 100
    for _op in ['add', 'subtract', 'multiply', 'divide', 'square', 'root']:
        #_op = 'subtract'
        x, y = generate_data(1000, z=INPUT, op=_op)
        x_tests = []
        y_tests = []

        N_tests = 100
        for i in range(N_tests):
            x_test, y_test = generate_data(100,min=MIN, max=MAX, op=_op)
            x_tests.append(x_test)
            y_tests.append(y_test)

        models = ['nac', 'nalu', 'relu6', 'random']
        test(models, x, y, x_tests, y_tests, INPUT, OUTPUT, HIDDEN,
                batch_size, N_epochs, "E_{}_small.txt".format(_op))
