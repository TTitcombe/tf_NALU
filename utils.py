import numpy as np
import tensorflow as tf

operations = {
'add': lambda x, y: x + y,
'subtract': lambda x,y: x - y,
'multiply': lambda x,y: x * y,
'divide': lambda x, y: x / (y+1e-6),
'square': lambda x, y: x**2,
'root': lambda x, y: np.sqrt(np.abs(x+1e-6))
}

def sumSubsetOfData(x, dim, n_per_subset, seed=42):
    if seed:
        np.random.seed(seed)

    _selection = np.random.choice(dim, n_per_subset*2)
    a = np.sum(x[:,_selection[:n_per_subset]], axis=1)
    b = np.sum(x[:,_selection[n_per_subset:]], axis=1)

    return a, b

def calcYFromOperation(x, op, dim=100, n_per_subset=10, seed=42):
    global operations
    func = operations[op]

    a, b = sumSubsetOfData(x, dim, n_per_subset, seed)
    y_flat = func(a,b)
    y = np.reshape(y_flat, (y_flat.shape[0], 1))

    return y

def gridSearch(x, y, x_test, y_test, Model,
               input_dim, hidden_dim, out_dim,
               N_epochs, LRs=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]):
    best_lr = None
    best_err = 100000000

    for lr in LRs:
        model = Model(input_dim, hidden_dim, out_dim, hyper={'lr':lr})
        test_error = model.train(x, y, x_test, y_test, N_epochs = N_epochs)
        if test_error < best_err:
            best_err = test_error
            best_lr = lr
        tf.reset_default_graph()

    return best_err, best_lr
