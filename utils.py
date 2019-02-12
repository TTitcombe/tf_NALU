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

def gridSearch(x, y, x_test, y_test, Model,
               input_dim, hidden_dim, out_dim,
               N_epochs, LRs=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]):
    best_lr = None
    best_err = np.inf

    for lr in LRs:
        model = Model(input_dim, hidden_dim, out_dim, hyper={'lr':lr})
        test_error = model.train(x, y, x_test, y_test, N_epochs = N_epochs)
        if not np.isnan(test_error):
            if test_error < best_err:
                best_err = test_error
                best_lr = lr
        tf.reset_default_graph()

    return best_err, best_lr

def create_data(n, dim, minVal, maxVal, minTest, maxTest, func, n_subset=15, seed=42):
    if seed:
        np.random.seed(42)

    split = np.random.randint((n_subset+1)//2,n_subset)
    #  how to partition the n_subset dims
    #  we take an int from half to full so that the first summed subset
    #  has at least at many dims
    # this ensures a-b > 0 and a/b > 1

    x = np.random.uniform(minVal, maxVal, size=(n, dim))
    x_test = np.random.uniform(minTest, maxTest, size=(1000, dim))

    selection = np.random.choice(dim, n_subset)
    _selection = selection[:split]
    _selection2 = selection[split:]

    a = np.sum(x[:,_selection], axis=1)
    b = np.sum(x[:,_selection2], axis=1)

    a_test = np.sum(x_test[:,_selection], axis=1)
    b_test = np.sum(x_test[:,_selection2], axis=1)

    y_flat = func(a,b)
    y = np.reshape(y_flat, (y_flat.shape[0], 1))

    y_flat_test = func(a_test,b_test)
    y_test = np.reshape(y_flat_test, (y_flat_test.shape[0], 1))

    return x, y, x_test, y_test
