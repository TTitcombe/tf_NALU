import numpy as np


def create_static_data(n, dim, minVal, maxVal, minTest, maxTest, func, n_subset=15, seed=42):
    if seed:
        np.random.seed(seed)

    split = np.random.randint((n_subset+1)//2, n_subset)
    #  how to partition the n_subset dims
    #  we take an int from half to full so that the first summed subset
    #  has at least at many dims
    # this ensures a-b > 0 and a/b > 1

    x = np.random.uniform(minVal, maxVal, size=(n, dim))
    x_test = np.random.uniform(minTest, maxTest, size=(100, dim))

    selection = np.random.choice(dim, n_subset)
    _selection = selection[:split]
    _selection2 = selection[split:]

    a = np.sum(x[:, _selection], axis=1)
    b = np.sum(x[:, _selection2], axis=1)

    a_test = np.sum(x_test[:, _selection], axis=1)
    b_test = np.sum(x_test[:, _selection2], axis=1)

    y_flat = func(a, b)
    y = np.reshape(y_flat, (y_flat.shape[0], 1))

    y_flat_test = func(a_test, b_test)
    y_test = np.reshape(y_flat_test, (y_flat_test.shape[0], 1))

    return x, y, x_test, y_test


def create_extrapolation_test_data(n, min_val, max_val, seed=None):
    if seed:
        np.random.seed(seed)

    return np.random.uniform(min_val, max_val, size=(n, 1))
