import numpy as np
import tensorflow as tf


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
