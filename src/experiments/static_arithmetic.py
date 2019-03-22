import os
import tensorflow as tf

from src.trainer import Trainer
from utils.collections import MODELS, OPERATIONS
from utils.data import create_static_data


def test_func(x, y, x_test, y_test, func_name, lr, **training_kwargs):
    if not tf.executing_eagerly():
        raise RuntimeError("You must enable tensorflow eager execution before running this function!")

    INPUT_DIM = x.shape[1]
    OUTPUT_DIM = y.shape[1]
    HIDDEN_DIM = [2]

    func_results = {}
    print("\n--------- TESTING {} ---------\n".format(func_name))
    for model in MODELS:
        print("Training {} model now....\n".format(model))
        trainer = Trainer(lr, model, "Adam",
                          INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM)
    trainer.train(x, y, x_test, y_test, **training_kwargs)
    extrapolation_loss = trainer.loss(x_test, y_test)

    func_results[func_name] = extrapolation_loss

    return func_results


def save_results(results, file_to_save):
    for func_name, func_dict in results.items():
        max_val = max(func_dict.keys(), key=(lambda k: func_dict[k]))


if __name__ == "__main__":
    tf.enable_eager_execution()

    # Data variables
    n = 50000
    dim = 100  # for a simple test, set dim = 2
    min_val = 0
    max_val = 1000
    min_test = 1000   # to test interpolation, set min_test = min_val
    max_test = 10000  # to test interpolation, set max_test = max_val
    seed = 42

    filename = os.path.join("results", "static_arithmetic.txt")  # Assuming you're in tf_NALU directory

    results = {}
    # Loop through arithmetic operations
    for op_name, func in OPERATIONS.items():
        x, y, x_test, y_test = create_static_data(n, dim, min_val, max_val, min_test,
                                                  max_test, func, n_subset=15, seed=seed)
        results[op_name] = test_func(x, y, x_test, y_test, 0.1, batch_size=100)

    save_results(results, filename)

