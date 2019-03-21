import tensorflow as tf

from src.trainer import Trainer
from utils.collections import ACT_FUNCS
from utils.data import create_extrapolation_test_data


def get_data(n, min_val, max_val, n_test, min_test, max_test, seed=None):
    x = create_extrapolation_test_data(n, min_val, max_val, seed=seed)
    x_test = create_extrapolation_test_data(n_test, min_test, max_test)
    return x, x_test


def test(x, x_test, lr, **training_kwargs):
    if not tf.executing_eagerly():
        raise RuntimeError("You must enable tensorflow eager execution before running this function!")

    INPUT_DIM = 1
    OUTPUT_DIM = 1
    HIDDEN_DIM = [8, 8, 8]

    results = {}

    for func in ACT_FUNCS:
        print("\n--------- TESTING {} ---------\n".format(func))
        trainer = Trainer(lr, "MLP", "Adam",
                          INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, act_func=func)
        trainer.train(x, x, x_test, x_test, **training_kwargs)
        y_extrapolation = trainer.model(x_test)

        results[func] = y_extrapolation

    return results


if __name__ == "__main__":
    tf.enable_eager_execution()
    x, x_extrapolation = get_data(5000, -5, 5, 1000, -20, 20, seed=42)
    results = test(x, x_extrapolation, 0.1, batch_size=100)
