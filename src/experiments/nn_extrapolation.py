import os
import matplotlib.pyplot as plt
import tensorflow as tf

from src.trainer import Trainer
from utils.collections import ACT_FUNCS
from utils.data import create_extrapolation_test_data


def get_data(n, min_val, max_val, n_test, min_test, max_test, seed=None):
    x = create_extrapolation_test_data(n, min_val, max_val, seed=seed)
    x_test = create_extrapolation_test_data(n_test, min_test, max_test)
    return x, x_test


def test(x, x_test, lr, **training_kwargs):
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
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

        results[func] = abs(y_extrapolation-x_test)

    return results


def plot_and_save(x, results, filename):
    for func_name, data in results.items():
        plt.scatter(x, data, label=func_name, alpha=0.8)

    plt.xlabel("X value")
    plt.ylabel("Absolute difference")
    plt.legend()
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    tf.enable_eager_execution()
    x, x_extrapolation = get_data(1000, -5, 5, 100, -20, 20, seed=42)
    results = test(x, x_extrapolation, 0.01, batch_size=100)

    save_filename = os.path.join("figures", "extrapolation_test_eager.png")  # Assuming you're in tf_NALU directory
    plot_and_save(x_extrapolation, results, save_filename)
