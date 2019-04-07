import os
import numpy as np
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
    filename = os.path.join("results", "static_arithmetic_eager.txt")

    for model in MODELS:
        print("\nTraining {} model now....".format(model))
        if model in ("MLP", "relu", "None"):
            model_args = [INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM]
            model = "MLP"
        else:
            model_args = [OUTPUT_DIM, HIDDEN_DIM]
        trainer = Trainer(lr, model, "RMS",
                          *model_args)
        trainer.train(x, y, x_test, y_test, **training_kwargs)
        extrapolation_loss = trainer.loss(x_test, y_test)

        func_results[model] = extrapolation_loss

    return func_results


def save_results(results, file_to_save, op_name):
    with open(file_to_save, "a") as f:
        f.write("\nOperation {}\n".format(op_name))
        max_loss = -1
        for _, loss in results.items():
            loss = float(loss)
            if not np.isnan(loss) and loss > max_loss:
                max_loss = loss
        for model_name, model_loss in results.items():
            model_loss = float(model_loss)
            f.write("{}: {:.3f} | {:.3f}\n".format(model_name, model_loss, model_loss/max_loss))


if __name__ == "__main__":
    tf.enable_eager_execution()

    # Data variables
    n = 500
    dim = 100  # for a simple test, set dim = 2
    min_val = 0
    max_val = 1000
    min_test = 1000   # to test interpolation, set min_test = min_val
    max_test = 10000  # to test interpolation, set max_test = max_val
    seed = 42

    filename = os.path.join("results", "static_arithmetic_eager.txt")  # Assuming you're in tf_NALU directory
    with open(filename, "a") as f:
        f.write("Results displayed are mean-squared error.\n")
        f.write("Data is written as absolute | relative, where relative is "
                "absolute scores normalised by the maximum score for that function.\n")

    # Loop through arithmetic operations
    for op_name, func in OPERATIONS.items():
        x, y, x_test, y_test = create_static_data(n, dim, min_val, max_val, min_test,
                                                  max_test, func, n_subset=15, seed=seed)
        op_results = test_func(x, y, x_test, y_test, op_name, 0.01, batch_size=100)
        save_results(op_results, filename, op_name)
