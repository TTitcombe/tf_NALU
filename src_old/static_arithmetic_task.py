from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from nac import NAC
from nalu import NALU
from mlp import MLP
from tf_NALU.Utils.utils import operations, create_data


def testAll():
    results_dir = "results/"
    filename = "static_arithmetic_test.txt"

    for _op, op_func in operations.items():
        x, y, x_test, y_test = create_data(50000, 100, 0, 1000, 1000, 10000, op_func)
        print("In operation {}".format(_op))
        print("NAC")
        model = NAC(100, 2, 1)
        nac_err = model.train(x, y, x_test, y_test)
        tf.reset_default_graph()

        counter = 0
        nalu_err = np.nan
        while np.isnan(nalu_err) and counter < 10:
            # NALU can often become NaN
            counter += 1
            print("NALU")
            model = NALU(100, 2, 1)
            nalu_err = model.train(x, y, x_test, y_test)
            tf.reset_default_graph()
        print("MLP")
        model = MLP(100, 2, 1)
        random_err, _ = model.validate(x_test, y_test)
        mlp_err = model.train(x, y, x_test, y_test)
        tf.reset_default_graph()

        max_score = np.nanmax([nac_err, nalu_err, random_err, mlp_err])

        with open(results_dir + filename, "a") as f:
            f.write("\n{}\n".format(_op))
            f.write("NAC err: {} | {}\n".format(nac_err, nac_err/max_score))
            f.write("NALU err: {} | {}\n".format(nalu_err, nalu_err/max_score))
            f.write("MLP err: {} | {}\n".format(mlp_err, mlp_err/max_score))
            f.write("Random err: {} | {}\n".format(random_err, random_err/max_score))


if __name__ == "__main__":
    pass
    #testAll()

