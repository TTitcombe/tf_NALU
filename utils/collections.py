import numpy as np
import tensorflow as tf

OPERATIONS = {
              'add': lambda x, y: x + y,
              'subtract': lambda x,y: x - y,
              'multiply': lambda x,y: x * y,
              'divide': lambda x, y: x / (y+1e-6),
              'square': lambda x, y: x**2,
              'root': lambda x, y: np.sqrt(np.abs(x+1e-6))
             }


ACT_FUNCS = ["relu", "elu", "leaky", "sigmoid", "tanh", "softplus", "None"]
# Note, known issue with eager relu6, so it has been ommitted from our test

MODELS = ["MLP", "NAC", "NALU"]

OPTIMS = {"Adam": tf.train.AdamOptimizer,
          "RMS": tf.train.RMSPropOptimizer}
