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


ACT_FUNCS = ["relu", "relu6", "elu", "leaky", "sigmoid", "tanh", "softplus", "None"]

LOSSES = {"Adam": tf.train.AdamOptimizer}
