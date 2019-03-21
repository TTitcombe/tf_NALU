from src.trainer import Trainer
from utils.collections import ACT_FUNCS
from utils.data import create_extrapolation_test_data


def get_data(n, min_val, max_val, n_test, min_test, max_test, seed=None):
    x = create_extrapolation_test_data(n, min_val, max_val, seed=seed)
    x_test = create_extrapolation_test_data(n_test, min_test, max_test)
    return x, x_test


def test(*args, **kwargs):
    x, x_test = get_data(*args, **kwargs)
