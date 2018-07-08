# -*- coding: utf-8 -*-
import numpy as np


def fitted_check(func):

    def wrapper(self, *args, **kw):
        if not hasattr(self, 'fitted'):
            raise AttributeError("This model instance is not fitted yet. Call 'fit' first.")
        return func(self, *args, **kw)

    return wrapper


def input_check(X):
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.ndim != 2:
        raise ValueError('Input X should be a 2d array-like object. Shape = (n_samples, n_features)')

    return X


def target_check(y):
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if y.ndim != 1:
        raise ValueError('Input y should be a 1d array-like object. Shape = (n_samples, )')

    return y
