# -*- coding: utf-8 -*-
from models.utils import input_check, target_check, fitted_check


class Base(object):
    def __int__(self):
        pass

    def fit(self, X, y):
        return NotImplementedError

    def predict(self, T):
        return NotImplementedError

    @fitted_check
    def score(self, X, y_true):
        """
        :param X: Input data. An array-like object. Shape = (n_samples, n_features)
        :param y_true: Target. An array-like object. Shape = (n_samples, )
        :return: Accuracy score.
        """
        X = input_check(X)
        y_true = target_check(y_true)
        preds = self.predict(X)
        return (preds == y_true).mean()
