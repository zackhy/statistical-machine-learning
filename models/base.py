import numpy as np

def fitted_check(func):

    def wrapper(self, *args, **kw):
        if not hasattr(self, 'fitted'):
            raise AttributeError("This model instance is not fitted yet. Call 'fit' first.")
        return func(self, *args, **kw)

    return wrapper


def input_check(func):

    def wrapper(self, X, y, *args, **kw):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if X.ndim != 2:
            raise ValueError('Input X should be a 2d array-like object. Shape = (n_samples, n_features)')
        if y.ndim != 1:
            raise ValueError('Input y should be a 1d array-like object. Shape = (n_samples, )')
        return func(self, X, y, *args, **kw)

    return wrapper


class Base(object):
    def __int__(self):
        pass

    def fit(self, X, y):
        return NotImplementedError

    def predict(self, T):
        return NotImplementedError

    @input_check
    def score(self, X, y_true):
        """
        :param X: Input data. An array-like object. Shape = (n_samples, n_features)
        :param y_true: Target. An array-like object. Shape = (n_samples, )
        :return: Accuracy score.
        """
        preds = self.predict(X)
        return (preds == y_true).mean()
