def fitted_check(func):

    def wrapper(self, *args, **kw):
        if not hasattr(self, 'fitted'):
            raise AttributeError("This model instance is not fitted yet. Call 'fit' first.")
        return func(self, *args, **kw)

    return wrapper

class Base(object):
    def __int__(self):
        pass

    def fit(self, X, y):
        return NotImplementedError

    def predict(self, T):
        return NotImplementedError

    def score(self, T, y_true):
        """
        :param T: Input data
        :param y_true: Target
        :return: Accuracy score
        """
        preds = self.predict(T)
        return (preds == y_true).mean()
