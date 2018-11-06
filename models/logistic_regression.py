# -*- coding: utf-8 -*-
import numpy as np
from models.base import Base
from models.helper import sigmoid, log_loss
from models.utils import input_check, target_check, fitted_check


class LogisticRegression(Base):
    """Implement a simple logistic regression"""
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit(self, X, y):
        """Fit the model using stochastic gradient descent"""
        X = input_check(X)
        y = target_check(y)

        n_features = X.shape[1]

        # Initialize weights
        coef = np.zeros(n_features)
        intercept = 0
        loss = log_loss(sigmoid(np.matmul(X, coef) + intercept), y)

        # Stochastic gradient descent
        while self.max_iter > 0:
            for x, y_true in zip(X, y):
                # Calculate prediction
                z = np.dot(x, coef) + intercept
                y_pred = sigmoid(z)

                error = y_pred - y_true

                # Calculate gradient
                gradient = x * error

                # Update weights
                coef = coef - self.learning_rate * gradient
                intercept = intercept - self.learning_rate * error

                self.max_iter -= 1

            loss = log_loss(sigmoid(np.matmul(X, coef) + intercept), y)

        self.coef_ = coef
        self.intercept_ = intercept
        self.log_loss_ = loss

        self.fitted = True

        return self

    @fitted_check
    def predict(self, X):
        X = input_check(X)
        z = np.matmul(X, self.coef_) + self.intercept_

        y_pred = sigmoid(z)

        return np.round(y_pred)
