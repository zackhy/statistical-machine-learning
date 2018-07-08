# -*- coding: utf-8 -*-
import numpy as np
from models.base import Base
from models.utils import input_check, target_check, fitted_check


class BernoulliNB(Base):
    """Bernoulli Naive Bayes Classifier that implements the fit(X, y) and predict(T) methods"""

    def fit(self, X, y):
        """
        Fit the Bernoulli Naive Bayes Classifier with input data X and target y
        :param X: Input data. An array-like object. Shape = (n_samples, n_features)
        :param y: Target. An array-like object. Shape = (n_samples, )
        :return: The fitted Bernoulli Naive Bayes Classifier
        """
        X = input_check(X)
        y = target_check(y)
        if np.min(X) < 0:
            raise ValueError('Input features should be greater than or equal to 0')

        # Convert the features to binary
        if np.max(X) > 1:
            X[X > 1] = 1

        self.uniq_classes_, num_docs = np.unique(y, return_counts=True)
        self.num_features_ = X.shape[1]

        # Compute prior probability for each class
        self.prior_prob_ = np.array([n / len(y) for n in num_docs])

        # Compute document frequencies for each term given a class
        doc_freq = np.vstack([(np.sum(X[y==c, :], axis=0)) for c in self.uniq_classes_])

        # Compute conditional probability for each term given a class.
        self.cond_prob_ = (doc_freq + 1) / (num_docs.reshape(-1, 1) + 2)

        self.fitted = True

        return self

    @fitted_check
    def predict(self, X):
        """
        Use the fitted Bernoulli Naive Bayes Classifier to make predictions
        :param X: Input data. An array-like object. Shape = (n_samples, n_features)
        :return: Predictions. A 1d numpy array. Shape = (n_samples, )
        """
        X = input_check(X)
        if X.shape[1] != self.num_features_:
            raise ValueError('Input X should have a shape of (,{})'.format(self.num_features_))

        preds = []
        for t in X:
            # Compute posterior probability
            post_prob = np.log(self.prior_prob_)
            likelihood = np.log(np.power(self.cond_prob_, t)) + np.log(np.power((1-self.cond_prob_), (1-t)))
            post_prob += np.sum(likelihood, axis=1)
            preds.append(np.argmax(post_prob))

        return np.array(self.uniq_classes_[preds])


class MultinomialNB(Base):
    """Multinomial Naive Bayes Classifier that implements the fit(X, y) and predict(T) methods"""

    def fit(self, X, y):
        """
        Fit the Multinomial Naive Bayes Classifier with input data X and target y
        :param X: Input data. An array-like object. Shape = (n_samples, n_features)
        :param y: Target. An array-like object. Shape = (n_samples, )
        :return: The fitted Multinomial Naive Bayes Classifier
        """
        X = input_check(X)
        y = target_check(y)
        if np.min(X) < 0:
            raise ValueError('Input features should be greater than or equal to 0')

        self.unique_classes_, num_docs = np.unique(y, return_counts=True)
        self.num_features_ = X.shape[1]

        # Compute the prior probability
        self.prior_prob_ = np.array([(n / len(y)) for n in num_docs])

        # Compute the term frequencies for each term given a class
        term_freq = np.vstack([np.sum(X[y == c, :], axis=0) for c in self.unique_classes_])
        # Add one to avoid zero
        term_freq = term_freq + 1

        # Compute the total term frequencies for each class
        tot_freq = np.sum(term_freq, axis=1)

        # Compute the conditional probability
        self.cond_prob_ = term_freq / tot_freq.reshape(-1, 1)

        self.fitted = True

        return self

    @fitted_check
    def predict(self, X):
        """
        Use the fitted Multinomial Naive Bayes Classifier to make predictions
        :param X: Input data. An array-like object. Shape = (n_samples, n_features)
        :return: Predictions. A 1d numpy array. Shape = (n_samples, )
        """
        X = input_check(X)
        if X.shape[1] != self.num_features_:
            raise ValueError('Input X should have a shape of (?,{})'.format(self.num_features_))

        preds = []
        for t in X:
            # Compute posterior probability
            post_prob = np.log(self.prior_prob_)
            post_prob += np.sum(np.log(np.power(self.cond_prob_, t)), axis=1)
            preds.append(np.argmax(post_prob))

        return np.array(self.unique_classes_[preds])
