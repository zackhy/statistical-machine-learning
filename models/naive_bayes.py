import numpy as np
from models.base import Base, fitted_check


class BernoulliNB(Base):
    """Bernoulli Naive Bayes Classifier that implements the fit(X, y) and predict(T) methods"""

    def fit(self, X, y):
        """
        Fit the Bernoulli Naive Bayes Classifier with input data X and target y
        :param X: Input data. A 2d array
        :param y: Target. A 1d array
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if X.ndim != 2:
            raise ValueError('Input X should be a 2d array')
        if y.ndim != 1:
            raise ValueError('Input y should be a 1d array')
        if np.max(X) > 1 or np.min(X) < 0:
            raise ValueError('Input X should only contain binary features')

        self.uniq_classes_, num_docs = np.unique(y, return_counts=True)
        self.num_features_ = X.shape[1]
        num_classes = len(self.uniq_classes_)

        # Compute prior probability for each class
        self.prior_prob_ = np.array([n / len(y) for n in num_docs])

        assert self.prior_prob_.shape == (num_classes, )

        # Compute document frequencies for each term given a class
        doc_freq = np.vstack([(np.sum(X[y==c, :], axis=0)) for c in self.uniq_classes_])

        assert doc_freq.shape == (num_classes, self.num_features_)

        # Compute conditional probability for each term given a class.
        self.cond_prob_ = (doc_freq + 1) / (num_docs.reshape(-1, 1) + 2)
        # print(self.cond_prob)

        assert self.cond_prob_.shape == (num_classes, self.num_features_)

        self.fitted = True

        return self

    @fitted_check
    def predict(self, T):
        """
        Use the fitted Bernoulli Naive Bayes Classifier to make predictions
        :param T: Input data. A 2d array
        :return: Predictions. A numpy 1d array
        """
        if not isinstance(T, np.ndarray):
            T = np.array(T)
        if T.ndim != 2:
            raise ValueError('Input T should be a 2d numpy array')
        if T.shape[1] != self.num_features_:
            raise ValueError('Input T should have a shape of (,{})'.format(self.num_features_))

        preds = []
        for t in T:
            # Compute posterior probability
            post_prob = np.log(self.prior_prob_)
            likelihood = np.log(np.power(self.cond_prob_, t)) + np.log(np.power((1-self.cond_prob_), (1-t)))
            post_prob += np.sum(likelihood, axis=1)
            preds.append(np.argmax(post_prob))

        return np.array(self.uniq_classes_[preds])
