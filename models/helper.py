# -*- coding: utf-8 -*-
import numpy as np


def sigmoid(n):
    return 1 / (1 + np.exp(-n))


def log_loss(probs, y_true):
    probs = np.array(probs)
    y_true = np.array(y_true)

    term_1 = np.dot(y_true, np.log(probs))
    term_2 = np.dot(1 - y_true, np.log(1 - probs))

    return - (1 / len(y_true)) * (term_1 + term_2)
