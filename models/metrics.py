# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import confusion_matrix as conm

def confusion_matrix(y_true, y_pred):
    """
    :param y_true: True targets. An array-like object. Shape = (n_samples, )
    :param y_pred: Predicted values. An array-like object. Shape = (n_samples, )
    :return: Consufison matrix.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_pred must have the same shape.')

    labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    if not np.isin(pred_labels, labels).all():
        raise ValueError('All the labels in y_pred must be in y_true')

    label_to_index = dict((l, i) for i, l in enumerate(labels))

    # Convert labels to index
    y_true = [label_to_index.get(l) for l in y_true]
    y_pred = [label_to_index.get(l) for l in y_pred]

    # Confustion matrix
    cm = np.zeros((len(labels), len(labels)), dtype=np.int32)

    for row, col in zip(y_true, y_pred):
        cm[row][col] += 1

    return cm
