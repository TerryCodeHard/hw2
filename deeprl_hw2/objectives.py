"""Loss functions."""

import tensorflow as tf
import semver
import numpy as np
import keras.backend as K

'''

def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    return tf.losses.huber_loss(y_true, y_pred, weights=1.0, delta=max_grad, scope=None, loss_collection=tf.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
'''
def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.
    See https://en.wikipedia.org/wiki/Huber_loss
    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.
    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    assert max_grad > 0., 'The maximum gradient should be larger than 0 while calculating huber loss'

    diff = y_true - y_pred
    if np.isinf(max_grad):  # handle edge case
        return .5 * K.square(diff)

    condition = K.abs(diff) < max_grad
    squared_loss = .5 * K.square(diff)
    linear_loss = max_grad * (K.abs(diff) - .5 * max_grad)
    if hasattr(tf, 'select'):
        return tf.select(condition, squared_loss, linear_loss)  # if difference less than max_grad, linear loss
    else:
        return tf.where(condition, squared_loss, linear_loss)  # if not, square loss


def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    temp = huber_loss(y_true, y_pred, weights=1.0, max_grad=1.)
    return tf.reduce_mean(temp)
