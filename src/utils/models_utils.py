from numpy import (
    ndarray,
    array,
    vectorize,
    vstack,
    apply_along_axis
)
from sklearn.metrics import confusion_matrix
from torch import sum, clamp
import tensorflow as tf


def decoding_pred(pred: ndarray):
    return vstack([array([
        apply_along_axis(lambda a: 1*(a == a.max()), 1,
                         pred[i])]) for i in range(pred.shape[0])])


def mean_pooling(model_output, attention_mask):
    """
    Mean Pooling - Take attention mask into account for correct averaging
    """
    token_embeddings = model_output[0]
    input_mask_expanded = (attention_mask
                           .unsqueeze(-1)
                           .expand(token_embeddings.size())
                           .float())
    sum_ = sum(token_embeddings * input_mask_expanded, 1)
    clamp_ = clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_/clamp_


def _reshape_decode(x: tf.Tensor, list_labels: list):
    x_reshaped = tf.reshape(x, [x.shape[0]*x.shape[1], x.shape[2]])
    return vectorize((lambda x: list_labels[x]))(
        array(x_reshaped).argmax(axis=1))


def _confusion_matrix(y: tf.Tensor,
                      yHat: tf.Tensor,
                      list_labels: list) -> ndarray:
    """
    Compute the confusion matrix after reshape inputs

    Args:
        y (tf.Tensor): y true
        yHat (tf.Tensor): y predicted
        list_labels (list): list of labls for annotation

    Returns:
        ndarray: confusion matrix
    """
    labels, labelsHat = (
                    _reshape_decode(y, list_labels),
                    _reshape_decode(yHat, list_labels)
                    )
    return confusion_matrix(
        labels, labelsHat, labels=list_labels)
