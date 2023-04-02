from pandas import DataFrame
from matplotlib.pyplot import (
    figure,
    savefig,
    show
)
from numpy import (
    ndarray,
    array,
    vectorize,
    vstack,
    apply_along_axis
)
from seaborn import set, heatmap, color_palette
from sklearn.metrics import confusion_matrix
from torch import sum, clamp
import tensorflow as tf


def decoding_pred(pred: ndarray) -> ndarray:
    """Transform continious outputs to binairies

    Args:
        pred (ndarray): array with the prediction as continious values

    Returns:
        ndarray: array with the prediction in binary format
    """
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


def fig_path(_input: tuple) -> str:
    """Return the figure path

    Args:
        _input (tuple): pipeline inputs

    Returns:
        str: path to save confusion matrix figure
    """
    _root = "./data/figs/"
    _input10 = _input[1].split('-')[0]
    return f"{_root}{_input[0][0]}_{_input10}_{_input[2][0]}.pdf"


def _plot_confusion_matrix(
        cm: ndarray,
        _input: tuple,
        _list_labels: list) -> None:
    """Plot and show the confusion matrix

    Args:
        cm (ndarray): confusion matrix in array format
        _input (tuple): pipeline inputs
        _list_labels (list): list of labels of the datasets in _input
    """
    df_cm = DataFrame(
                cm,
                _list_labels,
                _list_labels)
    d = len(_list_labels)
    figure(figsize=(d, d))
    set(font_scale=1)
    heatmap(
        df_cm,
        annot=True,
        annot_kws={"size": 8},
        cmap=color_palette("ch:start=.2,rot=-.3", as_cmap=True),
        cbar=False)
    savefig(
        fig_path(_input),
        bbox_inches='tight')
    show()
