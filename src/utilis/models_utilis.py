from numpy import ndarray, array, vstack, apply_along_axis
from torch import sum, clamp


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
