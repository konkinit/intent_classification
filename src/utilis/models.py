from numpy import ndarray, array, vstack, apply_along_axis


def decoding_pred(pred: ndarray):
    return vstack([array([
        apply_along_axis(lambda a: 1*(a == a.max()), 1,
                         pred[i])]) for i in range(pred.shape[0])])
