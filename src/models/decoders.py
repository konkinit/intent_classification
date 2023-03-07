from typing import List
from numpy import ndarray, vstack, array, array_equal, apply_along_axis, mean
import tensorflow as tf


class MLP:
    def __init__(self,
                 embeddingsDim: int,
                 n_layers: int,
                 output_dimension: List[int],
                 f_dropout: float) -> None:
        self.model = tf.keras.models.Sequential()
        self.embeddingsDim = embeddingsDim
        self.nLayers = n_layers
        self.outputDim = output_dimension
        self.fDropout = f_dropout

    @staticmethod
    def performance(y, yHat) -> float:
        pass

    def evaluation(self,
                   embeddings: List[tf.Tensor],
                   labels: List[tf.Tensor]) -> float:
        """
        Fit the MPL-based decoder and Evaluate the score on the
        test split
        """
        self.model.add(tf.keras.layers.InputLayer(input_shape=(self.embeddingsDim,)))
        for _ in range(self.nLayers-1):
            self.model.add(tf.keras.layers.Dense(64, activation="relu"))
        self.model.add(tf.keras.layers.Dense(self.outputDim[0]*self.outputDim[1], activation="softmax"))
        # self.model.add(tf.keras.layers.Dropout(self.fDropout))
        self.model.compile(loss="categorical_crossentropy", optimizer='adam')
        assert labels[0][0].shape == tf.TensorShape([self.outputDim[0]*self.outputDim[1]]), "Incompatible output shapes"
        self.model.fit(embeddings[0], labels[0],
                       validation_data=(embeddings[1], labels[1]))
        labelsHat = self.model.predict(embeddings[2])

        def decoding_pred(pred: ndarray):
            return vstack([array([apply_along_axis(lambda a: 1*(a == a.max()), 1,
                                                   pred[i])]) for i in range(pred.shape[0])])
        yhat = decoding_pred(labelsHat.reshape(labelsHat.shape[0], self.outputDim[1], self.outputDim[0]))
        y = array(labels[2]).reshape(labelsHat.shape[0], self.outputDim[1], self.outputDim[0])
        return mean([int(array_equal(y[i], yhat[i])) for i in range(y.shape[0])])
