import os
import sys
from typing import List
from numpy import array_equal, array, mean
import tensorflow as tf
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utilis.models import decoding_pred


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
            self.model.add(tf.keras.layers.Dense(200, activation="relu"))
            self.model.add(tf.keras.layers.Dropout(rate=0.2))
        self.model.add(tf.keras.layers.Dense(self.outputDim[0]*self.outputDim[1], activation="sigmoid"))
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                           optimizer='adam')
#                          metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)])
        assert labels[0][0].shape == tf.TensorShape([self.outputDim[0]*self.outputDim[1]]), "Incompatible output shapes"
        self.model.fit(embeddings[0],
                       labels[0],
                       epochs=500,
                       validation_data=(embeddings[1], labels[1]),
                       callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)])
        test_loss = self.model.evaluate(embeddings[2], labels[2])
        labelsHat = self.model.predict(embeddings[2])
        yhat = decoding_pred(labelsHat.reshape(labelsHat.shape[0],
                                               self.outputDim[1],
                                               self.outputDim[0]))
        y = array(labels[2]).reshape(labelsHat.shape[0],
                                     self.outputDim[1],
                                     self.outputDim[0])
        return mean([int(array_equal(y[i][j],
                                     yhat[i][j])) for i in range(y.shape[0]) for j in range(y[i].shape[0])]), labelsHat


class GRU:
    def __init__(self) -> None:
        pass

    def evaluation(self,
                   embeddings: List[tf.Tensor],
                   labels: List[tf.Tensor]) -> float:
        pass
