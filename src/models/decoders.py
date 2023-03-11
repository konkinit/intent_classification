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

    def _nNeurons(self,
                  inputDim: int,
                  outputDim: int) -> List[int]:
        """
        Return a list of number of neurons for
        hidden layers
        """
        pass

    def evaluate(self,
                 embeddings: List[tf.Tensor],
                 labels: List[tf.Tensor]) -> float:
        """
        Fit the MPL-based decoder and Evaluate the score on the
        test split
        """
        self.model.add(tf.keras.layers.InputLayer(input_shape=(self.embeddingsDim,)))
        """
        nNeurons_list = self._nNeurons(self.embeddingsDim,
                                       self.outputDim[0]*self.outputDim[1])
        """
        for _ in range(self.nLayers):
            self.model.add(tf.keras.layers.Dense(200, activation="relu"))
            self.model.add(tf.keras.layers.Dropout(rate=self.fDropout))
        self.model.add(tf.keras.layers.Dense(
                       self.outputDim[0]*self.outputDim[1],
                       activation="sigmoid"))
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                           optimizer='adam')
        assert labels[0][0].shape == tf.TensorShape([
            self.outputDim[0]*self.outputDim[1]]), "Incompatible output shapes"
        self.model.fit(embeddings[0],
                       labels[0],
                       batch_size=1,
                       epochs=500,
                       validation_data=(embeddings[1], labels[1]),
                       callbacks=[tf.keras.callbacks.EarlyStopping(
                                  monitor="val_loss", patience=10)])
        labelsHat = self.model.predict(embeddings[2])
        self.model.reset_states()
        yhat = decoding_pred(labelsHat.reshape(labelsHat.shape[0],
                                               self.outputDim[1],
                                               self.outputDim[0]))
        y = array(labels[2]).reshape(labelsHat.shape[0],
                                     self.outputDim[1],
                                     self.outputDim[0])
        return mean([int(array_equal(
            y[i][j], yhat[i][j])) for i in range(y.shape[0])
            for j in range(y[i].shape[0])])


class GRU:
    def __init__(self) -> None:
        pass

    def evaluate(self,
                 embeddings: List[tf.Tensor],
                 labels: List[tf.Tensor]) -> float:
        pass
