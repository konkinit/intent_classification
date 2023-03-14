import os
import sys
from typing import List, Tuple
from numpy import array_equal, array, mean, ndarray
import tensorflow as tf
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utils.models_utils import (
    decoding_pred,
    _confusion_matrix)


class Decoder:
    def __init__(self,
                 embeddingsDim: int,
                 output_dimension: List[int],
                 *args) -> None:
        self.model = tf.keras.models.Sequential()
        self.embeddingsDim = embeddingsDim
        self.outputDim = output_dimension
        self.nLayers = args[0]
        self.fDropout = args[1]

    def _fit(self,
             embeddings: List[tf.Tensor],
             labels: List[tf.Tensor]) -> NotImplemented:
        """
        Fit the MPL-based decoder
        """
        self._build()
        self.model.fit(embeddings[0],
                       labels[0],
                       batch_size=16,
                       epochs=500,
                       verbose=0,
                       validation_data=(embeddings[1], labels[1]),
                       callbacks=[tf.keras.callbacks.EarlyStopping(
                                  monitor="val_loss", patience=10)])


class MLP(Decoder):
    def __init__(self,
                 embeddingsDim: int,
                 output_dimension: List[int],
                 *args) -> None:
        super().__init__(embeddingsDim, output_dimension, args[0], args[1])

    def _nNeurons(self,
                  inputDim: int,
                  outputDim: int) -> int:
        """
        Return the number of neurons for hidden layers according
        to the statement << The number of hidden neurons should be 2/3
        the size of the input layer, plus the size of the output layer >>
        """
        return int((2/3)*inputDim + outputDim)

    def _build(self) -> None:
        """
        Build the model architecture
        """
        self.model.add(
            tf.keras.layers.InputLayer(input_shape=(self.embeddingsDim,)))
        nNeurons = self._nNeurons(
                        self.embeddingsDim,
                        self.outputDim[0]*self.outputDim[1])
        for _ in range(self.nLayers):
            self.model.add(tf.keras.layers.Dense(nNeurons, activation="relu"))
            self.model.add(tf.keras.layers.Dropout(rate=self.fDropout))
        self.model.add(tf.keras.layers.Dense(
                       self.outputDim[0]*self.outputDim[1],
                       activation="sigmoid"))
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                           optimizer='adam')

    def _inference(self,
                   embeddings: List[tf.Tensor],
                   labels: List[tf.Tensor],
                   list_labels: List[str]) -> Tuple[float, ndarray]:
        self._fit(embeddings, labels)
        _labelsHat = self.model.predict(embeddings[2])
        _labelsHat = decoding_pred(_labelsHat.reshape(
            _labelsHat.shape[0], self.outputDim[1], self.outputDim[0]))
        _labels = array(labels[2]).reshape(
            labels[2].shape[0], self.outputDim[1], self.outputDim[0])
        loss_list = [int(array_equal(
            _labels[i][j], _labelsHat[i][j])) for i in range(_labels.shape[0])
            for j in range(_labels[i].shape[0])]
        return mean(loss_list), _confusion_matrix(
            _labels, _labelsHat, list_labels)


class SequentialGRU(Decoder):
    def __init__(self,
                 embeddingsDim: int,
                 output_dimension: List[int],
                 *args) -> None:
        super().__init__(embeddingsDim, output_dimension, args[0], args[1])

    def _build(self) -> None:
        """
        Build the sequential GRU architecture
        """
        decoder_inputs = tf.keras.layers.Input(
                                shape=(self.outputDim[1],
                                       self.embeddingsDim))
        decoder_gru = tf.keras.layers.GRU(
                                self.outputDim[0],
                                return_sequences=True)
        hidden_states = decoder_gru(decoder_inputs)
        self.model = tf.keras.models.Model(
                    inputs=decoder_inputs, outputs=hidden_states)
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                           optimizer='adam')

    def _inference(self,
                   embeddings: List[tf.Tensor],
                   labels: List[tf.Tensor],
                   list_labels: List[str]) -> Tuple[float, ndarray]:
        self._fit(embeddings, labels)
        _labelsHat = self.model.predict(embeddings[2])
        _labelsHat = decoding_pred(_labelsHat)
        _labels = labels[2]
        loss_list = [int(array_equal(
            _labels[i][j], _labelsHat[i][j])) for i in range(_labels.shape[0])
            for j in range(_labels[i].shape[0])]
        return mean(loss_list), _confusion_matrix(
            _labels, _labelsHat, list_labels)
