from typing import List
import tensorflow as tf
from torch import Tensor


class MLP:
    def __init__(self,
                 embeddingsDim: int,
                 n_layers: int,
                 T: int,
                 f_dropout: float) -> None:
        self.model = tf.keras.models.Sequential()
        self.embeddingsDim = embeddingsDim
        self.nLayers = n_layers
        self.T = T
        self.fDropout = f_dropout

    def evaluation(self,
                   embeddings: List[Tensor],
                   labels: List[Tensor]) -> Tensor:
        """
        Fit the MPL-based decoder and Evaluate the score on the
        test split
        """
        self.model.add(tf.keras.layers.InputLayer(input_shape=(1, self.embeddingsDim)))
        for _ in range(self.nLayers-1):
            self.model.add(tf.keras.layers.Dense(256, activation="relu"))
        self.model.add(tf.keras.layers.Dense(self.T, activation="softmax"))
        self.model.add(tf.keras.layers.Dropout(self.fDropout))
        self.model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
        self.model.fit(embeddings[0], labels[0],
                       validation_data=(embeddings[1], labels[1]))
        return 100*(1 - self.model.evaluate(embeddings[2], labels[2]))
