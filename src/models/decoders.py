from typing import List
import tensorflow as tf
from torch import Tensor

class MLP:
    def __init__(self,
                 n_units: int,
                 n_layers: int,
                 f_dropout: float) -> None:
        self.model = tf.keras.models.Sequential()
        self.nUnints = n_units
        self.nLayers = n_layers
        self.fDropout = f_dropout

    def evaluation(self,
                   embeddings: List[Tensor],
                   labels: List[Tensor]) -> Tensor:
        """
        Fit the MPL-based decoder and Evaluate the score on the
        test split 
        """
        for _ in range(self.nLayers):
            self.model.add(tf.keras.layers.Dense(self.nUnints,
                                                 activation="relu"))
        self.model.add(tf.keras.layers.Dropout(self.fDropout))
        optimizer = tf.keras.optimizers.SGD(clipvalue=1.0)
        self.model.compile(loss="mse", optimizer=optimizer)
        self.model.fit(embeddings[0], labels[0],
                       validation_data=(embeddings[1], labels[1]))
        return self.model.evaluate(embeddings[2], labels[2])
