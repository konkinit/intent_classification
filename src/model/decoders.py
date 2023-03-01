from numpy import nan
import tensorflow as tf


class MLP:
    def __init__(self, 
                 n_units: int,
                 n_layers: int,
                 f_dropout: float) -> None:
        self.model = tf.keras.models.Sequential()
        self.nUnints = n_units
        self.nLayers = n_layers
        self.fDroupout = f_dropout

    def __build__(self):
        for _ in range(self.nLayers):
            self.model.add(tf.keras.layers.Dense(
                                        self.nUnints,
                                        activation="relu"))
        self.model.add(tf.keras.layers.Dropout(self.fDropout))

    def __compile__(self):
        optimizer = tf.keras.optimizers.SGD(clipvalue=1.0)
        self.model.compile(loss="mse", optimizer=optimizer)