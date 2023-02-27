from numpy import nan
from tensorflow import keras 


class MLP(keras.Sequential):
    def __init__(self, 
                 n_units: int,
                 n_layers: int) -> None:
        super().__init__()
        self.nUnints = n_units
        self.nLayers = n_layers

    def __build__(self):
        for _ in range(self.nLayers):
            self.add(keras.layers.Dense(self.nUnints,
                                        activation="relu"))
        self.add(keras.layers.Dropout(0.2))
