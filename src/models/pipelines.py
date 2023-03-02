import os
import sys
import torch
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.data.processing_data import StackedFormat
from src.models.encoders import BERTenocder
from src.models.decoders import MLP


class Pipeline:
    def __init__(self,
                 name: str,
                 split: str,
                 T_: int,
                 model_name: str) -> None:
        self.data_object = StackedFormat(name, split, T_)
        self.encoder_object = BERTenocder(model_name)
        self.embeddings = torch.Tensor([])
        self.decoder_object = MLP(768, 5, 0.01)

    def exec(self):
        contexts, labels = self.data_object.get_contexts_labels()
        self.embeddings = self.encoder_object.embedding(contexts)
        labelsHat = self.decoder_object.prediction(self.embeddings, labels)
