import os
import sys
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.data.processing_data import StackedFormat
from src.models.encoders import BERTenocder


class Pipeline:
    def __init__(self,
                 name: str,
                 split: str,
                 T_: int,
                 model_name: str) -> None:
        self.data_object = StackedFormat(name, split, 5)
        self.encoder_object = BERTenocder(model_name)
        #self.embeddings = 
        #self.decoder_object = 
