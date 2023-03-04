import os
import sys
import torch
from pandas import DataFrame
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.data.processing_data import StackedFormat
from src.models.encoders import BERTenocder
from src.models.decoders import MLP


class Pipeline:
    def __init__(self,
                 dataset_name: str,
                 T: int,
                 encoder_name: str,
                 decoder_name: str) -> None:
        self.dataset_name = dataset_name
        self.T = T
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.performance = 0

    def summary_exec(self) -> DataFrame:
        """
        Execute the encode-decode strategy on a dataset
        and Summarize the report in a dataframe
        """
        (contexts,
         labels) = StackedFormat(self.dataset_name, self.T).get_contexts_labels()
        embeddings = list([BERTenocder(self.encoder_name).embedding(contexts[i])
                           for i in range(len(contexts))])
        score_metrics = (MLP(embeddings[0].shape[2], 5, 0.01)
                         .evaluation(embeddings, labels))
        df_summary = DataFrame(columns=["dataset_name", ])
        pass

