import os
import sys
from pandas import DataFrame
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.data.processing_data import StackedFormat
from src.models.encoders import BERTencoder
from src.models.decoders import MLP


class Pipeline:
    def __init__(self,
                 dataset_name: str,
                 T: int,
                 encoder_name: str,
                 n_layers: int,
                 decoder_name: str) -> None:
        self.dataset_name = dataset_name
        self.T = T
        self.encoder_name = encoder_name
        self.nLayers = n_layers
        self.decoder_name = decoder_name
        self.performance = 0

    def summary_exec(self) -> DataFrame:
        """
        Execute the encode-decode strategy on a dataset
        and Summarize the report in a dataframe
        """
        (contexts,
         labels) = StackedFormat(self.dataset_name, self.T).get_contexts_labels()
        embeddings = list([BERTencoder(self.encoder_name).embedding(contexts[i])
                           for i in range(len(contexts))])
        self.performance = (MLP(embeddings[0].shape[2], self.nLayers, self.T, 0.01)
                            .evaluation(embeddings, labels))
        df_summary = DataFrame(data=[[self.dataset_name, self.encoder_name, self.decoder_name, self.performance]],
                               columns=["dataset_name", "encoder_model", "decoder_model", "performance"],
                               index=0)
        return df_summary
