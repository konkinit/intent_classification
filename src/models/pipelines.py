import os
import sys
from pandas import DataFrame
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.data.processing_data import Format
from src.models.encoders import TransformersEncoder
from src.models.decoders import MLP, SequentialGRU


_dict_decoder = {
                    "MLP": MLP,
                    "GRU": SequentialGRU
                }


class Pipeline:
    def __init__(self,
                 dataset_name: str,
                 data_format_type: str,
                 T: int,
                 encoder_name: str,
                 decoder_name: str,
                 n_layers: int,
                 f_dropout: float) -> None:
        self.dataset_name = dataset_name
        self.data_format_type = data_format_type
        self.T = T
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.nLayers = n_layers
        self.fDropout = f_dropout
        self.performance = 1.0

    def summary_exec(self) -> DataFrame:
        """
        Execute the encode-decode strategy on a dataset
        and Summarize the report in a dataframe
        """
        dimLabelSet, contexts, labels = (Format(self.dataset_name,
                                                self.T,
                                                self.data_format_type)
                                         .get_contexts_labels())
        embeddings = list([(TransformersEncoder(self.encoder_name)
                            .batch_embedding(contexts[i]))
                           for i in range(len(contexts))])
        self.performance = (_dict_decoder[self.decoder_name](
                                embeddings[0].shape[1],
                                self.nLayers,
                                [dimLabelSet, self.T],
                                self.fDropout).evaluate(
                                    embeddings, labels))
        df_report = DataFrame(data=[[self.dataset_name,
                                     self.encoder_name,
                                     self.decoder_name,
                                     self.performance]],
                              columns=["dataset_name",
                                       "encoder_model",
                                       "decoder_model",
                                       "performance"])
        return df_report
