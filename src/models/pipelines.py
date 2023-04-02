import os
import sys
from pandas import DataFrame
from numpy import array
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
                 T: int,
                 encoder_name: str,
                 decoder_name: str,
                 data_format_type: str,
                 *args) -> None:
        self.dataset_name = dataset_name
        self.T = T
        self.encoder_name = encoder_name
        self.data_format_type = data_format_type
        self.decoder_name = decoder_name
        self.args = args if len(args) > 0 else [3, 0.2]
        self.df_report = DataFrame()
        self.confusion_matrix = array([])

    def summarize(self) -> None:
        """Execute the encode-decode strategy on a dataset
        and Summarize the report in a dataframe
        """
        format_obj = Format(self.dataset_name,
                            self.T,
                            self.data_format_type)
        dimLabelSet, contexts, labels = format_obj.get_contexts_labels()
        set_labels = format_obj.get_distincts_labels()
        embeddings = list([(TransformersEncoder(self.encoder_name,
                                                self.T)
                            .get_embeddings(contexts[i]))
                           for i in range(len(contexts))])
        if self.data_format_type == "concatenate":
            embeddingsDim = embeddings[0].shape[1]
        else:
            embeddingsDim = embeddings[0].shape[2]
        performance, self.confusion_matrix = _dict_decoder[self.decoder_name](
                                embeddingsDim,
                                [dimLabelSet, self.T],
                                self.args[0],
                                self.args[1]
                                )._inference(
                                    embeddings, labels, set_labels)
        self.df_report = DataFrame(data=[[self.dataset_name,
                                         self.encoder_name,
                                         self.decoder_name,
                                         performance]],
                                   columns=["dataset_name",
                                            "encoder_model",
                                            "decoder_model",
                                            "performance"])
