import os
import sys
import tensorflow as tf
from pandas import DataFrame, read_csv
from typing import Tuple, List
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utils.data_utils import (
    get_file_path,
    context_nUtterances_split_level,
    contexts_labels_split_level)


_split_ = ["train", "validation", "test"]


class Format:
    def __init__(
            self,
            dataset_name: str,
            T: int,
            type_format: str) -> None:
        self.df = list([read_csv(get_file_path(dataset_name, split),
                                 encoding="utf-8",
                                 sep="|") for split in _split_])
        self.T = T
        self.type_format = type_format
        self.Labels = list()

    def get_distincts_labels(self) -> list:
        """
        Return the distinct labels accross the 3 splits of dataset
        """
        set_of_labels = set()
        for df_ in self.df:
            set_of_labels = set_of_labels | set(df_["Label"].unique())
        self.Labels = list(set_of_labels)
        return self.Labels

    def get_context_nUtterances(self) -> DataFrame:
        """
        Return the dataframe of dialogues truncate to
        T utterances
        """
        return list([context_nUtterances_split_level(self.df[i], self.T)
                     for i in range(len(self.df))])

    def get_contexts_labels(
            self) -> Tuple[int, List[List[str]], List[tf.Tensor]]:
        """
        Return the contexts and labels in a stacked format
        """
        contexts, labels = list([]), list([])
        set_of_labels = self.get_distincts_labels()
        for df in self.get_context_nUtterances():
            contexts.append(contexts_labels_split_level(
                df, set_of_labels, self.type_format)[0])
            labels.append(contexts_labels_split_level(
                df, set_of_labels, self.type_format)[1])
            len_label_set = len(set_of_labels)
        return len_label_set, contexts, labels
