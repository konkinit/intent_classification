import os
from numpy import array, hstack
from pandas import DataFrame, read_csv
from typing import Tuple, List
from torch import Tensor
import tensorflow as tf


os.chdir(os.getcwd())
_split_ = ["train", "validation", "test"]


class StackedFormat:
    def __init__(self,
                 dataset_name: str,
                 T: int) -> None:
        self.df = list([read_csv(f"./inputs_data/data_{dataset_name}_{split}.csv",
                                 encoding="utf-8",
                                 sep="|") for split in _split_])
        self.T = T

    def get_context_nUtterances(self) -> DataFrame:
        def context_nUtterances_split_level(df: DataFrame):
            df_label_freq = (df[["Dialogue_ID", "Label"]]
                             .groupby("Dialogue_ID")
                             .count()
                             .reset_index()
                             .sort_values(by="Label")
                             .reset_index(drop=True))
            return (df[df["Dialogue_ID"]
                       .isin(df_label_freq[df_label_freq["Label"] == self.T]["Dialogue_ID"].to_list())])
        return list([context_nUtterances_split_level(self.df[i]) for i in range(len(self.df))])

    def get_contexts_labels(self) -> Tuple[int, List[List[str]], List[tf.Tensor]]:
        """
        Return the contexts and dialog act in a stacked format
        """
        def contexts_labels_split_level(df_: DataFrame) -> Tuple[List[str], Tensor]:
            df = df_.copy()
            dim_act = df.Label.unique()
            df["LabelVector"] = df["Label"].apply(lambda x: array([int(x == _) for _ in dim_act]))
            return ((df.groupby("Dialogue_ID")["Utterance"]
                     .apply(list)
                     .to_frame()
                     .reset_index()
                     .apply(lambda x: " ".join(x.Utterance), axis=1)
                     .to_list()), tf.constant(array(df.groupby("Dialogue_ID")["LabelVector"]
                                                    .apply(lambda x: list(hstack(x)))
                                                    .to_list(), dtype="int32")))
        contexts, labels = list([]), list([])
        for df in self.get_context_nUtterances():
            contexts.append(contexts_labels_split_level(df)[0])
            labels.append(contexts_labels_split_level(df)[1])
            dim_act = len(df["Label"].unique())
        return dim_act, contexts, labels
