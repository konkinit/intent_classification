import os
from numpy import array, hstack
from pandas import DataFrame, read_csv
from typing import Tuple, List
from torch import Tensor
import tensorflow as tf


os.chdir(os.getcwd())
_split_ = ["train", "validation", "test"]


class Format:
    def __init__(self,
                 dataset_name: str,
                 T: int,
                 type_format: str) -> None:
        self.df = list([read_csv(f"./inputs_data/data_{dataset_name}_{split}.csv",
                                    encoding="utf-8",
                                    sep="|") for split in _split_])
        self.T = T
        self.type_format = type_format

    def get_dialogue_acts(self) -> list:
        """
        Return the distinct labels accross the 3 splits of dataset
        """
        list_labels = set()
        for df_ in self.df:
            list_labels = list_labels | set(df_["Label"].unique())
        return list(list_labels)

    def get_context_nUtterances(self) -> DataFrame:
        def context_min_nUtterances_split_level(df: DataFrame):
            df_label_freq = (df[["Dialogue_ID", "Label"]]
                             .groupby("Dialogue_ID")
                             .count()
                             .reset_index()
                             .sort_values(by="Label")
                             .reset_index(drop=True))
            return (df[df["Dialogue_ID"]
                       .isin(df_label_freq[df_label_freq["Label"] >= self.T]["Dialogue_ID"].to_list())])

        def context_nUtterances_split_level(df: DataFrame) -> DataFrame:
            grouped_index = (context_min_nUtterances_split_level(df)
                             .groupby('Dialogue_ID', as_index = False)
                             .apply(lambda x: x.reset_index(drop = True))
                             .reset_index())
            return (grouped_index[grouped_index.level_1 <= self.T-1][["Utterance", "Dialogue_ID", "Label"]]
                    .reset_index(drop=True))
        
        return list([context_nUtterances_split_level(self.df[i]) for i in range(len(self.df))])

    def get_contexts_labels(self) -> Tuple[int, List[List[str]], List[tf.Tensor]]:
        """
        Return the contexts and dialog act in a stacked format
        """
        def contexts_labels_split_level(df_: DataFrame) -> Tuple[List[str], Tensor]:
            df = df_.copy()
            set_dialogue_act = self.get_dialogue_acts()
            df["LabelVector"] = df["Label"].apply(lambda x: array([int(x == label) for label in set_dialogue_act]))
            if self.type_format == "stacked":
                return ((df.groupby("Dialogue_ID")["Utterance"]
                        .apply(list)
                        .to_frame()
                        .reset_index()
                        .apply(lambda x: " ".join(x.Utterance), axis=1)
                        .to_list()), tf.constant(array(df.groupby("Dialogue_ID")["LabelVector"]
                                                       .apply(lambda x: list(hstack(x)))
                                                       .to_list(), dtype="int32")))
            else:
                return ((df["Utterance"].to_list()), tf.constant(array(df["LabelVector"].to_list(), dtype="int32")))

        contexts, labels = list([]), list([])
        for df in self.get_context_nUtterances():
            contexts.append(contexts_labels_split_level(df)[0])
            labels.append(contexts_labels_split_level(df)[1])
            len_dialogue_act = len(self.get_dialogue_acts())
        return len_dialogue_act, contexts, labels
