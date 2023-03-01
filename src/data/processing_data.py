import os
import sys
from numpy import ndarray, array
from pandas import DataFrame, read_csv
from typing import Tuple


os.chdir(os.getcwd())

class StackedFormat:
    def __init__(self,
                 name: str,
                 split: str,
                 T_: int) -> None:
        self.df = read_csv(f"./inputs_data/data_{name}_{split}.csv",
                            encoding="utf-8",
                            sep="|")
        self.T = T_

    def get_context_nUtterances(self) -> DataFrame:
        df_label_freq = (self.df[["Dialogue_ID", "Label"]]
                         .groupby("Dialogue_ID")
                         .count()
                         .reset_index()
                         .sort_values(by="Label")
                         .reset_index(drop=True))
        return self.df[self.df["Dialogue_ID"].isin(
                df_label_freq[df_label_freq["Label"] == self.T]["Dialogue_ID"].to_list())]

    def get_contexts_labels(self) -> Tuple[list, ndarray]:
        return ((self.get_context_nUtterances()
                 .groupby("Dialogue_ID")["Utterance"]
                 .apply(list)
                 .to_frame()
                 .reset_index()
                 .apply(lambda x: " ".join(x.Utterance), axis=1)
                 .to_list()), array(self.get_context_nUtterances()
                                    .groupby("Dialogue_ID")["Label"]
                                    .apply(list)
                                    .to_list()))
