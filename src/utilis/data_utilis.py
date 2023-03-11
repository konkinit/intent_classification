from pandas import DataFrame
from typing import Tuple, List
from numpy import array, hstack
from torch import Tensor
import tensorflow as tf


def get_file_path(ds_ame, split):
    return f"./inputs_data/data_{ds_ame}_{split}.csv"


def context_min_nUtterances_split_level(df: DataFrame, T: int):
    df_label_freq = (df[["Dialogue_ID", "Label"]]
                        .groupby("Dialogue_ID")
                        .count()
                        .reset_index()
                        .sort_values(by="Label")
                        .reset_index(drop=True))
    return (df[df["Dialogue_ID"]
                .isin(df_label_freq[df_label_freq["Label"] >= T]["Dialogue_ID"].to_list())])


def context_nUtterances_split_level(df: DataFrame, T: int) -> DataFrame:
    grouped_index = (context_min_nUtterances_split_level(df, T)
                        .groupby('Dialogue_ID', as_index=False)
                        .apply(lambda x: x.reset_index(drop=True))
                        .reset_index())
    return (grouped_index[grouped_index.level_1 <= T-1][["Utterance", "Dialogue_ID", "Label"]]
            .reset_index(drop=True))


def contexts_labels_split_level(
        df_: DataFrame,
        set_labels: list, 
        type_format: str) -> Tuple[List[str], Tensor]:
    df = df_.copy()
    if type_format == "stacked":
        df["LabelVector"] = df["Label"].apply(lambda x: array([int(x == label) for label in set_labels]))
        return ((df.groupby("Dialogue_ID")["Utterance"]
                   .apply(list)
                   .to_frame()
                   .reset_index()
                   .apply(lambda x: " ".join(x.Utterance), axis=1)
                   .to_list()), tf.constant(
                        array(df.groupby("Dialogue_ID")["LabelVector"]
                                .apply(lambda x: list(hstack(x)))
                                .to_list(), dtype="int32")))
    else:
        def f_(x, label): return int(x == label)
        df["LabelVector"] = df["Label"].apply(lambda x: array([
            f_(x, label) for label in set_labels]))
        return df["Utterance"].to_list(), tf.constant(
                        array(df["LabelVector"].to_list(), dtype="int32"))
