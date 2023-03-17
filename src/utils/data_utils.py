from pandas import DataFrame
from typing import Tuple, List
from numpy import array, hstack
from torch import Tensor
import tensorflow as tf


def _reshape_(x: tf.Tensor, T: int) -> tf.Tensor:
    """
    Reshape a tensor of dimension 2 to a 3D tensor

    Args:
        x (tf.Tensor): a tensor
        T (int): maximal num:ber of utterance per dialogue

    Returns:
        tf.Tensor: _description_
    """
    return tf.reshape(x, [int(x.shape[0]/T), T, x.shape[1]])


def get_file_path(ds_ame: str, split: str) -> str:
    """
    Return the path of a dataset

    Args:
        ds_ame (str): dataset name
        split (str): split (train, validation, test)

    Returns:
        str: absolute path of the dataset
    """
    return f"./inputs_data/data_{ds_ame}_{split}.csv"


def context_min_nUtterances_split_level(df: DataFrame, T: int) -> DataFrame:
    """
    Return the dataset with its dialogues composed of
    at least T utterances

    Args:
        df (DataFrame): dataset
        T (int): maximal num:ber of utterance per dialogue

    Returns:
        DataFrame: dataset with dialogues of min T utterances
    """
    df_label_freq = (df[["Dialogue_ID", "Label"]]
                     .groupby("Dialogue_ID")
                     .count()
                     .reset_index()
                     .sort_values(by="Label")
                     .reset_index(drop=True))
    return (df[df["Dialogue_ID"]
            .isin(df_label_freq[df_label_freq["Label"] >= T]["Dialogue_ID"]
            .to_list())])


def context_nUtterances_split_level(df: DataFrame, T: int) -> DataFrame:
    """
    Return the dataset at split level where each dialogue is composed of
    its first T utterances

    Args:
        df (DataFrame): dataset at split level
        T (int): maximal num:ber of utterance per dialogue

    Returns:
        DataFrame: dataset with dialogues and their first T utterances
    """
    grouped_index = (context_min_nUtterances_split_level(df, T)
                     .groupby('Dialogue_ID', as_index=False)
                     .apply(lambda x: x.reset_index(drop=True))
                     .reset_index())
    return (grouped_index[grouped_index.level_1 <= T-1][["Utterance",
                                                         "Dialogue_ID",
                                                         "Label"]]
            .reset_index(drop=True))


def contexts_labels_split_level(
        df_: DataFrame,
        set_labels: set,
        type_format: str) -> Tuple[List[str], Tensor]:
    """
    Retrieve the first T utterances of dialogues with at least T
    utterances and preprocess it to a given format which can be cconca
    tenate or separate

    Args:
        df_ (DataFrame): dataset
        set_labels (set): set of classes in the dataset
        type_format (str): type of formatting data

    Returns:
        Tuple[List[str], Tensor]: a tuple where the 1st arg is the
        textual data and the 2nd the one-hot encoded labels
    """
    assert type_format in ("concatenate", "separate"), "Incorrec format type"
    df = df_.copy()
    def f_(x, label): return int(x == label)
    if type_format == "concatenate":
        df["LabelVector"] = df["Label"].apply(
            lambda x: array([int(x == label) for label in set_labels]))
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
        df["LabelVector"] = df["Label"].apply(lambda x: array([
            f_(x, label) for label in set_labels]))
        return (df.groupby("Dialogue_ID")["Utterance"]
                  .apply(list)
                  .to_list()), tf.constant(
                            array(df.groupby("Dialogue_ID")["LabelVector"]
                                    .apply(list)
                                    .to_list(), dtype="int32"))
