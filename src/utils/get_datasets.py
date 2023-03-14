import os
import sys
import requests as req
import pandas as pd
import tensorflow_datasets as tfds


if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())


def getDataHF(_url_: str) -> None:
    """
    Get the dataset using Hugging Face API

    Args:
        _url_ (str): dataset url

    Returns:
        _type_: None
    """
    dialogs = req.get(url=_url_).json()
    return pd.DataFrame.from_dict(
                [dialogs["rows"][i]["row"]
                    for i in range(len(dialogs["rows"]))])[["Idx",
                                                            "Utterance",
                                                            "Dialogue_ID",
                                                            "Label"]]


def getDataTF(dataset: str, split_: str, label_name: str) -> None:
    """
    Get the official split part of a dataset in csv format
    with a given label variable name

    Args:
        dataset (str): dataset name official name
        split_ (str): split
        label_name (str): Label variable which is Dialogue_Act,
                          Sentiment, Emotion
    """
    data_str = f'huggingface:silicone/{dataset}'
    if os.path.exists(
        os.path.join(os.getcwd(),
                     f"./inputs_data/data_{dataset}_{split}.csv")):
        pass
    else:
        (tfds.as_dataframe(tfds.load(data_str, split=split_))[["Idx",
                                                               "Utterance",
                                                               "Dialogue_ID",
                                                               label_name]]
         .apply(
            lambda x: x.apply(
                lambda z: z.decode("utf-8") if type(z) == bytes else z),
            axis=1)
         .sort_values(["Dialogue_ID", "Idx"])
         .reset_index(drop=True)
         .drop(axis=1, columns="Idx")
         .rename(columns={label_name: "Label"})
         .to_csv(f"./inputs_data/data_{dataset}_{split_}.csv",
                 index=False,
                 sep="|",
                 encoding='utf-8'))


_split_ = ["train", "validation", "test"]

_datasets_da_ = ["swda", "dyda_da", "mrda"]
_datasets_e_ = ["meld_e", "dyda_e", "iemocap"]
_datasets_s_ = ["meld_s", "sem"]

_dict_datasets = {
    "Dialogue_Act": _datasets_da_,
    "Emotion": _datasets_e_,
    "Sentiment": _datasets_s_
}


for _label in _dict_datasets.keys():
    for dataset in _dict_datasets[_label]:
        for split in _split_:
            getDataTF(dataset, split, _label)
