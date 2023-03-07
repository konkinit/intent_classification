import os
import sys
import requests as req
import pandas as pd
import tensorflow_datasets as tfds


if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())


def getDataHF(_url_: str):
    dialogs = req.get(url=_url_).json()
    return pd.DataFrame.from_dict(
                [dialogs["rows"][i]["row"]
                    for i in range(len(dialogs["rows"]))])[["Idx", "Utterance", "Dialogue_ID", "Label"]]


def getDataTF(dataset: str, split_: str) -> None:
    data_str = f'huggingface:silicone/{dataset}'
    if os.path.exists(os.path.join(os.getcwd(), f"inputs_data/data_{dataset}_{split}.csv")):
        pass
    else:
        (tfds.as_dataframe(tfds.load(data_str, split=split_))[["Idx", "Utterance", "Dialogue_ID", "Label"]]
         .apply(lambda x: x.apply(lambda z: z.decode("utf-8") if type(z) == bytes else z), axis=1)
         .sort_values(["Dialogue_ID", "Idx"])
         .reset_index(drop=True)
         .drop(axis=1, columns="Idx")
         .to_csv(f"./inputs_data/data_{dataset}_{split_}.csv",
                 index=False,
                 sep="|",
                 encoding='utf-8'))


_split_ = ["train", "validation", "test"]
_datasets_da_ = ["swda", "dyda_da", "mrda"]

for dataset in _datasets_da_:
    for split in _split_:
        getDataTF(dataset, split)
