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
                    for i in range(len(dialogs["rows"]))]
            )[["Utterance", "Dialogue_ID", "Label"]]


def getDataTF(dataset: str, split_: str):
    (tfds.as_dataframe(tfds.load(f'huggingface:silicone/{dataset}', split=split_))[["Utterance", "Dialogue_ID", "Label"]]
            .apply(lambda x: x.apply(lambda z: z.decode("utf-8") if type(z) == bytes else z), axis=1)
            .sort_values("Dialogue_ID").reset_index(drop=True)
            .to_csv(f"./inputs_data/data_{dataset}_{split_}.csv",
                    index=False,
                    sep="|",
                    encoding='utf-8'
                    )
    )
