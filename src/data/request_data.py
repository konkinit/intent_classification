import requests as req
import pandas as pd
from json import loads

def getData(_url_: str):
    dialogs = req.get(url=_url_).json()
    return pd.DataFrame.from_dict(
                [dialogs["rows"][i]["row"] 
                    for i in range(len(dialogs["rows"]))]
            )[["Utterance", "Dialogue_ID", "Label"]]
