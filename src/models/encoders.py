import sys
import os
from typing import List
import tensorflow as tf
from numpy import vstack
from tqdm import tqdm
from torch import no_grad
from torch.nn.functional import normalize
from transformers import (
        BertModel,
        BertTokenizer,
        XLNetModel,
        XLNetTokenizer)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utilis.models_utilis import (
    mean_pooling)

_dict = {
            "bert": [BertModel, BertTokenizer],
            "xlnet": [XLNetModel, XLNetTokenizer]
        }


class TransformersEncoder:
    def __init__(self,
                 model_name: str) -> None:
        _transformer = model_name.split('-')[0]
        self.model = _dict[_transformer][0].from_pretrained(model_name)
        self.tokenizer = _dict[_transformer][1].from_pretrained(model_name)

    def batch_embedding(self, list_texts: List[str]) -> tf.Tensor:
        def item_embedding(texts):
            encoded_input = self.tokenizer(texts,
                                           padding=True,
                                           truncation=True,
                                           return_tensors='pt')
            with no_grad():
                model_output = self.model(**encoded_input)
            return normalize(mean_pooling(model_output,
                                          encoded_input['attention_mask']),
                             p=2, dim=1)
        texts_embedded = list([])
        for texts in tqdm(list_texts):
            texts_embedded.append(item_embedding(texts))
        return tf.convert_to_tensor(vstack(texts_embedded))


class HierarchicalTransformersEncoder:
    def __init__(self) -> None:
        pass
