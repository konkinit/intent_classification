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
from src.utils.models_utils import (
    mean_pooling)
from src.utils.data_utils import (
    _reshape_)


_dict = {
            "bert": [BertModel, BertTokenizer],
            "xlnet": [XLNetModel, XLNetTokenizer]
        }


class TransformersEncoder:
    def __init__(self,
                 model_name: str,
                 format_type: str,
                 T: int) -> None:
        _transformer = model_name.split('-')[0]
        self.model = _dict[_transformer][0].from_pretrained(model_name)
        self.tokenizer = _dict[_transformer][1].from_pretrained(
                                                model_name,
                                                model_max_length=512)
        self.T = T
        self.format_type = format_type

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
        _embeddings = tf.convert_to_tensor(vstack(texts_embedded))
        if self.format_type == "stacked":
            return _embeddings
        return _reshape_(_embeddings, self.T)


class HierarchicalTransformersEncoder:
    def __init__(self) -> None:
        pass
