import sys
import os
from typing import List, Union
import tensorflow as tf
from numpy import vstack, array
from torch import no_grad, Tensor
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
            "prajjwal1/bert": [BertModel, BertTokenizer],
            "xlnet": [XLNetModel, XLNetTokenizer]
        }


class TransformersEncoder:
    def __init__(
            self,
            model_name: str,
            T: int) -> None:
        _transformer = model_name.split('-')[0]
        self.model = _dict[_transformer][0].from_pretrained(model_name)
        self.tokenizer = _dict[_transformer][1].from_pretrained(
                                                model_name)
        self.T = T

    def get_embeddings(
                self,
                list_inputs: Union[List[str],
                                   List[List[str]],
                                   List[tf.Tensor]]) -> tf.Tensor:
        """Return the embeddings from the initialized transform

        Args:
            list_inputs (
                Union[List[str],
                      List[List[str]],
                      List[tf.Tensor]]): Texts inputs

        Returns:
            tf.Tensor: embeddings
        """
        def _embedding(item_inputs:  Union[str, List[str]]):
            if type(item_inputs) in (str, list):
                encoded_input = self.tokenizer(
                                        item_inputs,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt')
                with no_grad():
                    model_output = self.model(**encoded_input)
                return normalize(
                        mean_pooling(model_output,
                                     encoded_input['attention_mask']),
                        p=2, dim=1)
            else:
                with no_grad():
                    model_output = self.model(
                                Tensor(array(item_inputs)).long())
                return model_output["last_hidden_state"]

        _embeddings = list([])
        for item in list_inputs:
            _embeddings.append(_embedding(item))
        _embeddings = tf.convert_to_tensor(vstack(_embeddings))
        return (_embeddings if type(list_inputs[0]) not in (list,)
                else _reshape_(_embeddings, self.T))


class HierarchicalTransformersEncoder(TransformersEncoder):
    def __init__(
            self,
            model_name: str,
            T: int) -> None:
        super().__init__(model_name, T)

    def hierarchical_embeddings(
                self,
                list_texts: List[List[str]]) -> tf.Tensor:
        _embeddings = self.get_embeddings(list_texts)
        return self.get_embeddings(_embeddings)
