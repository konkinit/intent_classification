from typing import List
import torch
from transformers import BertTokenizer, BertModel


class BERTenocder:
    def __init__(self,
                 model_name: str,
                 texts: str) -> None:
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.inputs = texts

    def embedding(self) -> torch.Tensor:
        inputs_ids = torch.tensor([self.tokenizer.encode(
                                                    self.inputs,
                                                    add_special_tokens=True)])
        with torch.no_grad():
            last_hidden_states = self.model(inputs_ids)[0]
        return last_hidden_states.mean(axis=1)
