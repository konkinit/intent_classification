from typing import List
import tensorflow as tf
from torch import no_grad, sum, clamp
from torch.nn.functional import normalize
from transformers import BertModel, BertTokenizer


class BERTencoder:
    def __init__(self,
                 model_name: str) -> None:
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        """
        token_embeddings = model_output[0]
        input_mask_expanded = (attention_mask
                               .unsqueeze(-1)
                               .expand(token_embeddings.size())
                               .float())
        return sum(token_embeddings * input_mask_expanded, 1) / clamp(input_mask_expanded.sum(1), min=1e-9)

    def embedding(self, texts: List[str]) -> tf.Tensor:
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with no_grad():
            model_output = self.model(**encoded_input)
        texts_embedded = normalize(self.mean_pooling(model_output, encoded_input['attention_mask']), p=2, dim=1)
        return tf.convert_to_tensor(texts_embedded.numpy())
