from typing import List
from torch import no_grad, Tensor, sum, clamp
from torch.nn.functional import normalize
from transformers import BertTokenizer, BertModel


class BERTenocder:
    def __init__(self,
                 model_name: str,
                 max_length: int) -> None:
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.maxLength = max_length

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
        return (sum(token_embeddings * input_mask_expanded, 1)
                / clamp(input_mask_expanded.sum(1), min=1e-9))

    def embedding(self, texts: List[str]) -> Tensor:
        tokenizer = BertTokenizer.from_pretrained(self.model)
        model = BertModel.from_pretrained(self.model)
        encoded_input = tokenizer(texts,
                                  padding=True,
                                  truncation=True,
                                  return_tensors='pt')
        with no_grad():
            model_output = model(**encoded_input)
        return normalize(self.mean_pooling(model_output,
                                           encoded_input['attention_mask']),
                         p=2,
                         dim=1)
