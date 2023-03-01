import os
import sys
import pytest
from torch import Size
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.model.encoders import BERTenocder


@pytest.mark.parametrize("text, embedding_shape",
                         [("a test for encoder", Size([1, 768]))])
def test_encoder(text, embedding_shape):
    assert BERTenocder('bert-base-uncased', text).embedding().shape == embedding_shape
