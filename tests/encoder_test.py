import os
import sys
import pytest
import tensorflow as tf
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.models.encoders import BERTencoder


@pytest.mark.parametrize("text, embedding_shape",
                         [(["a test for encoder"], tf.TensorShape([1, 768]))])
def test_encoder(text, embedding_shape):
    assert BERTencoder('bert-base-uncased').batch_embedding(text).shape == embedding_shape
