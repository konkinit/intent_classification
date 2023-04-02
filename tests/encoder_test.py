import os
import sys
import pytest
import tensorflow as tf
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.models import TransformersEncoder


@pytest.mark.parametrize(
    "text, embedding_shape",
    [(["a test for encoder"], tf.TensorShape([1, 768]))])
def test_encoder(text, embedding_shape):
    assert (TransformersEncoder('bert-base-uncased', 5)
            .get_embeddings(text)
            .shape) == embedding_shape
