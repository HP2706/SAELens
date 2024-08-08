from typing import Any, Callable

import pytest
import torch
from nnsight import NNsight
from parsing import get_nested_attr
from transformers import LlamaConfig, LlamaForCausalLM


def compare_values(
    value1: Any, value2: Any, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    if isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
        return torch.allclose(value1, value2, rtol=rtol, atol=atol)
    elif isinstance(value1, (list, tuple)) and isinstance(value2, (list, tuple)):
        if len(value1) != len(value2):
            return False
        return all(compare_values(v1, v2, rtol, atol) for v1, v2 in zip(value1, value2))
    elif isinstance(value1, dict) and isinstance(value2, dict):
        if value1.keys() != value2.keys():
            return False
        return all(
            compare_values(value1[k], value2[k], rtol, atol) for k in value1.keys()
        )
    else:
        return value1 == value2


def compare_values(
    value1: Any, value2: Any, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    if isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
        return torch.allclose(value1, value2, rtol=rtol, atol=atol)
    elif isinstance(value1, (list, tuple)) and isinstance(value2, (list, tuple)):
        if len(value1) != len(value2):
            return False
        return all(compare_values(v1, v2, rtol, atol) for v1, v2 in zip(value1, value2))
    elif isinstance(value1, dict) and isinstance(value2, dict):
        if value1.keys() != value2.keys():
            return False
        return all(
            compare_values(value1[k], value2[k], rtol, atol) for k in value1.keys()
        )
    else:
        return value1 == value2


small_llama_config = LlamaConfig(
    vocab_size=1000,
    hidden_size=8,
    intermediate_size=16,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=4,
    hidden_act="silu",
    max_position_embeddings=512,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    use_cache=True,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    pretraining_tp=1,
    tie_word_embeddings=False,
    rope_theta=10000.0,
    rope_scaling=None,
    attention_bias=False,
    attention_dropout=0.1,
    mlp_bias=False,
)


model = NNsight(LlamaForCausalLM(small_llama_config))


# Test the updated functions
test_cases = [
    (
        "model.layers[0].output[0]",
        lambda model: model.model.layers[0].output[0].save(),
    ),
    (
        "model.layers[0].output[0][0:5]",
        lambda model: model.model.layers[0].output[0][0:5].save(),
    ),
    (
        "model.layers[0].output[0][1:10:2]",
        lambda model: model.model.layers[0].output[0][1:10:2].save(),
    ),
    (
        "model.layers[0].output[0][:, 0]",
        lambda model: model.model.layers[0].output[0][:, 0].save(),
    ),
    (
        "model.layers[0].output[0][..., 0]",
        lambda model: model.model.layers[0].output[0][..., 0].save(),
    ),
    (
        "model.layers[0].output[0][1]",
        lambda model: model.model.layers[0].output[0][1].save(),
    ),
    (
        "model.layers[0].output[0][0, :, :]",
        lambda model: model.model.layers[0].output[0][0, :, :].save(),
    ),
]


def test(model: NNsight, input: Any, attr_str: str, fn: Callable[[NNsight], Any]):
    with model.trace(input):
        saved_value1 = fn(model)
        saved_value2 = get_nested_attr(model, attr_str).save()

    assert compare_values(saved_value1.value, saved_value2.value), (
        f"Values are not equal",
        f"got shapes {saved_value1.value.shape} and {saved_value2.value.shape}",
        f"got values {saved_value1.value} and {saved_value2.value}",
    )


@pytest.mark.parametrize("attr_str, fn", test_cases)
def test_parsing(attr_str: str, fn: Callable[[NNsight], Any]):
    test(model, input, attr_str, fn)
    test(model, input, attr_str, fn)
    test(model, input, attr_str, fn)
    test(model, input, attr_str, fn)
    test(model, input, attr_str, fn)
    test(model, input, attr_str, fn)
