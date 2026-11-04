"""CPU unit checks against the pinned upstream implementation, without weights."""
import torch
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5GatedDeltaNet
from nanoinspect.adapters.qwen35 import instrument


def tiny_config():
    config = Qwen3_5TextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"], num_attention_heads=4,
        num_key_value_heads=2, head_dim=4, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=4, linear_value_head_dim=4, linear_conv_kernel_dim=3)
    config._attn_implementation = "eager"
    return config


def test_attention_prefill_reference():
    torch.manual_seed(2)
    layer = Qwen3_5Attention(tiny_config(), 1).eval()
    x = torch.randn(1, 3, 16)
    cos, sin = torch.ones(1, 3, 4), torch.zeros(1, 3, 4)
    mask = torch.full((1, 1, 3, 3), float("-inf")).triu(1)
    with torch.no_grad():
        expected = layer(x, (cos, sin), mask)[0]
        names = []
        with instrument(layer, "attention", lambda name, value: names.append(name)):
            observed = layer(x, (cos, sin), mask)[0]
    torch.testing.assert_close(expected, observed, rtol=0, atol=0)
    assert {"query_rotary", "key_grouped", "probabilities", "projection"}.issubset(names)


def test_delta_prefill_reference():
    torch.manual_seed(2)
    layer = Qwen3_5GatedDeltaNet(tiny_config(), 0).eval()
    x = torch.randn(1, 3, 16)
    with torch.no_grad():
        expected = layer(x)
        names = []
        with instrument(layer, "delta", lambda name, value: names.append(name)):
            observed = layer(x)
    torch.testing.assert_close(expected, observed, rtol=1e-4, atol=1e-5)
    assert {"post_convolution_qkv", "key_normalized", "state_after_token_2", "projection"}.issubset(names)
