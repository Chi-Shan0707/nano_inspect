"""CPU unit checks against the pinned upstream implementation, without weights."""
import pytest
import torch
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as upstream
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5GatedDeltaNet, Qwen3_5ForCausalLM
from nanoinspect.adapters.qwen35 import instrument, explicit_recurrence
from nanoinspect.trace import AXES, ANCHORS, Recorder, TOLERANCES, check, source_lines


def tiny_config(layers=("linear_attention", "full_attention")):
    config = Qwen3_5TextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=len(layers),
        layer_types=list(layers), num_attention_heads=4, vocab_size=64,
        num_key_value_heads=2, head_dim=4, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=4, linear_value_head_dim=4, linear_conv_kernel_dim=3)
    config._attn_implementation = "eager"
    return config


def noop(*args, **kwargs):
    pass


def test_attention_prefill_reference():
    torch.manual_seed(2)
    layer = Qwen3_5Attention(tiny_config(), 1).eval()
    x = torch.randn(1, 3, 16)
    cos, sin = torch.ones(1, 3, 4), torch.zeros(1, 3, 4)
    mask = torch.full((1, 1, 3, 3), float("-inf")).triu(1)
    with torch.no_grad():
        expected = layer(x, (cos, sin), mask)[0]
        names = []
        with instrument(layer, "attention", lambda name, value, inputs=(), token=None: names.append(name)):
            observed = layer(x, (cos, sin), mask)[0]
    torch.testing.assert_close(expected, observed, rtol=0, atol=0)
    assert {"query_rotary", "key_grouped", "probabilities", "projection"}.issubset(names)
    assert "forward" not in layer.__dict__


def test_delta_upstream_mode_is_exact_and_recurrent_mode_is_close():
    torch.manual_seed(2)
    layer = Qwen3_5GatedDeltaNet(tiny_config(), 0).eval()
    x = torch.randn(1, 5, 16)
    with torch.no_grad():
        expected = layer(x)
        names = []
        with instrument(layer, "delta", lambda name, value, inputs=(), token=None: names.append(name)):
            observed = layer(x)
        with instrument(layer, "delta", noop, algorithm="recurrent"):
            recurrent = layer(x)
    torch.testing.assert_close(expected, observed, rtol=0, atol=0)
    torch.testing.assert_close(expected, recurrent, rtol=1e-4, atol=1e-5)
    assert {"post_convolution_qkv", "key_normalized", "state_after", "kernel_readout", "readout", "projection"}.issubset(names)
    assert "forward" not in layer.__dict__


def test_explicit_recurrence_matches_upstream_recurrent_rule():
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 6, 2, 4), torch.randn(1, 6, 2, 4), torch.randn(1, 6, 2, 4)
    beta, g = torch.rand(1, 6, 2), -torch.rand(1, 6, 2)
    expected, expected_state = upstream.torch_recurrent_gated_delta_rule(
        q, k, v, g=g, beta=beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
    fq, fk, fv, fb, fg = [x.transpose(1, 2).float() for x in (q, k, v, beta, g)]
    fq, fk = upstream.l2norm(fq), upstream.l2norm(fk)
    out, state = explicit_recurrence(fq / 2.0, fk, fv, fg, fb, torch.zeros(1, 2, 4, 4))
    torch.testing.assert_close(out.transpose(1, 2), expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


def test_instrumentation_is_rejected_when_nested_and_restored_after_errors():
    layer = Qwen3_5GatedDeltaNet(tiny_config(), 0).eval()
    with pytest.raises(RuntimeError, match="stop"):
        with instrument(layer, "delta", noop):
            raise RuntimeError("stop")
    assert "forward" not in layer.__dict__
    with instrument(layer, "delta", noop):
        with pytest.raises(RuntimeError, match="already instrumented"):
            with instrument(layer, "delta", noop):
                pass


def _model():
    torch.manual_seed(3)
    return Qwen3_5ForCausalLM(tiny_config(("linear_attention", "linear_attention", "full_attention"))).eval()


@pytest.mark.parametrize("index,kind,attr", [(0, "delta", "linear_attn"), (2, "attention", "self_attn")])
def test_tiny_model_prefill_decode_and_padding_parity(index, kind, attr):
    model = _model()
    module = getattr(model.model.layers[index], attr)
    ids = torch.randint(0, 64, (1, 7))
    mask = torch.ones_like(ids)
    padded_ids = torch.cat([torch.zeros(1, 2, dtype=ids.dtype), ids], 1)
    padded_mask = torch.cat([torch.zeros(1, 2, dtype=mask.dtype), mask], 1)

    def run():
        prefill = model(input_ids=ids, attention_mask=mask, use_cache=True)
        step = model(input_ids=ids[:, :1], attention_mask=torch.cat([mask, mask[:, :1]], 1),
                     past_key_values=prefill.past_key_values, use_cache=True)
        padded = model(input_ids=padded_ids, attention_mask=padded_mask).logits
        return prefill.logits, step.logits, padded

    with torch.no_grad():
        expected = run()
        with instrument(module, kind, noop):
            observed = run()
    for a, b in zip(expected, observed):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_tiny_model_cross_kernel_within_fp32_tolerance():
    model = _model()
    ids = torch.randint(0, 64, (1, 9))
    with torch.no_grad():
        expected = model(input_ids=ids).logits
        with instrument(model.model.layers[1].linear_attn, "delta", noop, algorithm="recurrent", explain=False):
            observed = model(input_ids=ids).logits
    result = check("recurrent_substitution", "logits", "test", "logits_cross_kernel", torch.float32, observed, expected,
                   greedy=bool(torch.equal(observed.argmax(-1), expected.argmax(-1))))
    assert result["passed"], result


def test_recorder_axes_cover_every_operation_and_decode():
    model = _model()
    ids = torch.randint(0, 64, (1, 6))
    lines = source_lines()
    for index, kind, attr in [(0, "delta", "linear_attn"), (2, "attention", "self_attn")]:
        module = getattr(model.model.layers[index], attr)
        prefill_rec = Recorder(kind, module, index, 5, 1, "prefill", lines, "https://example/source")
        decode_rec = Recorder(kind, module, index, 0, 1, "decode", lines, "https://example/source", position_offset=6)
        target = {"rec": prefill_rec}
        with torch.no_grad(), instrument(module, kind, lambda *a, **k: target["rec"](*a, **k)):
            out = model(input_ids=ids, use_cache=True)
            target["rec"] = decode_rec
            model(input_ids=ids[:, :1], past_key_values=out.past_key_values, use_cache=True)
        for rec in (prefill_rec, decode_rec):
            assert rec.operations, kind
            for op in rec.operations:
                assert op["source_line"] and op["source_url"].endswith(f"#L{op['source_line']}")
                assert len(op["axes"]) == len(op["shape"])
                assert {"mean", "std", "min", "max", "slice", "input_shapes", "dtype"} <= set(op)
        assert decode_rec.operations[0]["phase"] == "decode"
        if kind == "delta":
            assert any(op["name"] == "convolution_history" for op in decode_rec.operations)
            assert prefill_rec.trajectory and prefill_rec.trajectory[-1]["position"] == 5


def test_source_anchors_resolve_to_expected_code():
    lines = source_lines()
    text = open(upstream.__file__).read().splitlines()
    for group, table in ANCHORS.items():
        for name, spec in table.items():
            assert spec[1] in text[lines[group][name] - 1], (group, name)
    assert set(AXES["attention"]) <= set(ANCHORS["attention"])
    assert set(AXES["delta"]) <= set(ANCHORS["delta"])
    assert lines["site"]["mixer_output"] < lines["site"]["residual_output"]


def test_tolerance_check_semantics():
    a = torch.linspace(1.0, 2.0, 10)
    bump = torch.zeros(10)
    bump[0] = 0.25  # max 0.25, mean 0.025
    assert check("x", "y", "z", "same_kernel", torch.bfloat16, a, a.clone(), greedy=True)["passed"]
    assert not check("x", "y", "z", "same_kernel", torch.bfloat16, a + 2 ** -7, a, greedy=True)["passed"]
    assert check("x", "y", "z", "logits_cross_kernel", torch.bfloat16, a + bump, a, greedy=True)["passed"]
    assert not check("x", "y", "z", "logits_cross_kernel", torch.bfloat16, a + 0.25, a, greedy=True)["passed"]  # mean too large
    assert not check("x", "y", "z", "logits_cross_kernel", torch.bfloat16, a + bump, a, greedy=False)["passed"]
    assert not check("x", "y", "z", "logits_cross_kernel", torch.float32, a + bump, a, greedy=True)["passed"]
    result = check("x", "y", "z", "layer_cross_kernel", torch.bfloat16, a + 0.01, a)
    assert result["passed"] and abs(result["scaled_max_error"] - 0.005) < 1e-6
    assert not check("x", "y", "z", "layer_cross_kernel", torch.bfloat16, a + 0.04, a)["passed"]
    assert set(TOLERANCES["torch.bfloat16"]) == set(TOLERANCES["torch.float32"])
