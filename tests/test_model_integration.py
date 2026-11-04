"""Integration checks against the local Qwen3.5-0.8B snapshot (skipped when unavailable)."""
import json
import pytest
import torch
from nanoinspect.runtime import ROOT, read_config

MODEL_DIR = ROOT / read_config()["model_dir"]
pytestmark = pytest.mark.skipif(not (MODEL_DIR / "config.json").is_file() or not torch.cuda.is_available(),
                                reason="local model snapshot or CUDA unavailable")


@pytest.fixture(scope="module")
def runtime():
    from nanoinspect.runtime import load_model
    return load_model()


def test_trace_parity_and_operation_records(runtime, tmp_path):
    from nanoinspect.trace import run_trace
    config = {"prompt": "The red box contains the apple. The blue box contains the coin. What is in the blue box? The",
              "chat_template": False, "layer": [2, 3], "token": [-1], "head": "auto", "compare_tokens": [" coin"],
              "padding_sweep": [2]}
    result = run_trace(runtime, config, tmp_path)
    assert result["parity_summary"]["all_passed"], [p for p in result["parity"] if not p["passed"]]
    cases = {(p["case"], p.get("layer")) for p in result["parity"]}
    for layer in (2, 3):
        assert {("prefill", layer), ("cached_decode", layer), ("left_padding", layer)} <= cases
    assert ("recurrent_substitution", 2) in cases and ("causal_mask", 3) in cases
    names = {(op["layer"], op["name"]) for op in result["operations"]}
    assert {(2, "state_after"), (2, "log_decay"), (2, "post_convolution_qkv"), (3, "probabilities"),
            (3, "query_rotary"), (3, "gated_output")} <= names
    for op in result["operations"]:
        assert op["source_url"].endswith(f"#L{op['source_line']}") and len(op["axes"]) == len(op["shape"])
    for capture in result["captures"]:
        for key in ("run", "layer", "token_ids", "positions", "phase", "site", "shape", "dtype", "model_revision"):
            assert capture[key] is not None, key
    mixers = {example["layer"]: example["mixer"] for example in result["worked_examples"]}
    assert mixers[3] == "full_attention" and mixers[2].startswith("linear_attention")
    assert json.loads((tmp_path / "trace.json").read_text())["kind"] == "trace"
