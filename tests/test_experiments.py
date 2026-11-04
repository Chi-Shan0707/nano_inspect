"""Experiment machinery: statistics, aligned task construction, policies and an end-to-end mini run."""
import json
import random
import pytest
import torch
from nanoinspect.runtime import ROOT, read_config
from nanoinspect.experiments.stats import summarize
from nanoinspect.experiments.retrieval import batches
from nanoinspect.experiments.steering import generate, _steer
from nanoinspect.experiments import tasks as T
from nanoinspect.interventions import Intervention, hook_count

MODEL_DIR = ROOT / read_config()["model_dir"]
needs_tokenizer = pytest.mark.skipif(not (MODEL_DIR / "tokenizer.json").is_file(), reason="local snapshot unavailable")
needs_model = pytest.mark.skipif(not (MODEL_DIR / "config.json").is_file() or not torch.cuda.is_available(),
                                 reason="local model snapshot or CUDA unavailable")


def test_summarize_reports_uncertainty():
    stats = summarize([1.0, 2.0, 3.0, 4.0], seed=1)
    assert stats["n"] == 4 and stats["mean"] == 2.5 and stats["median"] == 2.5
    assert abs(stats["sem"] - (1.6666666 / 4) ** 0.5) < 1e-6
    assert stats["ci95"][0] <= 2.5 <= stats["ci95"][1]
    assert summarize([float("nan"), None])["n"] == 0
    assert summarize([5.0])["ci95"] == [5.0, 5.0]


class Item:
    def __init__(self, length):
        self.ids = [0] * length


def test_batches_never_mix_lengths_and_respect_budget():
    items = [Item(10)] * 5 + [Item(30)] * 3
    groups = list(batches(items, max_tokens=60, max_batch=4))
    assert sorted(i for g in groups for i in g) == list(range(8))
    for group in groups:
        assert len({len(items[i].ids) for i in group}) == 1
        assert len(group) * len(items[group[0]].ids) <= 60


@needs_tokenizer
def test_retrieval_examples_are_aligned_and_corruption_changes_one_token():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    for fmt in ("sentence", "list"):
        examples = T.retrieval_examples(tokenizer, fmt, 4, 3, 8, seed=0, split="test")
        assert len({len(e.ids) for e in examples}) == 1
        for example in examples:
            assert example.ids[example.positions["value"]] == example.target_id
            assert tokenizer.decode([example.ids[example.positions["query_key"]]]).strip() == example.keys[example.query]
            assert example.distance == len(example.ids) - 1 - example.positions["value"]
            corrupted = T.corrupt(tokenizer, example, random.Random(0))
            assert corrupted is not None and corrupted.target not in example.values
            assert [i for i, (a, b) in enumerate(zip(example.ids, corrupted.ids)) if a != b] == [example.positions["value"]]
    pilot = T.retrieval_examples(tokenizer, "sentence", 4, 0, 5, seed=0, split="pilot")
    heldout = T.retrieval_examples(tokenizer, "sentence", 4, 0, 5, seed=0, split="heldout")
    assert [e.text for e in pilot] != [e.text for e in heldout]
    assert [e.text for e in pilot] == [e.text for e in T.retrieval_examples(tokenizer, "sentence", 4, 0, 5, seed=0, split="pilot")]


@needs_tokenizer
def test_steering_splits_are_disjoint_and_aligned():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    T.check_disjoint()
    pairs = T.steering_pairs(tokenizer)
    assert len(pairs) >= 20
    heldout = T.steering_prompts(tokenizer, T.HELDOUT_NOUNS, limit=60)
    assert len(heldout) >= 50 and len({len(p["ids"]) for p in heldout}) == 1
    nouns = {p["noun"] for p in heldout}
    assert not nouns & set(T.TRAIN_NOUNS) and not nouns & set(T.PILOT_NOUNS)


def _tiny_model():
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    config = Qwen3_5TextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2, vocab_size=64,
                               layer_types=["linear_attention", "full_attention"], num_attention_heads=4,
                               num_key_value_heads=2, head_dim=4, linear_num_key_heads=2, linear_num_value_heads=2,
                               linear_key_head_dim=4, linear_value_head_dim=4, linear_conv_kernel_dim=3)
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(config).eval()


def test_prefill_only_and_decode_time_policies():
    model = _tiny_model()
    ids = torch.randint(0, 64, (2, 5))
    direction = torch.randn(16)
    calls = []

    def factory():
        stack = _steer(model, 0, "residual_output", direction, 3.0)
        calls.append(stack.intervention)
        return stack

    with torch.no_grad():
        plain = generate(model, ids, 4)
        zero = generate(model, ids, 4, lambda: _steer(model, 0, "residual_output", direction, 0.0))
        prefill = generate(model, ids, 4, factory, "prefill_only")
        decode = generate(model, ids, 4, factory, "decode_time")
    assert torch.equal(plain, zero)
    assert torch.equal(prefill[:, 0], decode[:, 0])  # identical prefill
    assert calls[0].calls == 1 and calls[1].calls == 4  # hook removed after prefill vs active every step
    assert hook_count(model) == 0
    with pytest.raises(ValueError):
        generate(model, ids, 2, factory, "sometimes")
    assert hook_count(model) == 0


@needs_model
def test_mini_experiment_end_to_end(tmp_path):
    from nanoinspect.experiments import run_experiment_command
    import yaml
    config = {
        "seed": 0, "studies": ["retrieval", "patching", "steering"], "max_batch_tokens": 8000, "max_batch": 32,
        "capture_max_bytes": 268435456,
        "retrieval": {"formats": ["sentence", "list"], "n_facts": [2], "filler_sentences": [0, 5],
                      "pilot_examples": 3, "heldout_examples": 4, "solvable_accuracy": 0.9},
        "patching": {"format": "sentence", "n_facts": 3, "filler_sentences": 0, "pilot_examples": 3, "heldout_examples": 4,
                     "candidate_multiplier": 4, "layers": [0, 12, 23], "sites": ["mixer_output", "residual_output"],
                     "positions": ["value", "final", "control"], "random_control_sites": ["residual_output"],
                     "random_control_positions": ["value"], "random_control_seeds": [0]},
        "steering": {"site": "residual_output", "candidate_layers": [9], "layer": "auto", "pilot_alpha": 0.25,
                     "alphas": [-0.25, 0.0, 0.25], "random_seeds": [0], "heldout_prompts": 50, "min_heldout": 50,
                     "generate_tokens": 3}}
    path = tmp_path / "mini.yaml"
    path.write_text(yaml.safe_dump(config))
    result = run_experiment_command(path, tmp_path / "run")
    assert result["summary"]["all_controls_passed"], result["summary"]
    patching = result["studies"]["patching"]
    assert patching["patched_layers"] == [0, 12, 23]
    value_zero = next(r for r in patching["summary"] if r["position"] == "value" and r["site"] == "residual_output" and r["layer"] == 0)
    assert value_zero["recovery_mean"] > 0.9  # restoring the queried value's own residual at layer 0 restores the answer
    control = [r for r in patching["summary"] if r["position"] == "control"]
    assert all(abs(r["recovery_mean"]) < 0.5 for r in control)
    for key in ("run", "layer", "positions", "phase", "site", "shape", "dtype", "model_revision"):
        assert patching["capture_metadata_example"][key] is not None
    steering = result["studies"]["steering"]
    assert len({r["direction"] for r in steering["sweep"]}) == 2
    for name in ("report.html", "report.md", "patching-heatmap.png", "steering-sweep.png", "details/patching.pt"):
        assert (tmp_path / "run" / name).is_file(), name
    assert json.loads((tmp_path / "run" / "experiment.json").read_text())["kind"] == "experiment"
