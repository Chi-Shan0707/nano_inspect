"""CPU checks of the localhost workbench analyses and server on a tiny random Qwen3.5 model."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
import pytest
import torch
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from nanoinspect import server, workbench
from nanoinspect.interventions import hook_count


class CharTokenizer:
    """One token per character (id = ord % 64): enough for the workbench's tokenizer calls."""
    eos_token_id = pad_token_id = None

    def convert_tokens_to_ids(self, token):
        return None

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [ord(c) % 64 for c in text]
        return SimpleNamespace(input_ids=torch.tensor([ids]) if return_tensors == "pt" else ids)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(64 + int(i)) for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        return f"<{messages[0]['content']}>"


@pytest.fixture(scope="module")
def runtime():
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=3,
        layer_types=["linear_attention", "linear_attention", "full_attention"], num_attention_heads=4, vocab_size=64,
        num_key_value_heads=2, head_dim=4, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=4, linear_value_head_dim=4, linear_conv_kernel_dim=3)
    config._attn_implementation = "eager"
    model = Qwen3_5ForCausalLM(config).eval()
    return SimpleNamespace(model=model, tokenizer=CharTokenizer(), device=torch.device("cpu"),
                           dtype=torch.float32, config={"model_id": "tiny"}, provenance={})


def test_lens_grid_final_layer_matches_model_output(runtime):
    result = workbench.lens(runtime, "abcdefg", top_k=3)
    assert [row["layer"] for row in result["lens"]] == ["embed", 0, 1, 2]
    assert all(len(row["cells"]) == 7 for row in result["lens"])
    final, model_top = result["lens"][-1]["cells"][-1], result["model_top"]
    assert [token for token, _ in final["top"]] == [x["token"] for x in model_top]
    assert final["top"][0][1] == pytest.approx(model_top[0]["prob"], abs=1e-3)
    assert final["target_rank"] == 1  # default target is the model's top-1 at the last position
    for row in result["lens"]:
        for cell in row["cells"]:
            probs = [p for _, p in cell["top"]]
            assert probs == sorted(probs, reverse=True) and cell["target_rank"] >= 1
    assert {site: (len(v), len(v[0])) for site, v in result["norms"].items()} == \
        {site: (3, 7) for site in workbench.LENS_SITES}
    assert hook_count(runtime.model) == 0


@pytest.mark.parametrize("layer,kind", [(0, "linear_attention"), (2, "full_attention")])
def test_walkthrough_records_operations_and_worked_example(runtime, layer, kind):
    result = workbench.walkthrough(runtime, "abcdef", layer, position=-1, head="auto", compare="b")
    assert result["kind"] == kind and result["position"] == 5 and result["head_chosen"] == "auto"
    assert 0 <= result["head"] < result["heads"]
    names = [op["name"] for op in result["operations"]]
    assert names[0] == "mixer_input" and names[-1] == "projection"
    expected = {"probabilities", "query_rotary", "gated_output"} if kind == "full_attention" else \
        {"state_after", "log_decay", "post_convolution_qkv"}
    assert expected <= set(names)
    assert all(len(op["axes"]) == len(op["shape"]) and op["source_url"].endswith(f"#L{op['source_line']}")
               for op in result["operations"])
    assert len(result["example"]["head_contributions"]) == result["heads"]
    chosen = workbench.walkthrough(runtime, "abcdef", layer, position=2, head=1)
    assert chosen["head"] == 1 and chosen["head_chosen"] == "given" and chosen["position"] == 2
    assert hook_count(runtime.model) == 0
    with pytest.raises(IndexError):
        workbench.walkthrough(runtime, "abcdef", layer, head=99)


def test_mixer_attention_is_causal_and_normalized(runtime):
    result = workbench.mixer(runtime, "abcdef", 2)
    probs = torch.tensor(result["probabilities"])
    assert result["kind"] == "full_attention" and probs.shape == (4, 6, 6)
    torch.testing.assert_close(probs.sum(-1), torch.ones(4, 6), atol=2e-3, rtol=0)
    assert torch.all(probs.triu(1) == 0)


def test_mixer_delta_reports_gates_and_state(runtime):
    result = workbench.mixer(runtime, "abcdef", 0)
    assert result["kind"] == "linear_attention" and result["heads"] == 2
    for key in ("beta", "decay", "state_norm", "write_norm"):
        assert torch.tensor(result[key]).shape == (6, 2), key
    decay = torch.tensor(result["decay"])
    assert torch.all((decay >= 0) & (decay <= 1))  # exp(g); random init can round to 0
    assert "forward" not in runtime.model.model.layers[0].linear_attn.__dict__


def test_patch_restoration_semantics(runtime):
    sites = ["residual_output", "post_mixer_residual", "mixer_output", "mlp_output"]
    result = workbench.patch(runtime, "abcdefgh", "abcxefgh", sites=sites, target="a", foil="b")
    assert result["differing_positions"] == [3] and result["sites"] == sites
    for site in sites:
        grid = torch.tensor(result["grids"][site]["restoration"])
        assert grid.shape == (3, 8), site
        # Positions before the edit carry identical activations: patching them cannot change anything.
        assert torch.all(grid[:, :3].abs() < 1e-4), site
    residual = torch.tensor(result["grids"]["residual_output"]["restoration"])
    assert residual[-1, -1].item() == pytest.approx(1.0, abs=1e-4)  # final residual at final position is the output
    assert hook_count(runtime.model) == 0


def test_patch_rejects_unequal_lengths(runtime):
    with pytest.raises(ValueError, match="equal token counts"):
        workbench.patch(runtime, "abc", "abcd", target="a", foil="b")


def test_custom_steering_zero_alpha_is_identity(runtime, monkeypatch):
    monkeypatch.setattr(workbench, "steering_assets", lambda: None)
    result = workbench.steer(runtime, "abcd", 1, 0.0, direction="custom", steps=4, positive=["ab", "cd"], negative=["ba", "dc"])
    assert result["baseline"] == result["steered"] and result["kl"] == pytest.approx(0, abs=1e-6)
    assert "this prompt" in result["unit_source"]
    moved = workbench.steer(runtime, "abcd", 1, 1.0, direction="custom", steps=4, positive=["ab", "cd"], negative=["ba", "dc"])
    assert moved["strength"] == pytest.approx(moved["unit_value"])
    assert hook_count(runtime.model) == 0
    with pytest.raises(ValueError, match="Saved Study C directions are absent"):
        workbench.steer(runtime, "abcd", 1, 0.5, direction="contrastive")


def test_steer_sweep_has_control_and_identity_at_zero(runtime, monkeypatch):
    monkeypatch.setattr(workbench, "steering_assets", lambda: None)
    result = workbench.steer_sweep(runtime, "abcd", 1, [-1, 1], direction="random_seed_3", control="random_seed_4",
                                   steps=3, watch_positive=["a"], watch_negative=["b"])
    assert set(result["series"]) == {"random_seed_3", "random_seed_4"}
    for rows in result["series"].values():
        assert [r["alpha"] for r in rows] == [-1.0, 0.0, 1.0]  # alpha 0 is always included
        zero = rows[1]
        assert zero["kl"] == pytest.approx(0, abs=1e-6) and "watch" in zero
    zeros = [rows[1] for rows in result["series"].values()]
    assert zeros[0]["text"] == zeros[1]["text"] and zeros[0]["watch"] == zeros[1]["watch"]
    assert hook_count(runtime.model) == 0


def test_static_paths_stay_inside_their_roots():
    assert server.static_path("/static/js/main.js") == (server.WEB / "js/main.js").resolve()
    for path in ("/runs/../pyproject.toml", "/runs/%2e%2e/pyproject.toml", "/static/../server.py",
                 "/runs/%2fetc%2fpasswd", "/docs/../../etc/passwd", "/elsewhere/x"):
        assert server.static_path(path) is None, path


def test_non_finite_values_become_strict_json():
    encoded = server._finite({"a": [float("-inf"), 1.5, float("nan")], "b": (float("inf"),), "c": "x"})
    assert encoded == {"a": ["-inf", 1.5, "nan"], "b": ["inf"], "c": "x"}
    json.dumps(encoded, allow_nan=False)


def test_server_round_trip(runtime):
    state = server.State()
    state.runtime, state.status = runtime, "ready"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(state))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        def post(name, body):
            request = urllib.request.Request(f"{base}/api/{name}", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(request))
        assert json.load(urllib.request.urlopen(f"{base}/api/status"))["status"] == "ready"
        assert b"nanoinspect" in urllib.request.urlopen(f"{base}/").read()
        assert post("mixer", {"prompt": "abcd", "layer": 2})["kind"] == "full_attention"
        # Masked attention scores are -inf; the response must still be strict JSON.
        request = urllib.request.Request(f"{base}/api/walkthrough", json.dumps({"prompt": "abcd", "layer": 2}).encode(),
                                         {"Content-Type": "application/json"})
        def reject(constant):
            raise ValueError(f"non-standard JSON constant {constant}")
        walk = json.loads(urllib.request.urlopen(request).read(), parse_constant=reject)
        assert walk["operations"][-1]["name"] == "projection"
        with pytest.raises(urllib.error.HTTPError) as error:
            post("mixer", {"prompt": "", "layer": 2})
        assert error.value.code == 400 and "empty" in json.load(error.value)["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()
