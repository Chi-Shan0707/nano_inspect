"""Reproducible, bounded numerical traces of Qwen3.5 mixers (M2 computation microscope).

``run_trace`` follows one prompt through a selected full-attention layer and a
selected gated-delta layer. Every recorded operation carries its input/output
shapes, axis meanings, dtype, summary statistics over the complete tensor, a
bounded value slice for the selected head/token, and the exact line of the
pinned upstream source. Worked numerical examples recompute one token/head by
hand, and parity checks compare instrumented with untouched execution using
the dtype-specific tolerances in ``TOLERANCES``.
"""
from __future__ import annotations

from contextlib import ExitStack, nullcontext
import gc
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as upstream
from .runtime import load_model, read_config, write_json
from .sites import decoder_layers, resolve_site, SITE_NAMES
from .capture import Capture, capture_metadata
from .interventions import Intervention, hook_count
from .adapters.qwen35 import instrument

# Tolerances are pass/fail criteria derived from measured baselines (see
# docs/findings.md, "Chunked versus recurrent gated-delta numerics"):
# * same_kernel: instrumented code executes the identical operation sequence,
#   so any difference is a bug. Measured: exactly 0 in BF16 and FP32.
# * layer_cross_kernel: chunked vs token-recurrent gated-delta output for one
#   layer, as max |a-b| / max |reference|. Measured <= 0.63 eps (BF16) and
#   <= 2.6 eps (FP32) on the trace prompt.
# * logits_cross_kernel: final logits after a mathematically equivalent but
#   differently rounded kernel. The untouched BF16 model itself moves by
#   0.16-0.25 (max) / <= 0.035 (mean) under left padding or cached-vs-uncached
#   decoding; one-layer recurrent substitution measured 0.19-0.31 / <= 0.021.
#   FP32 measured <= 3.0e-5 / 2.4e-6. Tolerances are ~1.6-3x those maxima.
TOLERANCES = {
    "torch.bfloat16": {
        "same_kernel": {"max_abs": 0.0, "mean_abs": 0.0, "greedy": True},
        "layer_cross_kernel": {"scaled_max": 2 * 2 ** -7},
        "logits_cross_kernel": {"max_abs": 0.5, "mean_abs": 0.05, "greedy": True},
    },
    "torch.float32": {
        "same_kernel": {"max_abs": 0.0, "mean_abs": 0.0, "greedy": True},
        "layer_cross_kernel": {"scaled_max": 64 * 2 ** -23},
        "logits_cross_kernel": {"max_abs": 1e-4, "mean_abs": 1e-5, "greedy": True},
    },
}
TOLERANCES["torch.float16"] = TOLERANCES["torch.bfloat16"]  # provisional: float16 not measured on this machine

# Axis meanings per recorded operation, batch axis first.
AXES = {
    "attention": {
        "mixer_input": ["batch", "token", "hidden"],
        "query_projection": ["batch", "token", "head", "feature"],
        "output_gate_logits": ["batch", "token", "head", "feature"],
        "key_projection": ["batch", "token", "kv_head", "feature"],
        "value_projection": ["batch", "token", "kv_head", "feature"],
        "query_normalized": ["batch", "head", "token", "feature"],
        "key_normalized": ["batch", "kv_head", "token", "feature"],
        "value": ["batch", "kv_head", "token", "feature"],
        "rotary_cos": ["batch", "token", "rotary_feature"],
        "rotary_sin": ["batch", "token", "rotary_feature"],
        "query_rotary": ["batch", "head", "token", "feature"],
        "key_rotary": ["batch", "kv_head", "token", "feature"],
        "key_grouped": ["batch", "head", "key_token", "feature"],
        "value_grouped": ["batch", "head", "key_token", "feature"],
        "scores": ["batch", "head", "token", "key_token"],
        "mask": ["batch", "mask_head", "token", "key_token"],
        "masked_scores": ["batch", "head", "token", "key_token"],
        "probabilities": ["batch", "head", "token", "key_token"],
        "value_aggregation": ["batch", "token", "head", "feature"],
        "output_gate": ["batch", "token", "head_feature"],
        "gated_output": ["batch", "token", "head_feature"],
        "projection": ["batch", "token", "hidden"],
    },
    "delta": {
        "mixer_input": ["batch", "token", "hidden"],
        "padding_masked_input": ["batch", "token", "hidden"],
        "pre_convolution_qkv": ["batch", "channel", "token"],
        "convolution_history": ["batch", "channel", "history"],
        "post_convolution_qkv": ["batch", "token", "channel"],
        "output_gate_logits": ["batch", "token", "head", "feature"],
        "beta_logits": ["batch", "token", "head"],
        "decay_logits": ["batch", "token", "head"],
        "beta": ["batch", "token", "head"],
        "log_decay": ["batch", "token", "head"],
        "value": ["batch", "token", "head", "feature"],
        "kernel_readout": ["batch", "token", "head", "feature"],
        "query_normalized_scaled": ["batch", "head", "token", "feature"],
        "key_normalized": ["batch", "head", "token", "feature"],
        "state_before": ["batch", "head", "key_feature", "value_feature"],
        "state_memory_readout": ["batch", "head", "value_feature"],
        "state_delta": ["batch", "head", "value_feature"],
        "state_after": ["batch", "head", "key_feature", "value_feature"],
        "readout": ["batch", "head", "token", "feature"],
        "gated_output": ["batch", "token", "head_feature"],
        "projection": ["batch", "token", "hidden"],
    },
}

# (scope marker, anchor text, occurrence) locating each operation in the pinned source.
_LAYER, _ATTN, _EAGER = "class Qwen3_5DecoderLayer(", "class Qwen3_5Attention(", "def eager_attention_forward("
_DELTA, _RECUR = "class Qwen3_5GatedDeltaNet(", "def torch_recurrent_gated_delta_rule("
ANCHORS = {
    "attention": {
        "mixer_input": (_LAYER, "hidden_states = self.input_layernorm(hidden_states)"),
        "query_projection": (_ATTN, "self.q_proj(hidden_states).view("),
        "output_gate_logits": (_ATTN, "self.q_proj(hidden_states).view("),
        "key_projection": (_ATTN, "key_states = self.k_norm(self.k_proj(hidden_states)"),
        "value_projection": (_ATTN, "value_states = self.v_proj(hidden_states)"),
        "query_normalized": (_ATTN, "query_states = self.q_norm("),
        "key_normalized": (_ATTN, "key_states = self.k_norm(self.k_proj(hidden_states)"),
        "value": (_ATTN, "value_states = self.v_proj(hidden_states)"),
        "rotary_cos": (_ATTN, "cos, sin = position_embeddings"),
        "rotary_sin": (_ATTN, "cos, sin = position_embeddings"),
        "query_rotary": (_ATTN, "apply_rotary_pos_emb(query_states, key_states, cos, sin)"),
        "key_rotary": (_ATTN, "apply_rotary_pos_emb(query_states, key_states, cos, sin)"),
        "key_grouped": (_EAGER, "key_states = repeat_kv(key"),
        "value_grouped": (_EAGER, "value_states = repeat_kv(value"),
        "scores": (_EAGER, "attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling"),
        "mask": (_EAGER, "attn_weights = attn_weights + attention_mask"),
        "masked_scores": (_EAGER, "attn_weights = attn_weights + attention_mask"),
        "probabilities": (_EAGER, "nn.functional.softmax(attn_weights"),
        "value_aggregation": (_EAGER, "attn_output = torch.matmul(attn_weights, value_states)"),
        "output_gate": (_ATTN, "attn_output = attn_output * torch.sigmoid(gate)"),
        "gated_output": (_ATTN, "attn_output = attn_output * torch.sigmoid(gate)"),
        "projection": (_ATTN, "attn_output = self.o_proj(attn_output)"),
    },
    "delta": {
        "mixer_input": (_LAYER, "hidden_states = self.input_layernorm(hidden_states)"),
        "padding_masked_input": (_DELTA, "hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)"),
        "pre_convolution_qkv": (_DELTA, "mixed_qkv = self.in_proj_qkv(hidden_states)"),
        "convolution_history": (_DELTA, "mixed_qkv = causal_conv1d_update("),
        "post_convolution_qkv": (_DELTA, "mixed_qkv = causal_conv1d_fn("),
        "output_gate_logits": (_DELTA, "z = self.in_proj_z(hidden_states)"),
        "beta_logits": (_DELTA, "b = self.in_proj_b(hidden_states)"),
        "decay_logits": (_DELTA, "a = self.in_proj_a(hidden_states)"),
        "beta": (_DELTA, "beta = b.sigmoid()"),
        "log_decay": (_DELTA, "g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)"),
        "value": (_DELTA, "query, key, value = torch.split("),
        "kernel_readout": (_DELTA, "core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule("),
        "query_normalized_scaled": (_RECUR, "query = query / (query.shape[-1] ** 0.5)"),
        "key_normalized": (_RECUR, "key = l2norm(key, dim=-1, eps=1e-6)"),
        "state_before": (_RECUR, "last_recurrent_state = last_recurrent_state * decay_t"),
        "state_memory_readout": (_RECUR, "kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)"),
        "state_delta": (_RECUR, "delta = (v_t - kv_mem) * beta_t"),
        "state_after": (_RECUR, "last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)"),
        "readout": (_RECUR, "core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)"),
        "gated_output": (_DELTA, "core_attn_out = self.norm(core_attn_out, z)"),
        "projection": (_DELTA, "output = self.out_proj(core_attn_out)"),
    },
    "site": {
        "residual_input": (_LAYER, "residual = hidden_states"),
        "mixer_output": (_LAYER, "hidden_states = residual + hidden_states", 1),
        "post_mixer_residual": (_LAYER, "hidden_states = self.post_attention_layernorm(hidden_states)"),
        "mlp_output": (_LAYER, "hidden_states = self.mlp(hidden_states)"),
        "residual_output": (_LAYER, "hidden_states = residual + hidden_states", 2),
        "final_norm": ("class Qwen3_5TextModel(", "hidden_states = self.norm(hidden_states)"),
    },
}


def source_line(scope, anchor, occurrence=1, lines=None):
    """1-based line of the ``occurrence``-th ``anchor`` after the ``scope`` marker in the pinned source."""
    lines = lines if lines is not None else Path(upstream.__file__).read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if scope in line), None)
    if start is None:
        raise LookupError(f"Scope {scope!r} not found in {upstream.__file__}")
    seen = 0
    for i in range(start, len(lines)):
        if anchor in lines[i]:
            seen += 1
            if seen == occurrence:
                return i + 1
    raise LookupError(f"Anchor {anchor!r} not found after {scope!r}")


def source_lines():
    """Resolve every anchor; raises if the installed source no longer matches."""
    text = Path(upstream.__file__).read_text().splitlines()
    return {group: {name: source_line(*spec, lines=text) for name, spec in table.items()}
            for group, table in ANCHORS.items()}


def errors(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    diff = (a - b).abs()
    scale = b.abs().max().item()
    return {"max_abs_error": diff.max().item(), "mean_abs_error": diff.mean().item(),
            "max_abs_reference": scale, "scaled_max_error": diff.max().item() / scale if scale else 0.0,
            "exact": bool(torch.equal(a, b))}


def check(case, target, comparison, tolerance_class, dtype, a, b, greedy=None, **extra):
    """Compare ``a`` (candidate) with ``b`` (reference) under a named tolerance."""
    tolerance = TOLERANCES[str(dtype)][tolerance_class]
    result = {"case": case, "target": target, "comparison": comparison, "dtype": str(dtype),
              "tolerance_class": tolerance_class, "tolerance": tolerance, **extra, **errors(a, b)}
    passed = True
    if "max_abs" in tolerance:
        passed &= result["max_abs_error"] <= tolerance["max_abs"]
    if "mean_abs" in tolerance:
        passed &= result["mean_abs_error"] <= tolerance["mean_abs"]
    if "scaled_max" in tolerance:
        passed &= result["scaled_max_error"] <= tolerance["scaled_max"]
    if greedy is not None:
        result["greedy_equal"] = bool(greedy)
        if tolerance.get("greedy"):
            passed &= bool(greedy)
    result["passed"] = bool(passed)
    return result


def format_prompt(tokenizer, config):
    prompt = config.get("prompt", "The red key opens the blue box.")
    if config.get("chat_template", True):
        return prompt, tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                     add_generation_prompt=True, enable_thinking=False)
    return prompt, prompt


def _as_list(value):
    return value if isinstance(value, list) else [value]


def _mixer(model, layer):
    site = resolve_site(model, layer, "mixer_output")
    return site.module, ("attention" if site.path.endswith("self_attn") else "delta")


def _text_model(model):
    path, _ = decoder_layers(model)
    return model.get_submodule(path.rsplit(".", 1)[0])


class Recorder:
    """Collect bounded operation records for one instrumented forward pass."""

    def __init__(self, kind, module, layer, token, head, phase, lines, source_url, position_offset=0):
        self.kind, self.module, self.layer, self.token, self.head, self.phase = kind, module, layer, token, head, phase
        self.lines, self.source_url, self.position_offset = lines, source_url, position_offset
        self.shapes, self.held, self.operations, self.trajectory = {}, {}, [], []
        if kind == "attention":
            self.head_dim = module.head_dim
            self.kv_head = head // module.num_key_value_groups
            self.channel = None
        else:
            self.head_dim = module.head_v_dim
            self.kv_head = head // (module.num_v_heads // module.num_k_heads)
            self.channel = (self.kv_head * module.head_k_dim, module.head_k_dim)  # query channels of the selected head

    def _view(self, data, axes):
        """Select the requested head/token from a batch-0 tensor; axes exclude batch."""
        for axis, label in reversed(list(enumerate(axes))):
            if label == "head":
                data = data.narrow(axis, self.head, 1)
            elif label in ("kv_head",):
                data = data.narrow(axis, self.kv_head, 1)
            elif label == "head_feature":
                data = data.narrow(axis, self.head * self.head_dim, self.head_dim)
            elif label == "channel" and self.channel:
                data = data.narrow(axis, *self.channel)
            elif label == "token" and data.shape[axis] > self.token:
                data = data.narrow(axis, self.token, 1)
        return data

    def __call__(self, name, tensor, inputs=(), token=None):
        tensor = tensor.detach()
        if token is not None:
            if name == "state_after":
                state = tensor[0, self.head].float()
                self.trajectory.append({"position": token + self.position_offset,
                                        "state_frobenius_norm": state.norm().item()})
            if token != self.token:
                return
        axes = AXES[self.kind].get(name)
        if axes is None:
            raise KeyError(f"No axis semantics for {self.kind} operation {name!r}")
        if len(axes) != tensor.ndim:
            raise ValueError(f"{name}: axes {axes} do not match shape {tuple(tensor.shape)}")
        full = tensor.float()
        held = tensor[0].float().cpu()
        self.held[name] = held
        self.shapes[name] = list(tensor.shape)
        view = self._view(held, axes[1:])
        input_shapes = {}
        for item in inputs:
            if item in self.shapes:
                input_shapes[item] = self.shapes[item]
            else:
                try:
                    input_shapes[item] = list(self.module.get_parameter(item).shape)
                except AttributeError:
                    input_shapes[item] = "external (layer input, cache, mask or position state)"
        line = self.lines[self.kind].get(name)
        self.operations.append({
            "layer": self.layer, "mixer": self.kind, "phase": self.phase, "name": name,
            "order": len(self.operations), "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "axes": axes, "inputs": list(inputs), "input_shapes": input_shapes,
            "selected_head": self.head if any(a in axes for a in ("head", "head_feature")) else None,
            "selected_kv_head": self.kv_head if "kv_head" in axes else None,
            "selected_token": (self.token + self.position_offset) if ("token" in axes or token is not None) else None,
            "slice_shape": list(view.shape), "slice": view.flatten()[:8].tolist(),
            "mean": full.mean().item(), "std": full.std(unbiased=False).item(),
            "min": full.min().item(), "max": full.max().item(),
            "source_line": line, "source_url": f"{self.source_url}#L{line}"})


class _Switch:
    """Recorder indirection, so a cached decode can record only the decode step."""
    def __init__(self):
        self.target = None

    def __call__(self, *args, **kwargs):
        if self.target is not None:
            self.target(*args, **kwargs)


def _final_residual(model):
    """Context capturing the input of the final RMSNorm (the full residual stream)."""
    norm = _text_model(model).norm
    store = {}
    def hook(module, args, kwargs):
        store["value"] = (args[0] if args else kwargs["hidden_states"]).detach().float().cpu()
    handle = norm.register_forward_pre_hook(hook, with_kwargs=True)
    return handle, store


class Unembed:
    """Direct logit attribution through the final RMSNorm with its scale frozen at the observed residual.

    logit_y(c) = rsqrt(mean(x^2) + eps) * ((1 + w) * c) . W_U[y], where x is
    the actual final residual. Summing over all residual components recovers
    the model's logit up to BF16 rounding of the original computation.
    """
    def __init__(self, model, final_residual):
        norm = _text_model(model).norm
        self.weight = 1.0 + norm.weight.detach().float().cpu()
        self.scale = torch.rsqrt(final_residual.pow(2).mean() + norm.eps).item()
        self.unembedding = model.get_output_embeddings().weight

    def __call__(self, vector, token_ids):
        rows = self.unembedding[token_ids].detach().float().cpu()
        return (rows @ (vector.float().cpu() * self.weight * self.scale)).tolist()


def _token_label(tokenizer, token_id):
    return tokenizer.convert_ids_to_tokens(int(token_id))


def attention_example(rec, module, unembed, token_ids, tokens, selected):
    h, t, kvh = rec.head, rec.token, rec.kv_head
    held = rec.held
    q = held["query_rotary"][h, t]
    keys, values = held["key_grouped"][h], held["value_grouped"][h]
    manual_scores = (keys[: t + 1] @ q) * module.scaling
    recorded_scores = held["scores"][h, t, : t + 1]
    masked = held["masked_scores"][h, t]
    manual_probs = torch.softmax(masked, dim=-1)
    probs = held["probabilities"][h, t]
    manual_agg = probs @ values
    agg = held["value_aggregation"][t, h]
    gate = held["output_gate"][t, h * rec.head_dim:(h + 1) * rec.head_dim]
    gated = held["gated_output"][t]
    weight = module.o_proj.weight.detach().float().cpu()
    heads = gated.view(-1, rec.head_dim)
    contributions = torch.stack([weight[:, i * rec.head_dim:(i + 1) * rec.head_dim] @ heads[i] for i in range(heads.shape[0])])
    top = torch.topk(probs[: t + 1], k=min(6, t + 1))
    return {
        "layer": rec.layer, "mixer": "full_attention", "token_position": t, "token": tokens[t], "query_head": h,
        "kv_head": kvh, "kv_group_size": module.num_key_value_groups, "head_dim": rec.head_dim, "scaling": module.scaling,
        "steps": [
            {"step": "query (after q_norm + partial RoPE), first 6 features", "values": q[:6].tolist(),
             "norm": q.norm().item()},
            {"step": f"score_j = (q . k_j) * {module.scaling:.6f}; highest-probability key positions",
             "rows": [{"key_position": int(j), "key_token": tokens[int(j)], "raw_dot": (keys[int(j)] @ q).item(),
                       "scaled_score": manual_scores[int(j)].item(), "recorded_score": recorded_scores[int(j)].item(),
                       "probability": probs[int(j)].item()} for j in top.indices]},
            {"step": "causal mask: probability mass on future key positions", "value": probs[t + 1:].sum().item()},
            {"step": "softmax recomputed from recorded masked scores (FP32) vs recorded probabilities",
             "max_abs_error": (manual_probs - probs).abs().max().item(), "probability_sum": probs.sum().item()},
            {"step": "value aggregation sum_j p_j v_j, first 6 features",
             "manual": manual_agg[:6].tolist(), "recorded": agg[:6].tolist(),
             "max_abs_error": (manual_agg - agg).abs().max().item()},
            {"step": "output gate sigmoid(g), first 6 features", "values": gate[:6].tolist(),
             "mean_gate": gate.mean().item()},
            {"step": "gated head output = aggregation * gate, first 6 features",
             "values": (agg * gate)[:6].tolist(), "recorded": heads[h][:6].tolist()},
            {"step": "o_proj column block for this head -> residual-width contribution",
             "weight_block_shape": [weight.shape[0], rec.head_dim], "contribution_norm": contributions[h].norm().item(),
             "sum_of_head_contributions_vs_projection_max_abs_error":
                 (contributions.sum(0) - held["projection"][t]).abs().max().item()},
        ],
        "probabilities_over_keys": [{"key_position": j, "key_token": tokens[j], "probability": probs[j].item()}
                                    for j in range(t + 1)],
        "head_contributions": _head_table(contributions, unembed, selected),
    }


def delta_example(rec, module, unembed, token_ids, tokens, selected):
    h, t = rec.head, rec.token
    held = rec.held
    kd, vd = module.head_k_dim, module.head_v_dim
    key_channel = module.key_dim + rec.kv_head * kd  # first key channel of this head
    pre = held["pre_convolution_qkv"][key_channel]
    taps = module.conv1d.weight.detach().float().cpu()[key_channel, 0]
    window = torch.stack([pre[t - module.conv_kernel_size + 1 + i] if t - module.conv_kernel_size + 1 + i >= 0
                          else torch.tensor(0.0) for i in range(module.conv_kernel_size)])
    conv_manual = F.silu((window * taps).sum())
    post = held["post_convolution_qkv"]
    q_raw = post[t, rec.kv_head * kd:(rec.kv_head + 1) * kd]
    k_raw = post[t, module.key_dim + rec.kv_head * kd: module.key_dim + (rec.kv_head + 1) * kd]
    v = held["value"][t, h]
    q = held["query_normalized_scaled"][h, t]
    k = held["key_normalized"][h, t]
    beta = held["beta"][t, h].item()
    g = held["log_decay"][t, h].item()
    before = held["state_before"][h]
    decayed = before * torch.exp(torch.tensor(g))
    memory = k @ decayed
    delta = (v - memory) * beta
    after = decayed + torch.outer(k, delta)
    readout = q @ after
    recorded_after = held["state_after"][h]
    kernel = held["kernel_readout"][t, h]
    z = held["output_gate_logits"][t, h]
    norm_weight = module.norm.weight.detach().float().cpu()
    normalized = readout / torch.sqrt(readout.pow(2).mean() + module.norm.variance_epsilon)
    gated_manual = norm_weight * normalized * F.silu(z)
    gated = held["gated_output"][t].view(-1, vd)
    weight = module.out_proj.weight.detach().float().cpu()
    contributions = torch.stack([weight[:, i * vd:(i + 1) * vd] @ gated[i] for i in range(gated.shape[0])])
    return {
        "layer": rec.layer, "mixer": "linear_attention (gated delta)", "token_position": t, "token": tokens[t],
        "value_head": h, "key_head": rec.kv_head, "key_dim": kd, "value_dim": vd,
        "steps": [
            {"step": f"depthwise causal convolution, key channel {key_channel}: SiLU(sum_i w_i x_(t-{module.conv_kernel_size - 1}+i))",
             "inputs_oldest_to_newest": window.tolist(), "taps": taps.tolist(), "manual": conv_manual.item(),
             "recorded": post[t, key_channel].item()},
            {"step": "L2-normalize q and k; scale q by 1/sqrt(key_dim)",
             "q_norm_before": q_raw.norm().item(), "k_norm_before": k_raw.norm().item(),
             "q_norm_after": q.norm().item(), "k_norm_after": k.norm().item(), "q_first4": q[:4].tolist(),
             "k_first4": k[:4].tolist()},
            {"step": "controls: beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias); decay = exp(g)",
             "beta": beta, "log_decay": g, "decay": float(torch.exp(torch.tensor(g)))},
            {"step": "state decay S' = exp(g) S", "state_shape": list(before.shape),
             "norm_before": before.norm().item(), "norm_after_decay": decayed.norm().item()},
            {"step": "memory readout m = k^T S' (what the state already predicts for this key), first 4",
             "values": memory[:4].tolist(), "recorded": held["state_memory_readout"][h][:4].tolist()},
            {"step": "delta = beta (v - m), first 4", "v_first4": v[:4].tolist(), "values": delta[:4].tolist(),
             "recorded": held["state_delta"][h][:4].tolist()},
            {"step": "rank-one write S = S' + k delta^T", "norm_after_update": after.norm().item(),
             "max_abs_error_vs_recorded_state": (after - recorded_after).abs().max().item()},
            {"step": "readout o = q^T S, first 4 (explicit recurrence vs upstream chunked kernel)",
             "manual": readout[:4].tolist(), "recorded_recurrence": held["readout"][h, t][:4].tolist(),
             "upstream_kernel": kernel[:4].tolist(),
             "recurrence_vs_kernel_max_abs_error": (held["readout"][h, t] - kernel).abs().max().item()},
            {"step": "gated RMSNorm: w * o / rms(o) * SiLU(z), first 4", "manual": gated_manual[:4].tolist(),
             "recorded": gated[h][:4].tolist(), "max_abs_error": (gated_manual - gated[h]).abs().max().item()},
            {"step": "out_proj column block -> residual-width contribution",
             "weight_block_shape": [weight.shape[0], vd], "contribution_norm": contributions[h].norm().item(),
             "sum_of_head_contributions_vs_projection_max_abs_error":
                 (contributions.sum(0) - held["projection"][t]).abs().max().item()},
        ],
        "state_trajectory": rec.trajectory,
        "controls_over_tokens": [{"position": j, "token": tokens[j], "beta": held["beta"][j, h].item(),
                                  "decay": float(torch.exp(held["log_decay"][j, h]))}
                                 for j in range(held["beta"].shape[0])],
        "head_contributions": _head_table(contributions, unembed, selected),
    }


def _head_contributions(module, kind, gated):
    """Per-head residual contributions: output-projection column block times that head's gated output."""
    width = module.head_dim if kind == "attention" else module.head_v_dim
    weight = (module.o_proj if kind == "attention" else module.out_proj).weight.detach().float().cpu()
    heads = gated.view(-1, width)
    return torch.stack([weight[:, i * width:(i + 1) * width] @ heads[i] for i in range(heads.shape[0])])


def _auto_head(model, module, kind, ids, mask, token, unembed, selected):
    """Head with the largest direct target-vs-comparison logit effect (or largest norm with one token)."""
    store = {}
    def keep(name, tensor, inputs=(), token=None):
        if name == "gated_output":
            store["gated"] = tensor[0, token_position].detach().float().cpu()
    token_position = token
    with instrument(module, kind, keep, explain=False):
        model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
    table = _head_table(_head_contributions(module, kind, store["gated"]), unembed, selected)
    key = (lambda row: row["direct_logit_difference"]) if len(selected) >= 2 else (lambda row: row["contribution_norm"])
    return max(table, key=key)["head"]


def _head_table(contributions, unembed, selected):
    ids = [s["id"] for s in selected]
    rows = []
    for index, vector in enumerate(contributions):
        dla = unembed(vector, ids)
        row = {"head": index, "contribution_norm": vector.norm().item(),
               "direct_logit": {s["token"]: value for s, value in zip(selected, dla)}}
        if len(dla) >= 2:
            row["direct_logit_difference"] = dla[0] - dla[1]
        rows.append(row)
    return rows


def run_trace(runtime, config, output):
    """Write trace.json and selected capture tensors. Parity checks carry explicit tolerances."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, tokenizer, dtype = runtime.model, runtime.tokenizer, runtime.dtype
    prompt, formatted = format_prompt(tokenizer, config)
    inputs = tokenizer(formatted, return_tensors="pt").to(runtime.device)
    ids, mask = inputs.input_ids, inputs.attention_mask
    length = ids.shape[1]
    limit = int(config.get("max_trace_tokens", 128))
    if length > limit:
        raise ValueError(f"Prompt has {length} tokens; trace limit is {limit} (max_trace_tokens)")
    token_list = ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_list)
    layer_list = [int(x) for x in _as_list(config.get("layer", [0, 3]))]
    positions = [int(x) for x in _as_list(config.get("token", [-1]))]
    token = positions[0] % length
    names = _as_list(config.get("site", list(SITE_NAMES)))
    head_setting = config.get("head", 0)
    lines = source_lines()
    source = runtime.provenance["transformers_immutable_source_url"]
    run_name = output.name
    hooks_before = hook_count(model)
    with torch.inference_mode():
        top_id = None
        # 1. Untouched baseline with read-only captures (selected positions and full-sequence layer outputs).
        selections = [(layer, name) for layer in layer_list for name in names]
        metadata = capture_metadata(runtime, run_name, "prefill", token_list)
        budget = int(config.get("capture_max_bytes", 8 * 1024 * 1024))
        expected = {"batch_size": 1, "hidden_size": model.config.get_text_config().hidden_size,
                    "element_size": torch.tensor([], dtype=dtype).element_size()}
        handle, final = _final_residual(model)
        try:
            with Capture(model, selections, positions=positions, metadata=metadata, max_bytes=budget,
                         expected=expected) as cap, \
                 Capture(model, [(l, s) for l in layer_list for s in ("mixer_output", "residual_output")],
                         positions="all", metadata=metadata, max_bytes=budget,
                         expected={**expected, "sequence_length": length}) as full_cap:
                baseline = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[0].float().cpu()
            plain = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[0].float().cpu()
        finally:
            handle.remove()
        final_residual = final["value"][0, token]
        top_id = int(baseline[token].argmax())
        selected = [{"token": _token_label(tokenizer, top_id), "id": top_id, "role": "top-1 prediction"}]
        for text in _as_list(config.get("compare_tokens", [])):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"compare_tokens entry {text!r} is not a single token: {encoded}")
            if encoded[0] != top_id:
                selected.append({"token": _token_label(tokenizer, encoded[0]), "id": encoded[0], "role": "comparison"})
        unembed = Unembed(model, final_residual)
        captures = []
        for index, record in enumerate(cap.records):
            filename = f"capture_{index:03d}.pt"
            torch.save(record["tensor"], output / filename)
            line = lines["site"][record["site"]]
            captures.append({**{k: v for k, v in record.items() if k != "tensor"}, "tensor_path": filename,
                             "source_line": line, "source_url": f"{source}#L{line}"})
        parity = [check("read_only_capture", "logits", "captured_vs_plain_forward", "same_kernel", dtype,
                        baseline, plain, greedy=int(plain[token].argmax()) == top_id)]
        operations, examples, effects = [], [], []
        heads = {}
        for layer in layer_list:
            module, kind = _mixer(model, layer)
            head = _auto_head(model, module, kind, ids, mask, token, unembed, selected) \
                if head_setting == "auto" else int(head_setting)
            heads[layer] = head
            rec = Recorder(kind, module, layer, token, head, "prefill", lines, source)
            with instrument(module, kind, rec), \
                 Capture(model, [(layer, "mixer_output"), (layer, "residual_output")], positions="all",
                         metadata=metadata, max_bytes=budget) as observed:
                changed = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[0].float().cpu()
            operations.extend(rec.operations)
            greedy = int(changed[token].argmax()) == top_id
            parity.append(check("prefill", "logits", "instrumented_vs_untouched", "same_kernel", dtype,
                                changed, baseline, greedy=greedy, layer=layer, mixer=kind))
            for site in ("mixer_output", "residual_output"):
                parity.append(check("prefill", site, "instrumented_vs_untouched", "same_kernel", dtype,
                                    observed.tensor(layer, site), full_cap.tensor(layer, site), layer=layer, mixer=kind))
            # Cached one-token decoding: untouched and instrumented prefill + decode of the same forced token.
            switch = _Switch()
            decode_rec = Recorder(kind, module, layer, 0, head, "decode", lines, source, position_offset=length)
            def cached(adapted):
                context = instrument(module, kind, switch) if adapted else nullcontext()
                with context:
                    switch.target = None
                    prefill = model(input_ids=ids, attention_mask=mask, use_cache=True)
                    next_id = torch.tensor([[top_id]], device=ids.device)
                    switch.target = decode_rec if adapted else None
                    step = model(input_ids=next_id, attention_mask=torch.cat([mask, mask[:, :1]], dim=1),
                                 past_key_values=prefill.past_key_values, use_cache=True)
                    switch.target = None
                    return prefill.logits[0, -1].float().cpu(), step.logits[0, -1].float().cpu()
            base_prefill, base_step = cached(False)
            inst_prefill, inst_step = cached(True)
            operations.extend(decode_rec.operations)
            parity.append(check("cached_prefill", "logits", "instrumented_vs_untouched", "same_kernel", dtype,
                                inst_prefill, base_prefill, greedy=int(inst_prefill.argmax()) == int(base_prefill.argmax()),
                                layer=layer, mixer=kind))
            parity.append(check("cached_decode", "logits", "instrumented_vs_untouched", "same_kernel", dtype,
                                inst_step, base_step, greedy=int(inst_step.argmax()) == int(base_step.argmax()),
                                layer=layer, mixer=kind, forced_token=selected[0]["token"]))
            # Left padding: both runs padded identically, so the same kernels must agree exactly.
            padded_ids, padded_mask = _left_pad(ids, mask, tokenizer, int(config.get("padding_tokens", 3)))
            pad_base = model(input_ids=padded_ids, attention_mask=padded_mask, use_cache=False).logits[0, -1].float().cpu()
            with instrument(module, kind, lambda *a, **k: None):
                pad_inst = model(input_ids=padded_ids, attention_mask=padded_mask, use_cache=False).logits[0, -1].float().cpu()
            parity.append(check("left_padding", "logits", "instrumented_vs_untouched", "same_kernel", dtype,
                                pad_inst, pad_base, greedy=int(pad_inst.argmax()) == int(pad_base.argmax()),
                                layer=layer, mixer=kind, padding_tokens=int(config.get("padding_tokens", 3))))
            if kind == "delta":
                parity.extend(cross_kernel_checks(model, ids, mask, layer, dtype, full_cap, baseline, token))
                parity.append(check("explanatory_recurrence", "core_readout", "explicit_recurrence_vs_chunk_kernel",
                                    "layer_cross_kernel", dtype, rec.held["readout"].transpose(0, 1).to(dtype),
                                    rec.held["kernel_readout"], layer=layer, mixer=kind))
                examples.append(delta_example(rec, module, unembed, token_list, tokens, selected))
            else:
                probs = rec.held["probabilities"]
                future = torch.triu(torch.ones_like(probs[0]), diagonal=1).bool()
                parity.append({"case": "causal_mask", "target": "attention_probabilities", "layer": layer,
                               "comparison": "probability_mass_on_future_keys", "dtype": str(dtype),
                               "max_future_probability": probs[:, future].max().item() if future.any() else 0.0,
                               "passed": bool((probs[:, future] == 0).all())})
                examples.append(attention_example(rec, module, unembed, token_list, tokens, selected))
            effects.append(residual_effects(model, cap, layer, token, unembed, selected, ids, mask, baseline))
        parity.extend(baseline_checks(model, tokenizer, ids, mask, dtype, baseline, token, config))
        parity.append({"case": "hook_cleanup", "target": "registered_hooks", "comparison": "before_vs_after_trace",
                       "hooks_before": hooks_before, "hooks_after": hook_count(model),
                       "instance_forward_overrides": sum("forward" in m.__dict__ for m in model.modules()),
                       "passed": hook_count(model) == hooks_before and not any("forward" in m.__dict__ for m in model.modules())})
        direct_total = unembed(final_residual, [s["id"] for s in selected])
        attribution_check = {"direct_logit_of_full_residual": direct_total,
                             "model_logits": [baseline[token, s["id"]].item() for s in selected],
                             "note": "Frozen-scale unembedding of the observed final residual; differences reflect BF16 rounding."}
    result = {"schema_version": 2, "kind": "trace",
              "metadata": {"run": run_name, "prompt": prompt, "formatted_prompt": formatted, "chat_template": bool(config.get("chat_template", True)),
                           "token_ids": token_list, "tokens": tokens, "selected_token_position": token,
                           "selected_token": tokens[token], "head_requested": head_setting, "heads": heads, "layers": layer_list,
                           "sites": names, "selected_next_tokens": selected, "dtype": str(dtype),
                           "model_revision": metadata["model_revision"], "provenance": runtime.provenance},
              "tolerances": TOLERANCES[str(dtype)], "parity": parity,
              "parity_summary": {"checks": len(parity), "passed": sum(p["passed"] for p in parity),
                                 "all_passed": all(p["passed"] for p in parity)},
              "captures": captures, "operations": operations, "worked_examples": examples,
              "residual_effects": effects, "attribution_check": attribution_check,
              "limitations": [
                  "The gated-delta explicit recurrence is an explanation computed beside the upstream chunked kernel; "
                  "its BF16 output is compared under the layer_cross_kernel tolerance, not required to be bit-identical.",
                  "Direct logit attribution freezes the final RMSNorm scale at the observed residual; it measures the direct "
                  "path only. Ablation effects include downstream processing at the selected position.",
                  "Operation slices are bounded to the selected batch row, head and token; statistics cover the full tensor.",
                  "Recurrent state is reported as a matrix, its norm trajectory and its read/write vectors, never as an attention map."]}
    write_json(output / "trace.json", result)
    return result


def _left_pad(ids, mask, tokenizer, count):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    padded_ids = torch.cat([torch.full((ids.shape[0], count), pad_id, device=ids.device, dtype=ids.dtype), ids], dim=1)
    padded_mask = torch.cat([torch.zeros((mask.shape[0], count), device=mask.device, dtype=mask.dtype), mask], dim=1)
    return padded_ids, padded_mask


def cross_kernel_checks(model, ids, mask, layer, dtype, full_cap, baseline, token):
    """Replace one layer's chunked kernel by the explicit recurrence and measure the propagated difference."""
    module, kind = _mixer(model, layer)
    with instrument(module, kind, lambda *a, **k: None, algorithm="recurrent", explain=False), \
         Capture(model, [(layer, "mixer_output"), (layer, "residual_output")], positions="all") as observed:
        recurrent = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[0].float().cpu()
    results = [check("recurrent_substitution", site, "explicit_recurrence_vs_chunk_kernel", "layer_cross_kernel", dtype,
                     observed.tensor(layer, site), full_cap.tensor(layer, site), layer=layer, mixer=kind)
               for site in ("mixer_output", "residual_output")]
    results.append(check("recurrent_substitution", "logits", "explicit_recurrence_vs_chunk_kernel", "logits_cross_kernel",
                         dtype, recurrent, baseline, greedy=int(recurrent[token].argmax()) == int(baseline[token].argmax()),
                         layer=layer, mixer=kind))
    return results


def _agreement(a, b, token):
    """Greedy equality at the selected position, plus the fraction of agreeing positions."""
    return {"greedy": int(a[token].argmax()) == int(b[token].argmax()),
            "argmax_agreement_fraction": (a.argmax(-1) == b.argmax(-1)).float().mean().item()}


def baseline_checks(model, tokenizer, ids, mask, dtype, baseline, token, config):
    """Properties of the untouched model: causality, padding and cached decoding.

    Greedy equality is required at the selected position (the prediction under
    study); agreement across all positions is reported, since near-tied
    positions can legitimately flip under equivalent kernels.
    """
    results = []
    length = ids.shape[1]
    if length > 1:
        prefix = model(input_ids=ids[:, :-1], attention_mask=mask[:, :-1], use_cache=False).logits[0].float().cpu()
        results.append(check("causal_prefix", "logits_before_last_token", "truncated_vs_full_sequence",
                             "logits_cross_kernel", dtype, prefix, baseline[:-1],
                             **_agreement(prefix, baseline[:-1], min(token, length - 2))))
        altered = ids.clone()
        altered[0, -1] = (altered[0, -1] + 1) % model.config.get_text_config().vocab_size
        changed = model(input_ids=altered, attention_mask=mask, use_cache=False).logits[0].float().cpu()
        results.append(check("causal_future_token", "logits_before_last_token", "last_token_replaced_vs_original",
                             "logits_cross_kernel", dtype, changed[:-1], baseline[:-1],
                             **_agreement(changed[:-1], baseline[:-1], min(token, length - 2))))
    for count in _as_list(config.get("padding_sweep", [1, 3, 8])):
        padded_ids, padded_mask = _left_pad(ids, mask, tokenizer, int(count))
        padded = model(input_ids=padded_ids, attention_mask=padded_mask, use_cache=False).logits[0, count:].float().cpu()
        results.append(check("left_padding_baseline", "logits", "padded_vs_unpadded_untouched", "logits_cross_kernel",
                             dtype, padded, baseline, **_agreement(padded, baseline, token),
                             padding_tokens=int(count)))
    prefill = model(input_ids=ids[:, :-1], attention_mask=mask[:, :-1], use_cache=True)
    step = model(input_ids=ids[:, -1:], attention_mask=mask, past_key_values=prefill.past_key_values, use_cache=True)
    results.append(check("cached_vs_uncached_baseline", "last_logits", "cached_decode_vs_full_prefill",
                         "logits_cross_kernel", dtype, step.logits[0, -1].float().cpu(), baseline[-1],
                         greedy=int(step.logits[0, -1].argmax()) == int(baseline[-1].argmax())))
    return results


def residual_effects(model, cap, layer, token, unembed, selected, ids, mask, baseline):
    """Residual decomposition at the selected token, direct logit attribution and zero-ablation effects."""
    def at(site):
        index = [i for i, r in enumerate(cap.records) if r["layer"] == layer and r["site"] == site]
        if not index:
            return None
        record = cap.records[index[0]]
        return record["tensor"][0, record["input_positions"].index(token)].float()
    parts = {site: at(site) for site in SITE_NAMES}
    ids_selected = [s["id"] for s in selected]
    result = {"layer": layer, "token_position": token, "selected_tokens": selected, "components": {}}
    for site, vector in parts.items():
        if vector is None:
            continue
        dla = unembed(vector, ids_selected)
        entry = {"norm": vector.norm().item(), "first4": vector[:4].tolist(),
                 "direct_logit": {s["token"]: v for s, v in zip(selected, dla)}}
        if len(dla) >= 2:
            entry["direct_logit_difference"] = dla[0] - dla[1]
        result["components"][site] = entry
    if all(parts[s] is not None for s in SITE_NAMES):
        dtype = cap.records[0]["tensor"].dtype
        r, m, p, f, o = (parts[s].to(dtype) for s in SITE_NAMES)
        result["identities"] = {
            "post_mixer_residual = residual_input + mixer_output (max_abs_error)": ((r + m).float() - p.float()).abs().max().item(),
            "residual_output = post_mixer_residual + mlp_output (max_abs_error)": ((p + f).float() - o.float()).abs().max().item(),
            "cosine(mixer_output, residual_input)": F.cosine_similarity(parts["mixer_output"], parts["residual_input"], dim=0).item(),
            "cosine(mlp_output, post_mixer_residual)": F.cosine_similarity(parts["mlp_output"], parts["post_mixer_residual"], dim=0).item()}
    ablations = {}
    for site in ("mixer_output", "mlp_output"):
        hidden = model.config.get_text_config().hidden_size
        with Intervention(model, layer, site, [token], patch=torch.zeros(1, 1, hidden)) as intervention:
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[0, token].float().cpu()
        delta = [(logits[i] - baseline[token, i]).item() for i in ids_selected]
        entry = {"hook_calls": intervention.calls, "logit_change": {s["token"]: d for s, d in zip(selected, delta)},
                 "new_top1": int(logits.argmax()), "top1_unchanged": int(logits.argmax()) == ids_selected[0]}
        if len(delta) >= 2:
            entry["logit_difference_change"] = delta[0] - delta[1]
        ablations[f"zero_{site}"] = entry
    result["ablations_at_selected_token"] = ablations
    return result


def calibrate(config, dtype_name, layers, output_entries):
    """Repeat cross-kernel and untouched-baseline measurements in another dtype (e.g. FP32)."""
    cfg = {**config, "dtype": dtype_name}
    runtime = load_model(cfg)
    try:
        model, tokenizer, dtype = runtime.model, runtime.tokenizer, runtime.dtype
        _, formatted = format_prompt(tokenizer, cfg)
        inputs = tokenizer(formatted, return_tensors="pt").to(runtime.device)
        ids, mask = inputs.input_ids, inputs.attention_mask
        token = [int(x) for x in _as_list(cfg.get("token", [-1]))][0] % ids.shape[1]
        with torch.inference_mode():
            delta_layers = [l for l in layers if _mixer(model, l)[1] == "delta"]
            with Capture(model, [(l, s) for l in delta_layers for s in ("mixer_output", "residual_output")],
                         positions="all", max_bytes=64 * 1024 * 1024) as full_cap:
                baseline = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[0].float().cpu()
            for layer in delta_layers:
                output_entries.extend(cross_kernel_checks(model, ids, mask, layer, dtype, full_cap, baseline, token))
            output_entries.extend(baseline_checks(model, tokenizer, ids, mask, dtype, baseline, token, cfg))
    finally:
        del runtime
        gc.collect()
        torch.cuda.empty_cache()
    return output_entries


def run_trace_command(config, output, overrides=None):
    """CLI entry: main trace in the configured dtype, then optional calibration dtypes."""
    cfg = read_config(config)
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    runtime = load_model(cfg)
    result = run_trace(runtime, cfg, output)
    del runtime
    gc.collect()
    torch.cuda.empty_cache()
    calibration = []
    for dtype_name in _as_list(cfg.get("calibration_dtypes", [])):
        calibrate(cfg, dtype_name, result["metadata"]["layers"], calibration)
    if calibration:
        result["calibration"] = {"dtypes": _as_list(cfg["calibration_dtypes"]),
                                 "tolerances": {f"torch.{d}": TOLERANCES[f"torch.{d}"] for d in _as_list(cfg["calibration_dtypes"])},
                                 "checks": calibration, "all_passed": all(c["passed"] for c in calibration)}
        write_json(Path(output) / "trace.json", result)
    return result
