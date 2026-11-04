"""Reproducible, bounded numerical traces of Qwen3.5 mixers."""
from pathlib import Path
from contextlib import nullcontext
import torch
from .runtime import write_json
from .sites import resolve_site
from .capture import Capture
from .adapters.qwen35 import instrument

OPERATION_INPUTS = {
    "query_projection": ["hidden_states", "q_proj.weight"], "key_projection": ["hidden_states", "k_proj.weight"],
    "value_projection": ["hidden_states", "v_proj.weight"], "query_normalized": ["query_projection", "q_norm.weight"],
    "key_normalized": ["key_projection_or_convolution", "normalization"],
    "query_rotary": ["query_normalized", "rotary_cos", "rotary_sin"],
    "key_rotary": ["key_normalized", "rotary_cos", "rotary_sin"],
    "key_grouped": ["key_rotary", "kv_cache"], "value_grouped": ["value", "kv_cache"],
    "scores": ["query_rotary", "key_grouped", "scale"], "masked_scores": ["scores", "mask"],
    "probabilities": ["masked_scores"], "value_aggregation": ["probabilities", "value_grouped"],
    "gated_output": ["value_aggregation_or_readout", "gate"], "projection": ["gated_output", "output_weight"],
    "pre_convolution_qkv": ["hidden_states", "in_proj_qkv.weight"],
    "post_convolution_qkv": ["pre_convolution_qkv", "conv1d.weight", "conv_cache"],
    "beta": ["hidden_states", "in_proj_b.weight"], "log_decay": ["hidden_states", "in_proj_a.weight", "A_log", "dt_bias"],
    "query_normalized_scaled": ["post_convolution_qkv"], "readout": ["state", "query_normalized_scaled"],
}


def _error(a, b):
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    return {"max_abs_error": diff.max().item(), "mean_abs_error": diff.mean().item(),
            "max_rel_error": (diff / a.abs().clamp_min(1e-6)).max().item()}


def run_trace(runtime, config, output):
    """Write trace.json and selected tensor copies. Parity results are measurements."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model, tokenizer = runtime.model, runtime.tokenizer
    prompt = config.get("prompt", "The red key opens the blue box.")
    formatted = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(formatted, return_tensors="pt").to(runtime.device)
    ids, mask = inputs.input_ids, inputs.attention_mask
    layer_list = config.get("layer", [0, 3]); layer_list = layer_list if isinstance(layer_list, list) else [layer_list]
    positions = config.get("token", [-1]); positions = positions if isinstance(positions, list) else [positions]
    names = config.get("site", ["residual_input", "mixer_output", "post_mixer_residual", "mlp_output", "residual_output"])
    names = names if isinstance(names, list) else [names]
    head = int(config.get("head", 0)); captures = []; operations = []; parity = []
    source = runtime.provenance["transformers_immutable_source_url"]
    with torch.inference_mode():
        with Capture(model, [(layer, name) for layer in layer_list for name in names], positions=positions,
                     max_bytes=int(config.get("capture_max_bytes", 8 * 1024 * 1024))) as cap:
            baseline = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.detach().float().cpu()
        reference_layers = {record["layer"]: record["tensor"] for record in cap.records if record["site"] == "residual_output"}
        for index, record in enumerate(cap.records):
            tensor = record.pop("tensor")
            filename = f"capture_{index:03d}.pt"; torch.save(tensor, output / filename)
            captures.append({**record, "tensor_path": filename, "token_ids": ids[0].tolist(),
                             "phase": "prefill", "source_url": source + "#L848"})
        for layer in layer_list:
            site = resolve_site(model, layer, "mixer_output")
            kind = "attention" if site.path.endswith("self_attn") else "delta"
            def recorder(name, tensor):
                axes = None
                if tensor.ndim == 4:
                    if name.startswith("state_after_token_"):
                        axes = ["batch", "head", "key_feature", "value_feature"]
                    elif name in ("query_normalized", "key_normalized", "query_rotary", "key_rotary", "value", "key_grouped", "value_grouped", "scores", "masked_scores", "probabilities", "query_normalized_scaled", "readout"):
                        axes = ["batch", "head", "token", "feature_or_key_token"]
                    else:
                        axes = ["batch", "token", "head", "feature"]
                elif tensor.ndim == 3:
                    axes = ["batch", "channel", "token"] if name == "pre_convolution_qkv" else ["batch", "token", "feature"]
                elif tensor.ndim == 2:
                    axes = ["batch", "feature"]
                view = tensor.detach()
                if axes and "batch" in axes:
                    view = view.narrow(axes.index("batch"), 0, 1)
                if axes and "head" in axes:
                    axis = axes.index("head")
                    if head >= view.shape[axis] or head < 0:
                        raise IndexError(f"Head {head} outside {name} shape {tuple(tensor.shape)}")
                    view = view.narrow(axis, head, 1)
                if axes and "token" in axes:
                    axis = axes.index("token"); token = positions[0] % view.shape[axis]
                    view = view.narrow(axis, token, 1)
                sample = view.flatten()[:8].float().cpu()
                operations.append({"layer": layer, "mixer": kind, "name": name,
                                   "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                                   "axes": axes, "inputs": OPERATION_INPUTS.get(name, []),
                                   "selected_head": head if axes and "head" in axes else None,
                                   "selected_token": positions[0] if axes and "token" in axes else None,
                                   "slice": sample.tolist(), "source_url": source + ("#L763" if kind == "attention" else "#L536")})
            with instrument(site.module, kind, recorder):
                with Capture(model, [(layer, "residual_output")], positions=positions) as observed:
                    changed = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.detach().float().cpu()
            reference = reference_layers.get(layer)
            parity.append({"case": "prefill", "layer": layer, "target": "logits", **_error(baseline, changed)})
            if reference is not None:
                parity.append({"case": "prefill", "layer": layer, "target": "layer_output", **_error(reference, observed.records[0]["tensor"])})
            def run_cached(adapted, forced_token=None):
                context = instrument(site.module, kind, lambda name, tensor: None) if adapted else nullcontext()
                with context:
                    prefill = model(input_ids=ids, attention_mask=mask, use_cache=True)
                    predicted_id = prefill.logits[:, -1:].argmax(dim=-1)
                    next_id = predicted_id if forced_token is None else forced_token.to(ids.device)
                    step = model(input_ids=next_id, attention_mask=torch.cat([mask, mask[:, :1]], dim=1),
                                 past_key_values=prefill.past_key_values, use_cache=True)
                    return predicted_id.cpu(), step.logits[:, -1].float().cpu()
            baseline_token, cached_baseline = run_cached(False)
            adapted_token, cached_changed = run_cached(True, baseline_token)
            parity.append({"case": "cached_decode", "layer": layer, "target": "logits",
                           "same_greedy_prefill_prediction": torch.equal(baseline_token, adapted_token),
                           **_error(cached_baseline, cached_changed)})
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            padded_ids = torch.cat([torch.full_like(ids[:, :1], pad_id), ids], dim=1)
            padded_mask = torch.cat([torch.zeros_like(mask[:, :1]), mask], dim=1)
            pad_baseline = model(input_ids=padded_ids, attention_mask=padded_mask, use_cache=False).logits[:, -1].float().cpu()
            with instrument(site.module, kind, lambda name, tensor: None):
                pad_changed = model(input_ids=padded_ids, attention_mask=padded_mask, use_cache=False).logits[:, -1].float().cpu()
            parity.append({"case": "left_padding", "layer": layer, "target": "logits", **_error(pad_baseline, pad_changed)})
        if ids.shape[1] > 1:
            prefix = model(input_ids=ids[:, :-1], attention_mask=mask[:, :-1], use_cache=False).logits[:, -1].float().cpu()
            parity.append({"case": "causal_prefix", "target": "baseline_logits", **_error(baseline[:, -2], prefix)})
    result = {"schema_version": 1, "kind": "trace", "metadata": {"prompt": prompt, "formatted_prompt": formatted,
              "token_ids": ids[0].tolist(), "tokens": tokenizer.convert_ids_to_tokens(ids[0].tolist()),
              "head_requested": head, "provenance": runtime.provenance}, "captures": captures,
              "operations": operations, "parity": parity,
              "limitations": ["Parity errors are measurements and are not yet calibrated into dtype-specific pass criteria.",
                              "Operation slices are bounded samples; some intermediate operations lack individual input shapes and source line mapping."]}
    write_json(output / "trace.json", result)
    return result
