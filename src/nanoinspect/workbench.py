"""Interactive analyses behind the localhost workbench (``nanoinspect serve``).

Each analysis takes a loaded Runtime plus JSON-friendly arguments and returns a
JSON-friendly result. They reuse the package's semantic sites, scoped
Capture/Intervention hooks and mixer adapters, so no hook outlives a call.
Logit-lens readouts apply the model's final RMSNorm and tied unembedding to an
intermediate residual; they show what that residual would predict if it were
final, not what a layer "knows". Gated-delta layers are shown as gates and
recurrent-state trajectories, never as softmax attention maps.
"""
from __future__ import annotations

import json
from pathlib import Path
import torch
from .adapters.qwen35 import instrument
from .capture import Capture
from .interventions import Intervention, contrastive_direction, hook_count, random_direction
from .runtime import ROOT
from .sites import decoder_layers, resolve_site

MAX_TOKENS = 256
MAX_PATCH_TOKENS = 96
LENS_SITES = ("residual_output", "mixer_output", "mlp_output")


def layer_types(model):
    _, layers = decoder_layers(model)
    return [getattr(block, "layer_type", getattr(block, "block_type", "unknown")) for block in layers]


def _text_model(model):
    path, _ = decoder_layers(model)
    return model.get_submodule(path.rsplit(".", 1)[0])


def encode(runtime, prompt, chat_template=False, limit=MAX_TOKENS):
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is empty")
    text = runtime.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False) \
        if chat_template else prompt
    ids = runtime.tokenizer(text, return_tensors="pt").input_ids
    if ids.shape[1] > limit:
        raise ValueError(f"Prompt has {ids.shape[1]} tokens; this view is limited to {limit}")
    return ids.to(runtime.device)


def token_strings(tokenizer, ids):
    return [tokenizer.decode([int(i)]) for i in ids]


def token_id(tokenizer, text):
    """First token of ``text`` (keep a leading space for word-initial tokens)."""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"{text!r} encodes to no tokens")
    return ids[0]


def _position(position, length):
    index = position if position >= 0 else length + position
    if not 0 <= index < length:
        raise IndexError(f"Position {position} outside sequence of {length} tokens")
    return index


def _unembed(model, residual):
    """Final RMSNorm + tied unembedding on ``[n, hidden]`` residual vectors (FP32 logits)."""
    norm = _text_model(model).norm
    weight = model.get_output_embeddings().weight
    with torch.inference_mode():
        return (norm(residual.to(weight.device, weight.dtype)) @ weight.T).float()


def _top(tokenizer, logits, k):
    probs = logits.softmax(-1)
    values, ids = probs.topk(k)
    return [{"token": tokenizer.decode([int(i)]), "id": int(i), "prob": round(float(v), 5)}
            for v, i in zip(values.tolist(), ids.tolist())]


def _round(rows, digits=4):
    return [[round(float(x), digits) for x in row] for row in rows]


def lens(runtime, prompt, chat_template=False, top_k=5, target=None):
    """Logit lens over every (layer, position), plus per-layer, per-token component norms.

    Row 0 reads the token embedding; row ``l + 1`` reads the residual stream
    after decoder layer ``l``. The last row equals the model's own prediction.
    """
    model, tokenizer = runtime.model, runtime.tokenizer
    ids = encode(runtime, prompt, chat_template)
    length = ids.shape[1]
    types = layer_types(model)
    selections = [(layer, site) for layer in range(len(types)) for site in LENS_SITES]
    selections.append((0, "residual_input"))
    with torch.inference_mode(), Capture(model, selections, positions="all", max_bytes=512 << 20) as cap:
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    norms = {site: _round([cap.tensor(layer, site)[0].float().norm(dim=-1).tolist() for layer in range(len(types))], 3)
             for site in LENS_SITES}
    target_id = token_id(tokenizer, target) if target else int(logits.argmax())
    decoded = {}
    decode = lambda i: decoded.setdefault(i, tokenizer.decode([i]))
    residuals = [cap.tensor(0, "residual_input")[0]] + [cap.tensor(layer, "residual_output")[0] for layer in range(len(types))]
    rows = []
    for row, residual in enumerate(residuals):
        cells = []
        for start in range(0, length, 64):  # bound the [positions, vocabulary] logits held at once
            chunk = _unembed(model, residual[start:start + 64])
            logprobs = chunk.log_softmax(-1)
            values, top = logprobs.topk(top_k, dim=-1)
            probs, top = values.exp().tolist(), top.tolist()
            target_probs = logprobs[:, target_id].exp().tolist()
            ranks = ((chunk > chunk[:, target_id:target_id + 1]).sum(-1) + 1).tolist()
            for i in range(len(top)):
                cells.append({"top": [[decode(t), round(v, 4)] for v, t in zip(probs[i], top[i])],
                              "target_prob": round(target_probs[i], 5), "target_rank": ranks[i]})
        rows.append({"layer": "embed" if row == 0 else row - 1, "type": "embedding" if row == 0 else types[row - 1],
                     "cells": cells})
    return {"tokens": token_strings(tokenizer, ids[0]), "layer_types": types,
            "target": {"token": tokenizer.decode([target_id]), "id": target_id, "chosen": "given" if target else "model top-1"},
            "model_top": _top(tokenizer, logits, top_k), "lens": rows, "norms": norms}


def walkthrough(runtime, prompt, layer, position=-1, head="auto", compare=None, chat_template=False):
    """Every recorded operation of one mixer for one token and head, with a hand-recomputed worked example.

    Uses the same Recorder and worked-example code as ``nanoinspect trace``,
    without its parity sweep. ``head="auto"`` picks the head whose output most
    raises the top-1 token over ``compare`` (or, without ``compare``, the head
    with the largest output).
    """
    from .trace import Recorder, Unembed, _auto_head, _final_residual, _mixer, attention_example, delta_example, source_lines
    model, tokenizer = runtime.model, runtime.tokenizer
    ids = encode(runtime, prompt, chat_template, 128)
    token = _position(int(position), ids.shape[1])
    layer = int(layer)
    module, kind = _mixer(model, layer)
    tokens = token_strings(tokenizer, ids[0])
    source = runtime.provenance.get("transformers_immutable_source_url") or \
        "https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py"
    with torch.inference_mode():
        handle, final = _final_residual(model)
        try:
            logits = model(input_ids=ids, use_cache=False).logits[0, token].float()
        finally:
            handle.remove()
        top_id = int(logits.argmax())
        selected = [{"token": tokenizer.decode([top_id]), "id": top_id, "role": "top-1 prediction"}]
        if compare:
            compare_id = token_id(tokenizer, compare)
            if compare_id != top_id:
                selected.append({"token": tokenizer.decode([compare_id]), "id": compare_id, "role": "comparison"})
        unembed = Unembed(model, final["value"][0, token])
        heads = module.config.num_attention_heads if kind == "attention" else module.num_v_heads
        automatic = head in ("auto", None, "")
        head = _auto_head(model, module, kind, ids, None, token, unembed, selected) if automatic else int(head)
        if not 0 <= head < heads:
            raise IndexError(f"Head {head} outside [0, {heads})")
        recorder = Recorder(kind, module, layer, token, head, "prefill", source_lines(), source)
        with instrument(module, kind, recorder):
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
        example = (attention_example if kind == "attention" else delta_example)(recorder, module, unembed, ids[0].tolist(),
                                                                                 tokens, selected)
    if hook_count(model) or "forward" in module.__dict__:
        raise RuntimeError("Instrumentation remained attached after the walkthrough")
    return {"layer": layer, "kind": "full_attention" if kind == "attention" else "linear_attention", "head": head,
            "heads": heads, "head_chosen": "auto" if automatic else "given", "position": token,
            "tokens": tokens, "selected": selected, "operations": recorder.operations, "example": example,
            "source_url": source}


def mixer(runtime, prompt, layer, chat_template=False):
    """Softmax probabilities of a full-attention layer, or gates and state norms of a gated-delta layer."""
    model, tokenizer = runtime.model, runtime.tokenizer
    ids = encode(runtime, prompt, chat_template)
    layer = int(layer)
    site = resolve_site(model, layer, "mixer_output")
    kind = "attention" if site.path.endswith("self_attn") else "delta"
    store = {"state": [], "delta": []}

    def record(name, tensor, inputs=(), token=None):
        if kind == "attention" and name == "probabilities":
            store["probs"] = tensor[0].float().cpu()
        elif kind == "delta":
            if name == "beta":
                store["beta"] = tensor[0].float().cpu()
            elif name == "log_decay":
                store["decay"] = tensor[0].float().exp().cpu()
            elif name == "state_after":
                store["state"].append(tensor[0].float().flatten(1).norm(dim=-1).cpu())
            elif name == "state_delta":
                store["delta"].append(tensor[0].float().norm(dim=-1).cpu())

    with torch.inference_mode(), instrument(site.module, kind, record, explain=True):
        model(input_ids=ids, use_cache=False)
    result = {"layer": layer, "kind": "full_attention" if kind == "attention" else "linear_attention",
              "tokens": token_strings(tokenizer, ids[0])}
    if kind == "attention":
        config = site.module.config
        result.update(heads=int(store["probs"].shape[0]), kv_heads=int(config.num_key_value_heads),
                      probabilities=[_round(head.tolist()) for head in store["probs"]])
    else:
        # [token, head] series; "decay" is exp(g), the fraction of state kept per token.
        result.update(heads=int(store["beta"].shape[-1]), beta=_round(store["beta"].tolist()),
                      decay=_round(store["decay"].tolist(), 5),
                      state_norm=_round(torch.stack(store["state"]).tolist(), 4),
                      write_norm=_round(torch.stack(store["delta"]).tolist(), 4))
    return result


def _logit_difference(logits, target, foil):
    return logits[..., target] - logits[..., foil]


def patch(runtime, clean, corrupted, sites=("residual_output",), target=None, foil=None, chat_template=False):
    """Patch one clean activation per (site, layer, position) into the corrupted run.

    Restoration = (LD_patched - LD_corrupted) / (LD_clean - LD_corrupted) with
    LD = logit(target) - logit(foil) at the final position: 0 means no effect,
    1 means the clean margin is fully restored.
    """
    model, tokenizer = runtime.model, runtime.tokenizer
    sites = [sites] if isinstance(sites, str) else list(sites)
    if not sites:
        raise ValueError("Select at least one site")
    clean_ids = encode(runtime, clean, chat_template, MAX_PATCH_TOKENS)
    corrupt_ids = encode(runtime, corrupted, chat_template, MAX_PATCH_TOKENS)
    if clean_ids.shape != corrupt_ids.shape:
        raise ValueError(f"Clean and corrupted prompts must have equal token counts "
                         f"({clean_ids.shape[1]} vs {corrupt_ids.shape[1]}); adjust a word so both tokenize to the same length")
    types = layer_types(model)
    length = clean_ids.shape[1]
    selections = [(layer, site) for site in sites for layer in range(len(types))]
    with torch.inference_mode(), Capture(model, selections, positions="all", max_bytes=1 << 30) as cap:
        clean_logits = model(input_ids=clean_ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    with torch.inference_mode():
        corrupt_logits = model(input_ids=corrupt_ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    target_id = token_id(tokenizer, target) if target else int(clean_logits.argmax())
    foil_id = token_id(tokenizer, foil) if foil else int(corrupt_logits.argmax())
    if target_id == foil_id:
        raise ValueError("Target and foil are the same token; give them explicitly")
    ld_clean = float(_logit_difference(clean_logits, target_id, foil_id))
    ld_corrupt = float(_logit_difference(corrupt_logits, target_id, foil_id))
    span = ld_clean - ld_corrupt
    if abs(span) < 1e-3:
        raise ValueError("Clean and corrupted runs give the same logit difference; nothing to restore")
    grids = {}
    for site in sites:
        grid, raw = [], []
        for layer in range(len(types)):
            activations = cap.tensor(layer, site)[0]  # [token, hidden]
            values = []
            for start in range(0, length, 32):
                rows = list(range(start, min(start + 32, length)))
                batch = corrupt_ids.expand(len(rows), -1)
                replacement = activations[rows].unsqueeze(1)
                with torch.inference_mode(), Intervention(model, layer, site, [[p] for p in rows], patch=replacement):
                    logits = model(input_ids=batch, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                values += _logit_difference(logits, target_id, foil_id).tolist()
            raw.append([round(v, 4) for v in values])
            grid.append([round((v - ld_corrupt) / span, 4) for v in values])
        grids[site] = {"restoration": grid, "logit_difference": raw}
    if hook_count(model):
        raise RuntimeError("Hooks remained registered after patching")
    differing = [i for i in range(length) if clean_ids[0, i] != corrupt_ids[0, i]]
    probs = lambda logits: {"target": round(float(logits.softmax(-1)[target_id]), 4),
                            "foil": round(float(logits.softmax(-1)[foil_id]), 4)}
    return {"sites": sites, "layer_types": types, "tokens": token_strings(tokenizer, corrupt_ids[0]),
            "clean_tokens": token_strings(tokenizer, clean_ids[0]), "differing_positions": differing,
            "target": {"token": tokenizer.decode([target_id]), "id": target_id},
            "foil": {"token": tokenizer.decode([foil_id]), "id": foil_id},
            "clean_probs": probs(clean_logits), "corrupted_probs": probs(corrupt_logits),
            "ld_clean": round(ld_clean, 4), "ld_corrupted": round(ld_corrupt, 4), "grids": grids}


def steering_assets(run=ROOT / "runs/experiment"):
    """Directions and per-layer scales saved by Study C, or ``None`` if that run is absent."""
    run = Path(run)
    path = run / "details/steering_directions.pt"
    if not path.is_file():
        return None
    saved = torch.load(path, weights_only=True)
    scales = {}
    experiment = run / "experiment.json"
    if experiment.is_file():
        conventions = json.loads(experiment.read_text())["studies"].get("steering", {}).get("conventions", {})
        scales = {int(layer): value["mean_residual_norm"] for layer, value in conventions.items()}
    return {"directions": {int(k): v for k, v in saved["directions"].items()}, "random": saved["random_directions"],
            "selected_layer": saved["selected_layer"], "site": saved["site"], "scales": scales}


def _final_residuals(runtime, prompts, layer):
    out = []
    for prompt in prompts:
        ids = encode(runtime, prompt)
        with torch.inference_mode(), Capture(runtime.model, [(layer, "residual_output")], positions=[-1]) as cap:
            runtime.model(input_ids=ids, use_cache=False)
        out.append(cap.tensor(layer, "residual_output")[0, 0])
    return torch.stack(out)


def _direction(runtime, direction, layer, positive=None, negative=None):
    """Unit steering direction and the raw norm of its unnormalized contrast (when one was computed)."""
    assets = steering_assets()
    width = runtime.model.get_output_embeddings().weight.shape[1]
    if direction == "custom":
        positive = [p for p in (positive or []) if p.strip()]
        negative = [p for p in (negative or []) if p.strip()]
        if not positive or len(positive) != len(negative):
            raise ValueError("Custom directions need equally many positive and negative prompts, one per line")
        return contrastive_direction(_final_residuals(runtime, positive, layer), _final_residuals(runtime, negative, layer))
    if direction.startswith("random_seed_"):
        return random_direction(width, int(direction.rsplit("_", 1)[1])), None
    if assets is None:
        raise ValueError("Saved Study C directions are absent; use a custom or random direction")
    if direction == "contrastive":
        if layer not in assets["directions"]:
            raise ValueError(f"Study C saved its direction only for layers {sorted(assets['directions'])}; "
                             "pick one of those or use a custom direction")
        return assets["directions"][layer].float(), None
    if direction in assets["random"]:
        vector = assets["random"][direction].float()
        return vector / vector.norm(), None
    raise ValueError(f"Unknown direction {direction!r}")


def _unit(runtime, ids, layer):
    """Residual-norm unit for alpha: Study C's convention when recorded for this layer, else this prompt's."""
    assets = steering_assets()
    if layer in (assets or {}).get("scales", {}):
        return assets["scales"][layer], "Study C mean residual norm at this layer"
    with torch.inference_mode(), Capture(runtime.model, [(layer, "residual_output")], positions="all") as cap:
        runtime.model(input_ids=ids, use_cache=False)
    residual = cap.tensor(layer, "residual_output")[0, 1:].float()
    if residual.shape[0] == 0:
        residual = cap.tensor(layer, "residual_output")[0].float()
    return residual.norm(dim=-1).mean().item(), "mean residual norm of this prompt (positions >= 1)"


def _watch(tokenizer, words):
    return sorted({token_id(tokenizer, w) for w in (words or []) if w.strip()})


def steer(runtime, prompt, layer, alpha, direction="contrastive", steps=24, policy="decode_time",
          chat_template=False, positive=None, negative=None):
    """Greedy generation with and without h' = h + alpha * unit_value * v_hat at ``residual_output``.

    ``direction`` is ``contrastive`` (Study C's saved sentiment direction),
    ``random_<k>`` (Study C's matched random controls), ``random_seed_<n>`` (a
    fresh Gaussian direction) or ``custom`` (mean final-token difference of
    the given positive/negative prompt lists).
    """
    from .experiments.steering import _steer, _truncate, generate
    model, tokenizer = runtime.model, runtime.tokenizer
    layer, alpha, steps = int(layer), float(alpha), max(1, min(int(steps), 128))
    vector, raw_norm = _direction(runtime, direction, layer, positive, negative)
    ids = encode(runtime, prompt, chat_template)
    unit, unit_source = _unit(runtime, ids, layer)
    strength = alpha * unit
    factory = lambda: _steer(model, layer, "residual_output", vector, strength)
    with torch.inference_mode():
        base_logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        with factory():
            steered_logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        base = generate(model, ids, steps)[0].tolist()
        steered = generate(model, ids, steps, factory, policy)[0].tolist()
    if hook_count(model):
        raise RuntimeError("Hooks remained registered after steering")
    kl = torch.sum(base_logits.softmax(-1) * (base_logits.log_softmax(-1) - steered_logits.log_softmax(-1))).item()
    return {"layer": layer, "alpha": alpha, "strength": round(strength, 5), "unit_value": round(unit, 5),
            "unit_source": unit_source, "direction": direction, "policy": policy, "kl": round(kl, 5),
            "raw_direction_norm": raw_norm,
            "baseline": {"text": tokenizer.decode(_truncate(tokenizer, base)), "top": _top(tokenizer, base_logits, 8)},
            "steered": {"text": tokenizer.decode(_truncate(tokenizer, steered)), "top": _top(tokenizer, steered_logits, 8)}}


def steer_sweep(runtime, prompt, layer, alphas, direction="contrastive", control="random_seed_0", steps=16,
                policy="decode_time", chat_template=False, positive=None, negative=None,
                watch_positive=None, watch_negative=None):
    """Sweep alpha for a direction and a norm-matched control direction.

    Per alpha: the greedy continuation, the next-token KL divergence from the
    unsteered model, and the watch score log P(watch_positive) - log P(watch_negative)
    at the last prompt position (each set's probabilities summed).
    """
    from .experiments.steering import _steer, _truncate, generate
    model, tokenizer = runtime.model, runtime.tokenizer
    layer, steps = int(layer), max(1, min(int(steps), 64))
    alphas = sorted({round(float(a), 6) for a in alphas} | {0.0})
    if len(alphas) > 15:
        raise ValueError("At most 15 alpha values per sweep")
    ids = encode(runtime, prompt, chat_template)
    unit, unit_source = _unit(runtime, ids, layer)
    pos_ids, neg_ids = _watch(tokenizer, watch_positive), _watch(tokenizer, watch_negative)
    directions = {direction: _direction(runtime, direction, layer, positive, negative)[0]}
    if control and control != direction:
        directions[control] = _direction(runtime, control, layer)[0]
    with torch.inference_mode():
        base_logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()

    def measure(logits):
        logprobs = logits.log_softmax(-1)
        out = {"kl": round(torch.sum(base_logits.softmax(-1) * (base_logits.log_softmax(-1) - logprobs)).item(), 5),
               "top": _top(tokenizer, logits, 3)}
        if pos_ids and neg_ids:
            out["watch"] = round((logprobs[pos_ids].logsumexp(0) - logprobs[neg_ids].logsumexp(0)).item(), 4)
        return out

    series = {}
    for name, vector in directions.items():
        rows = []
        for alpha in alphas:
            factory = lambda a=alpha: _steer(model, layer, "residual_output", vector, a * unit)
            with torch.inference_mode():
                with factory():
                    logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                text = tokenizer.decode(_truncate(tokenizer, generate(model, ids, steps, factory, policy)[0].tolist()))
            rows.append({"alpha": alpha, "text": text, **measure(logits)})
        series[name] = rows
    if hook_count(model):
        raise RuntimeError("Hooks remained registered after steering")
    return {"layer": layer, "unit_value": round(unit, 5), "unit_source": unit_source, "policy": policy,
            "direction": direction, "control": control if control != direction else None,
            "watch": {"positive": [tokenizer.decode([i]) for i in pos_ids], "negative": [tokenizer.decode([i]) for i in neg_ids]},
            "series": series}


def architecture(inventory_path=ROOT / "runs/inventory/inventory.json", model_dir=None):
    """Architecture summary from the M1 inventory (or just the model config when no inventory exists)."""
    inventory_path = Path(inventory_path)
    doc = ROOT / "docs/architecture.md"
    result = {"notes": doc.read_text() if doc.is_file() else None, "inventory": inventory_path.is_file()}
    if not inventory_path.is_file():
        from .runtime import read_config
        config_path = Path(model_dir or ROOT / read_config()["model_dir"]) / "config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        text = config.get("text_config", config)
        result.update(config=text, layers=[{"index": i, "type": t} for i, t in enumerate(text.get("layer_types", []))])
        return result
    data = json.loads(inventory_path.read_text())
    text = data["config"]["text_config"]
    layers = []
    for layer in data["layers"]:
        prefix = layer["path"] + "."
        flow = [{"module": path[len(prefix):] if path.startswith(prefix) else path,
                 "input": [x.get("shape") for x in (io.get("input") or []) if isinstance(x, dict)],
                 "output": (io.get("output") or {}).get("shape") if isinstance(io.get("output"), dict) else None}
                for path, io in (layer.get("observed") or {}).items()]
        layers.append({"index": layer["index"], "type": layer["type"], "mixer_class": layer["mixer_class"],
                       "parameters": layer["parameters"], "weight_shapes": layer["weight_shapes"], "dataflow": flow})
    caches = []
    for measurement in data["cache_measurements"]:
        by_kind = {}
        for tensor in measurement["tensors"]:
            kind = "kv" if any(s in tensor["name"] for s in ("key", "value")) else \
                "recurrent" if "recurrent" in tensor["name"] else "conv" if "conv" in tensor["name"] else "other"
            by_kind[kind] = by_kind.get(kind, 0) + tensor["bytes"]
        caches.append({"sequence_length": measurement["sequence_length"],
                       "measured_storage_bytes": measurement["measured_storage_bytes"], "bytes_by_kind": by_kind})
    svgs = sorted(p.name for p in inventory_path.parent.glob("*.svg"))
    result.update(config=text, vision=data["config"].get("vision_config"), layers=layers,
                  accounting={k: v for k, v in data["accounting"].items() if k not in ("loading_info",)},
                  caches=caches, svgs=[f"/runs/{inventory_path.parent.name}/{name}" for name in svgs],
                  provenance={k: data["provenance"].get(k) for k in ("model_id", "download_revision",
                                                                      "transformers_source_revision", "packages")})
    return result


def runs(root=ROOT / "runs"):
    """Run directories with reports, their kind and figures."""
    out = []
    for path in sorted(Path(root).rglob("report.html")):
        run = path.parent
        kind = None
        for name in ("experiment.json", "trace.json", "inventory.json", "baseline.json"):
            if (run / name).is_file():
                kind = name.split(".")[0]
                break
        rel = run.relative_to(root).as_posix()
        out.append({"name": rel, "kind": kind, "report": f"/runs/{rel}/report.html",
                    "modified": path.stat().st_mtime,
                    "figures": [f"/runs/{rel}/{p.name}" for p in sorted(run.glob("*.png"))]})
    return out
