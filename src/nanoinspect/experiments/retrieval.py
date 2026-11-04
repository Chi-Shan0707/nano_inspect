"""Study A: key/value retrieval across distance and distractors. Study B: clean-to-corrupted patching."""
from __future__ import annotations

import random
import torch
from ..capture import Capture, capture_metadata
from ..interventions import Intervention, random_direction, hook_count
from ..inventory import cache_tensors, storage_key
from ..sites import decoder_layers
from .stats import summarize
from .tasks import retrieval_examples, corrupt


def batches(items, max_tokens, max_batch):
    """Group equal-length examples (no padding) into batches bounded by a token budget."""
    groups = {}
    for index, item in enumerate(items):
        groups.setdefault(len(item.ids), []).append(index)
    for length, indices in sorted(groups.items()):
        size = max(1, min(max_batch, max_tokens // length))
        for start in range(0, len(indices), size):
            yield indices[start:start + size]


def final_logits(model, rows, device):
    ids = torch.tensor(rows, device=device)
    return model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1].float()


def score_examples(model, tokenizer, examples, device, max_tokens, max_batch):
    """Per-example correctness, target log-probability/rank and margin over the best in-prompt distractor."""
    rows = [None] * len(examples)
    for batch in batches(examples, max_tokens, max_batch):
        logits = final_logits(model, [examples[i].ids for i in batch], device)
        logprobs = logits.log_softmax(-1)
        for j, i in enumerate(batch):
            example, values, lp = examples[i], logits[j], logprobs[j]
            target = example.target_id
            others = [v for v in example.value_ids if v != target]
            best = max(others, key=lambda v: values[v].item()) if others else None
            top = int(values.argmax())
            rows[i] = {**example.condition, "query_index": example.query, "n_tokens": len(example.ids),
                       "distance": example.distance, "target": example.target, "top1": tokenizer.decode([top]),
                       "correct": top == target, "top1_is_prompt_value": top in example.value_ids,
                       "target_logprob": lp[target].item(), "target_rank": int((values > values[target]).sum()),
                       "margin_vs_best_distractor": (values[target] - values[best]).item() if best is not None else None,
                       "best_distractor": tokenizer.decode([best]) if best is not None else None}
    return rows


def cache_bytes(model, ids, device):
    """Unique cached storage after a prefill, grouped by state type (batch size one)."""
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([ids], device=device), use_cache=True, logits_to_keep=1)
    totals, seen = {"attention_kv": 0, "recurrent_state": 0, "convolution_state": 0}, set()
    for name, tensor in cache_tensors(out.past_key_values):
        key = storage_key(tensor)
        if key in seen:
            continue
        seen.add(key)
        kind = "recurrent_state" if "recurrent" in name else "convolution_state" if "conv" in name else "attention_kv"
        totals[kind] += tensor.numel() * tensor.element_size()
    del out
    return {**totals, "sequence_length": len(ids)}


def study_a(runtime, config, details):
    """Baseline retrieval grid. Pilot accuracy alone decides which conditions are 'solvable'."""
    cfg = config["retrieval"]
    model, tokenizer, device = runtime.model, runtime.tokenizer, runtime.device
    seed, threshold = int(config.get("seed", 0)), float(cfg.get("solvable_accuracy", 0.9))
    budget, max_batch = int(config.get("max_batch_tokens", 12000)), int(config.get("max_batch", 64))
    conditions, per_example = [], []
    for fmt in cfg["formats"]:
        for n_facts in cfg["n_facts"]:
            for filler in cfg["filler_sentences"]:
                pilot = retrieval_examples(tokenizer, fmt, n_facts, filler, int(cfg["pilot_examples"]), seed, "pilot")
                heldout = retrieval_examples(tokenizer, fmt, n_facts, filler, int(cfg["heldout_examples"]), seed, "heldout")
                with torch.inference_mode():
                    pilot_rows = score_examples(model, tokenizer, pilot, device, budget, max_batch)
                    held_rows = score_examples(model, tokenizer, heldout, device, budget, max_batch)
                pilot_accuracy = sum(r["correct"] for r in pilot_rows) / len(pilot_rows)
                accuracy = summarize([float(r["correct"]) for r in held_rows], seed)
                margins = summarize([r["margin_vs_best_distractor"] for r in held_rows], seed)
                by_position = {}
                for row in held_rows:
                    third = min(2, 3 * row["query_index"] // n_facts)
                    by_position.setdefault(["early", "middle", "late"][third], []).append(float(row["correct"]))
                conditions.append({
                    "format": fmt, "n_facts": n_facts, "distractors": n_facts - 1, "filler_sentences": filler,
                    "n_tokens": len(heldout[0].ids), "mean_distance": sum(e.distance for e in heldout) / len(heldout),
                    "pilot_n": len(pilot_rows), "pilot_accuracy": pilot_accuracy, "solvable": pilot_accuracy >= threshold,
                    "heldout_accuracy": accuracy, "heldout_margin": margins,
                    "heldout_target_logprob": summarize([r["target_logprob"] for r in held_rows], seed),
                    "accuracy_by_query_third": {k: sum(v) / len(v) for k, v in by_position.items()},
                    "errors_to_prompt_values": sum((not r["correct"]) and r["top1_is_prompt_value"] for r in held_rows),
                    "cache": cache_bytes(model, heldout[0].ids, device)})
                per_example.extend({**r, "split": "pilot"} for r in pilot_rows)
                per_example.extend({**r, "split": "heldout"} for r in held_rows)
    failures = [r for r in per_example if not r["correct"] and r["split"] == "heldout"][:12]
    details["retrieval_examples.json"] = per_example
    return {"question": "How is earlier information retained across distance and distractors?",
            "solvable_threshold": threshold, "conditions": conditions, "failure_examples": failures,
            "per_example_artifact": "details/retrieval_examples.json"}


def _pairs(runtime, cfg, seed, split, count):
    """Baseline-solvable aligned clean/corrupted pairs; exclusions are counted, not hidden."""
    tokenizer = runtime.tokenizer
    rng = random.Random(f"{seed}:corrupt:{split}")
    pool = retrieval_examples(tokenizer, cfg["format"], int(cfg["n_facts"]), int(cfg["filler_sentences"]),
                              count * int(cfg.get("candidate_multiplier", 3)), seed, f"patch-{split}")
    clean, corrupted, excluded = [], [], {"alignment": 0, "clean_incorrect": 0, "corrupted_not_flipped": 0}
    for example in pool:
        other = corrupt(tokenizer, example, rng)
        if other is None:
            excluded["alignment"] += 1
            continue
        clean.append(example); corrupted.append(other)
    top_clean = _logits(runtime, clean).argmax(-1)
    top_corrupt = _logits(runtime, corrupted).argmax(-1)
    kept = []
    for i, (a, b) in enumerate(zip(clean, corrupted)):
        if int(top_clean[i]) != a.target_id:
            excluded["clean_incorrect"] += 1
        elif int(top_corrupt[i]) != b.target_id:
            excluded["corrupted_not_flipped"] += 1
        else:
            kept.append(i)
        if len(kept) == count:
            break
    if len(kept) < count:
        raise RuntimeError(f"Only {len(kept)} solvable {split} pairs; increase candidate_multiplier")
    return [clean[i] for i in kept], [corrupted[i] for i in kept], excluded, i + 1


def _logits(runtime, examples):
    """Final-position logits on CPU for examples of any length (batched by length)."""
    cfg = runtime.config
    rows = torch.zeros(len(examples), runtime.model.config.get_text_config().vocab_size)
    with torch.inference_mode():
        for batch in batches(examples, int(cfg.get("max_batch_tokens", 12000)), int(cfg.get("max_batch", 64))):
            rows[batch] = final_logits(runtime.model, [examples[i].ids for i in batch], runtime.device).cpu()
    return rows


def _difference(logits, clean, corrupted):
    """Target (clean answer) minus distractor (corrupted answer) logit, per example."""
    index = torch.arange(len(clean))
    a = torch.tensor([e.target_id for e in clean])
    b = torch.tensor([e.target_id for e in corrupted])
    return logits[index, a] - logits[index, b]


def study_b(runtime, config, details):
    """Restore clean activations into corrupted runs at every layer, semantic site and aligned position."""
    cfg = config["patching"]
    model, device = runtime.model, runtime.device
    seed = int(config.get("seed", 0))
    _, layers = decoder_layers(model)
    layer_types = [layer.block_type for layer in layers]
    sites, position_names = list(cfg["sites"]), list(cfg["positions"])
    hidden = model.config.get_text_config().hidden_size
    results = {"question": "Which layers, sites and positions mediate a retrieved answer?", "condition": {
        k: cfg[k] for k in ("format", "n_facts", "filler_sentences")}, "layer_types": layer_types,
        "sites": sites, "positions": position_names}
    splits = {}
    for split, count in (("pilot", int(cfg["pilot_examples"])), ("heldout", int(cfg["heldout_examples"]))):
        clean, corrupted, excluded, candidates = _pairs(runtime, cfg, seed, split, count)
        splits[split] = (clean, corrupted)
        results[f"{split}_selection"] = {"kept": len(clean), "candidates_examined": candidates, "excluded": excluded,
                                          "rule": "clean top-1 = clean answer and corrupted top-1 = corrupted answer"}
    clean, corrupted = splits["heldout"]
    pilot_clean, pilot_corrupted = splits["pilot"]
    pilot_ld = (_difference(_logits(runtime, pilot_clean), pilot_clean, pilot_corrupted),
                _difference(_logits(runtime, pilot_corrupted), pilot_clean, pilot_corrupted))
    results["pilot_behavior"] = {"clean_logit_difference": summarize(pilot_ld[0].tolist(), seed),
                                 "corrupted_logit_difference": summarize(pilot_ld[1].tolist(), seed)}
    n = len(clean)
    positions = {name: [e.positions[name] for e in clean] for name in position_names}
    batch_ids = torch.tensor([e.ids for e in clean], device=device)
    corrupt_ids = torch.tensor([e.ids for e in corrupted], device=device)
    if batch_ids.shape != corrupt_ids.shape:
        raise RuntimeError("Clean and corrupted batches must share one token length")
    alignment = [[i for i, (a, b) in enumerate(zip(x.ids, y.ids)) if a != b] == [x.positions["value"]]
                 for x, y in zip(clean, corrupted)]
    hooks_before = hook_count(model)
    patch_layers = [int(x) for x in cfg.get("layers", range(len(layers)))]
    control_layers = {0, len(layers) // 2, len(layers) - 1}
    selections = [(l, s) for l in sorted(set(patch_layers) | control_layers) for s in sites]
    per_row = [[e.positions[name] for name in position_names] for e in clean]
    budget = int(config.get("capture_max_bytes", 512 * 1024 * 1024))
    expected = {"batch_size": n, "hidden_size": hidden, "element_size": torch.tensor([], dtype=runtime.dtype).element_size()}

    rows_index = torch.arange(n, device=device)
    target_ids = torch.tensor([e.target_id for e in clean], device=device)
    distractor_ids = torch.tensor([e.target_id for e in corrupted], device=device)

    def forward(ids, full=False):
        """Final-position logits; by default only (clean answer, corrupted answer) columns."""
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        if full:
            return logits.cpu()
        return torch.stack([logits[rows_index, target_ids], logits[rows_index, distractor_ids]], -1).cpu()

    with torch.inference_mode():
        plain_clean = forward(batch_ids, full=True)
        meta = capture_metadata(runtime, "experiment", "prefill", [e.ids for e in clean])
        with Capture(model, selections, positions=per_row, metadata=meta, max_bytes=budget, expected=expected) as clean_cap:
            logits_clean = forward(batch_ids, full=True)
        meta = capture_metadata(runtime, "experiment", "prefill", [e.ids for e in corrupted])
        with Capture(model, selections, positions=per_row, metadata=meta, max_bytes=budget, expected=expected) as corrupt_cap:
            logits_corrupt = forward(corrupt_ids, full=True)
        clean_acts = {(r["layer"], r["site"]): r["tensor"] for r in clean_cap.records}
        corrupt_acts = {(r["layer"], r["site"]): r["tensor"] for r in corrupt_cap.records}
        capture_record_example = {k: v for k, v in clean_cap.records[0].items() if k not in ("tensor", "token_ids")}
        ld_clean = _difference(logits_clean, clean, corrupted)
        ld_corrupt = _difference(logits_corrupt, clean, corrupted)
        span = ld_clean - ld_corrupt
        if (span <= 0).any():
            raise RuntimeError("Every held-out pair must have a positive clean-minus-corrupted logit difference")

        def recovery(pair):
            return (pair[:, 0] - pair[:, 1] - ld_corrupt) / span

        recover = torch.zeros(len(position_names), len(sites), len(patch_layers), n)
        magnitude = torch.zeros(len(position_names), len(sites), len(patch_layers), n)
        for p, name in enumerate(position_names):
            rows = [[x] for x in positions[name]]
            for s, site in enumerate(sites):
                for li, layer in enumerate(patch_layers):
                    patch = clean_acts[(layer, site)][:, p:p + 1]
                    magnitude[p, s, li] = (patch.float() - corrupt_acts[(layer, site)][:, p:p + 1].float()).norm(dim=-1)[:, 0]
                    with Intervention(model, layer, site, rows, patch=patch):
                        recover[p, s, li] = recovery(forward(corrupt_ids))
        # Patch-magnitude control: same-norm random displacement instead of the clean activation.
        control_sites = [s for s in cfg.get("random_control_sites", ["residual_output", "mixer_output"]) if s in sites]
        control_positions = [p for p in cfg.get("random_control_positions", ["value", "final"]) if p in position_names]
        seeds = [int(x) for x in cfg.get("random_control_seeds", [0, 1, 2])]
        random_recover = torch.zeros(len(control_positions), len(control_sites), len(patch_layers), len(seeds), n)
        for pi, name in enumerate(control_positions):
            p = position_names.index(name)
            rows = [[x] for x in positions[name]]
            for si, site in enumerate(control_sites):
                s = sites.index(site)
                for li, layer in enumerate(patch_layers):
                    base = corrupt_acts[(layer, site)][:, p:p + 1].float()
                    for k, seed_value in enumerate(seeds):
                        noise = torch.stack([random_direction(hidden, seed_value * 100003 + layer * 1009 + row, 1.0)
                                             for row in range(n)])[:, None]
                        patch = base + noise * magnitude[p, s, li][:, None, None]
                        with Intervention(model, layer, site, rows, patch=patch):
                            random_recover[pi, si, li, k] = recovery(forward(corrupt_ids))
        controls = _controls(runtime, clean, corrupted, clean_acts, corrupt_acts, logits_corrupt, plain_clean,
                             logits_clean, forward, corrupt_ids, positions, sites, position_names, len(layers), hidden)
        repeat = forward(corrupt_ids, full=True)
        controls.append({"control": "repeat_after_sweep", "description": "Corrupted baseline rerun after all patches",
                         "max_abs_logit_difference": (repeat - logits_corrupt).abs().max().item(),
                         "passed": torch.equal(repeat, logits_corrupt)})
        controls.append({"control": "hook_cleanup", "description": "Registered hooks before vs after the study",
                         "hooks_before": hooks_before, "hooks_after": hook_count(model),
                         "passed": hook_count(model) == hooks_before})
        if "post_mixer_residual" in sites and "residual_output" in sites:
            gap = (recover[:, sites.index("post_mixer_residual")] - recover[:, sites.index("residual_output")]).abs().max().item()
            controls.append({"control": "position_wise_mlp_equivalence",
                             "description": "Clean post-mixer residual patch must reproduce the clean block output at that "
                                            "position (the MLP is position-wise), so its recovery equals residual_output's",
                             "max_abs_recovery_difference": gap, "passed": gap == 0.0})
        controls.append({"control": "token_alignment", "description": "Clean/corrupted ids differ only at the queried value",
                         "aligned_pairs": sum(alignment), "pairs": n, "passed": all(alignment)})
    summary = []
    for p, name in enumerate(position_names):
        for s, site in enumerate(sites):
            for li, layer in enumerate(patch_layers):
                summary.append({"position": name, "site": site, "layer": layer, "layer_type": layer_types[layer],
                                **{f"recovery_{k}": v for k, v in summarize(recover[p, s, li].tolist(), seed).items()},
                                "mean_patch_norm": magnitude[p, s, li].mean().item()})
    random_summary = []
    for pi, name in enumerate(control_positions):
        for si, site in enumerate(control_sites):
            for li, layer in enumerate(patch_layers):
                stats = summarize(random_recover[pi, si, li].mean(0).tolist(), seed)
                random_summary.append({"position": name, "site": site, "layer": layer, "layer_type": layer_types[layer],
                                       **{f"recovery_{k}": v for k, v in stats.items()},
                                       "max_abs_single_seed_mean": random_recover[pi, si, li].mean(-1).abs().max().item()})
    by_type = {}
    for name in position_names:
        for site in sites:
            for kind in sorted(set(layer_types)):
                cells = [r["recovery_mean"] for r in summary if r["position"] == name and r["site"] == site and r["layer_type"] == kind]
                by_type.setdefault(name, {}).setdefault(site, {})[kind] = {"mean_over_layers": sum(cells) / len(cells),
                                                                           "max_over_layers": max(cells)}
    torch.save({"recovery": recover, "random_control_recovery": random_recover, "patch_norm": magnitude,
                "layers": patch_layers, "axes": {"recovery": ["position", "site", "layer", "example"],
                         "random_control_recovery": ["position", "site", "layer", "seed", "example"]},
                "positions": position_names, "sites": sites, "control_positions": control_positions,
                "control_sites": control_sites, "seeds": seeds, "clean_logit_difference": ld_clean,
                "corrupted_logit_difference": ld_corrupt}, details["_dir"] / "patching.pt")
    details["patching_examples.json"] = [
        {"index": i, "clean_text": e.text, "clean_answer": e.target, "corrupted_answer": c.target,
         "positions": {k: e.positions[k] for k in position_names}, "distance": e.distance,
         "clean_logit_difference": ld_clean[i].item(), "corrupted_logit_difference": ld_corrupt[i].item(),
         "recovery": {name: {site: [round(recover[p, s, li, i].item(), 5) for li in range(len(patch_layers))]
                             for s, site in enumerate(sites)} for p, name in enumerate(position_names)}}
        for i, (e, c) in enumerate(zip(clean, corrupted))]
    results.update({
        "heldout_behavior": {"clean_logit_difference": summarize(ld_clean.tolist(), seed),
                             "corrupted_logit_difference": summarize(ld_corrupt.tolist(), seed),
                             "mean_distance": sum(e.distance for e in clean) / n, "n_tokens": len(clean[0].ids)},
        "metric": "recovery = (LD_patched - LD_corrupted) / (LD_clean - LD_corrupted), LD = logit(clean answer) - logit(corrupted answer) at the final position",
        "patched_layers": patch_layers, "summary": summary, "random_control": random_summary, "random_control_seeds": seeds,
        "by_layer_type": by_type, "controls": controls,
        "controls_passed": all(c["passed"] for c in controls), "capture_metadata_example": capture_record_example,
        "artifacts": {"tensors": "details/patching.pt", "per_example": "details/patching_examples.json"}})
    return results


def _controls(runtime, clean, corrupted, clean_acts, corrupt_acts, logits_corrupt, plain_clean, logits_clean, forward,
              corrupt_ids, positions, sites, position_names, n_layers, hidden):
    """Read-only, zero-strength, identity-patch, position/site isolation and cached-decode reset checks."""
    model = runtime.model
    controls = [{"control": "read_only_capture", "description": "Clean logits with all-site capture vs plain forward",
                 "max_abs_logit_difference": (logits_clean - plain_clean).abs().max().item(),
                 "passed": torch.equal(logits_clean, plain_clean)}]
    sample_layers = sorted({0, n_layers // 2, n_layers - 1})
    zero, identity = True, True
    for layer in sample_layers:
        for site in sites:
            with Intervention(model, layer, site, "all", direction=random_direction(hidden, layer), alpha=0.0):
                zero &= torch.equal(forward(corrupt_ids, full=True), logits_corrupt)
            p = position_names.index("value")
            with Intervention(model, layer, site, [[x] for x in positions["value"]],
                              patch=corrupt_acts[(layer, site)][:, p:p + 1]):
                identity &= torch.equal(forward(corrupt_ids, full=True), logits_corrupt)
    controls.append({"control": "zero_strength_steering", "description": f"alpha=0 at all positions, layers {sample_layers}, all sites",
                     "passed": bool(zero)})
    controls.append({"control": "identity_patch", "description": "Corrupted activation patched into corrupted run",
                     "passed": bool(identity)})
    # Position/site isolation: patch the value position at one layer, observe every position.
    value_rows = [[x] for x in positions["value"]]
    isolation = []
    for layer in sample_layers:
        p = position_names.index("value")
        with Capture(model, [(layer, "residual_output"), (n_layers - 1, "residual_output")], positions="all",
                     max_bytes=1 << 30) as before:
            forward(corrupt_ids)
        with Intervention(model, layer, "residual_output", value_rows, patch=clean_acts[(layer, "residual_output")][:, p:p + 1]), \
             Capture(model, [(layer, "residual_output"), (n_layers - 1, "residual_output")], positions="all",
                     max_bytes=1 << 30) as after:
            forward(corrupt_ids)
        changed = (before.tensor(layer, "residual_output") != after.tensor(layer, "residual_output")).any(-1)
        expected = torch.zeros_like(changed)
        for row, x in enumerate(positions["value"]):
            expected[row, x] = True
        downstream = (before.tensor(n_layers - 1, "residual_output") != after.tensor(n_layers - 1, "residual_output")).any(-1)
        earlier_untouched = all(not downstream[row, :x].any() for row, x in enumerate(positions["value"]))
        isolation.append({"layer": layer, "declared_positions_changed": bool(changed[expected].all()),
                          "no_undeclared_position_changed_at_site": bool(not (changed & ~expected).any()),
                          "positions_before_patch_unchanged_at_last_layer": bool(earlier_untouched)})
        with Intervention(model, layer, "mixer_output", value_rows, patch=clean_acts[(layer, "mixer_output")][:, p:p + 1]), \
             Capture(model, [(layer, "residual_input"), (layer, "post_mixer_residual")], positions="all", max_bytes=1 << 30) as site_after:
            forward(corrupt_ids)
        with Capture(model, [(layer, "residual_input"), (layer, "post_mixer_residual")], positions="all", max_bytes=1 << 30) as site_before:
            forward(corrupt_ids)
        upstream_same = torch.equal(site_before.tensor(layer, "residual_input"), site_after.tensor(layer, "residual_input"))
        post_changed = (site_before.tensor(layer, "post_mixer_residual") != site_after.tensor(layer, "post_mixer_residual")).any(-1)
        isolation[-1]["mixer_patch_leaves_residual_input_unchanged"] = bool(upstream_same)
        isolation[-1]["mixer_patch_changes_only_declared_positions"] = bool(not (post_changed & ~expected).any())
    controls.append({"control": "position_site_isolation", "description": "Patch affects only the declared site/positions; causal past untouched",
                     "layers": isolation, "passed": all(all(v for k, v in row.items() if k != "layer") for row in isolation)})
    # A patch at position 0 (identical in clean and corrupted runs) must be an exact no-op.
    with Intervention(model, sample_layers[1], "residual_output", [[0]] * len(clean),
                      patch=clean_acts_at_zero(model, clean, sample_layers[1], runtime)):
        exact = torch.equal(forward(corrupt_ids, full=True), logits_corrupt)
    controls.append({"control": "pre_corruption_position", "description": "Clean activation at position 0 (identical prefix) patched into corrupted run",
                     "passed": bool(exact)})
    # Cached decoding repeated around a patched run: caches and hooks must reset.
    def greedy(ids, steps=3):
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        tokens, past = [out.logits[:, -1].argmax(-1, keepdim=True)], out.past_key_values
        for _ in range(steps - 1):
            out = model(input_ids=tokens[-1], past_key_values=past, use_cache=True)
            tokens.append(out.logits[:, -1].argmax(-1, keepdim=True)); past = out.past_key_values
        return torch.cat(tokens, 1).cpu()
    first = greedy(corrupt_ids[:4])
    p = position_names.index("value")
    with Intervention(model, n_layers // 2, "residual_output", [[x] for x in positions["value"][:4]],
                      patch=clean_acts[(n_layers // 2, "residual_output")][:4, p:p + 1]):
        patched_tokens = greedy(corrupt_ids[:4], 1)
    second = greedy(corrupt_ids[:4])
    controls.append({"control": "cached_decode_reset", "description": "Greedy cached decoding before/after a patched run",
                     "tokens_equal": torch.equal(first, second), "patched_first_tokens": patched_tokens[:, 0].tolist(),
                     "passed": torch.equal(first, second)})
    return controls


def clean_acts_at_zero(model, clean, layer, runtime):
    with Capture(model, [(layer, "residual_output")], positions=[0], max_bytes=1 << 26) as cap:
        model(input_ids=torch.tensor([e.ids for e in clean], device=runtime.device), use_cache=False, logits_to_keep=1)
    return cap.records[0]["tensor"]
