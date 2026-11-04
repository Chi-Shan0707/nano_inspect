"""Study C: contrastive additive steering h' = h + alpha * v at one residual site."""
from __future__ import annotations

from contextlib import ExitStack
import torch
import torch.nn.functional as F
from ..capture import Capture, capture_metadata
from ..interventions import Intervention, contrastive_direction, random_direction, hook_count
from .stats import summarize
from . import tasks as T


def _groups(rows):
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(len(row), []).append(index)
    return groups.values()


def _steer(model, layer, site, direction, alpha):
    """Context steering every input position (prefill: all prompt tokens; decode: the new token)."""
    if direction is None:
        return ExitStack()
    stack = ExitStack()
    stack.intervention = stack.enter_context(Intervention(model, layer, site, "all", direction=direction, alpha=alpha))
    return stack


def sentiment_scores(model, rows, device, positive, negative, steer=None):
    """Next-token log-odds of the positive vs negative evaluation word sets, per prompt."""
    scores = [None] * len(rows)
    for group in _groups(rows):
        with (steer() if steer else ExitStack()):
            logits = model(input_ids=torch.tensor([rows[i] for i in group], device=device), use_cache=False,
                           logits_to_keep=1).logits[:, -1].float()
        logprobs = logits.log_softmax(-1)
        values = logprobs[:, positive].logsumexp(-1) - logprobs[:, negative].logsumexp(-1)
        for j, i in enumerate(group):
            scores[i] = values[j].item()
    return scores


def capital_quality(model, prompts, device, steer=None):
    rows = [p["ids"] for p in prompts]
    result = [None] * len(rows)
    for group in _groups(rows):
        with (steer() if steer else ExitStack()):
            logits = model(input_ids=torch.tensor([rows[i] for i in group], device=device), use_cache=False,
                           logits_to_keep=1).logits[:, -1].float()
        logprobs = logits.log_softmax(-1)
        for j, i in enumerate(group):
            target = prompts[i]["target_id"]
            result[i] = {"correct": int(logits[j].argmax()) == target, "target_logprob": logprobs[j, target].item(),
                         "top1_id": int(logits[j].argmax())}
    return result


def neutral_nll(model, rows, device, steer=None):
    """Mean next-token negative log-likelihood per text (teacher forcing, every position steered)."""
    result = [None] * len(rows)
    for group in _groups(rows):
        ids = torch.tensor([rows[i] for i in group], device=device)
        with (steer() if steer else ExitStack()):
            logits = model(input_ids=ids, use_cache=False).logits[:, :-1].float()
        nll = F.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="none").mean(-1)
        for j, i in enumerate(group):
            result[i] = nll[j].item()
    return result


def generate(model, ids, steps, steer_factory=None, policy="decode_time"):
    """Greedy cached generation. prefill_only removes the steering hook before the first decode step."""
    stack = steer_factory() if steer_factory else ExitStack()
    try:
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        if policy == "prefill_only":
            stack.close()
        elif policy != "decode_time":
            raise ValueError(f"Unknown policy {policy!r}")
        past, tokens = out.past_key_values, [out.logits[:, -1].argmax(-1, keepdim=True)]
        for _ in range(steps - 1):
            out = model(input_ids=tokens[-1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            tokens.append(out.logits[:, -1].argmax(-1, keepdim=True))
    finally:
        stack.close()
    return torch.cat(tokens, 1).cpu()


def _truncate(tokenizer, ids):
    stops = {tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    out = []
    for token in ids:
        if token in stops:
            break
        out.append(token)
    return out


def judge_scores(model, tokenizer, texts, device):
    """Unsteered model, chat-formatted: 'Positive' minus 'Negative' first-answer-token logit."""
    positive, negative = (T.single_token(tokenizer, w) for w in T.JUDGE_ANSWERS)
    rows = [tokenizer.apply_chat_template([{"role": "user", "content": T.JUDGE_INSTRUCTION.format(text=text)}],
                                          tokenize=True, add_generation_prompt=True, enable_thinking=False)
            for text in texts]
    rows = [list(r["input_ids"]) if isinstance(r, dict) or hasattr(r, "keys") else list(r) for r in rows]
    scores = [None] * len(rows)
    for group in _groups(rows):
        logits = model(input_ids=torch.tensor([rows[i] for i in group], device=device), use_cache=False,
                       logits_to_keep=1).logits[:, -1].float()
        for j, i in enumerate(group):
            scores[i] = (logits[j, positive] - logits[j, negative]).item()
    return scores


def lexicon_score(text):
    words = [w.strip(".,!?;:\"'()").lower() for w in text.split()]
    return sum(w in T.POSITIVE_LEXICON for w in words) - sum(w in T.NEGATIVE_LEXICON for w in words)


def repetition(ids):
    return 1.0 - len(set(ids)) / len(ids) if ids else 0.0


def study_c(runtime, config, details):
    cfg = config["steering"]
    model, tokenizer, device = runtime.model, runtime.tokenizer, runtime.device
    seed = int(config.get("seed", 0))
    hidden = model.config.get_text_config().hidden_size
    T.check_disjoint()
    pairs = T.steering_pairs(tokenizer)
    pilot = T.steering_prompts(tokenizer, T.PILOT_NOUNS)
    heldout = T.steering_prompts(tokenizer, T.HELDOUT_NOUNS, limit=int(cfg.get("heldout_prompts", 60)))
    if len(heldout) < int(cfg.get("min_heldout", 50)):
        raise RuntimeError(f"Only {len(heldout)} single-token held-out nouns available")
    positive = [T.single_token(tokenizer, w) for w in T.POSITIVE_EVAL]
    negative = [T.single_token(tokenizer, w) for w in T.NEGATIVE_EVAL]
    site, candidates = cfg.get("site", "residual_output"), [int(x) for x in cfg["candidate_layers"]]
    hooks_before = hook_count(model)
    pos_ids = torch.tensor([p["positive_ids"] for p in pairs], device=device)
    neg_ids = torch.tensor([p["negative_ids"] for p in pairs], device=device)
    directions, conventions = {}, {}
    with torch.inference_mode():
        # 1. Training split only: aligned final-token residuals and the site's typical norm.
        selections = [(layer, site) for layer in candidates]
        with Capture(model, selections, positions=[-1], max_bytes=1 << 26,
                     metadata=capture_metadata(runtime, "experiment", "train", pos_ids.tolist())) as cap_pos:
            model(input_ids=pos_ids, use_cache=False, logits_to_keep=1)
        with Capture(model, selections, positions=[-1], max_bytes=1 << 26,
                     metadata=capture_metadata(runtime, "experiment", "train", neg_ids.tolist())) as cap_neg:
            model(input_ids=neg_ids, use_cache=False, logits_to_keep=1)
        with Capture(model, selections, positions="all", max_bytes=1 << 28) as cap_all:
            model(input_ids=torch.cat([pos_ids, neg_ids]), use_cache=False, logits_to_keep=1)
        for layer in candidates:
            a, b = cap_pos.tensor(layer, site)[:, 0], cap_neg.tensor(layer, site)[:, 0]
            direction, raw = contrastive_direction(a, b)
            differences = a.float() - b.float()
            residual = cap_all.tensor(layer, site).float()[:, 1:].norm(dim=-1).mean().item()
            directions[layer] = direction
            conventions[layer] = {"raw_direction_norm": raw, "unit_direction": True,
                                  "mean_residual_norm": residual,
                                  "pair_cosine_with_direction": summarize(
                                      F.cosine_similarity(differences, direction[None], dim=-1).tolist(), seed),
                                  "position": "final token of each aligned training sentence",
                                  "residual_norm_positions": "all training positions except position 0"}
        # 2. Pilot split: confirm the behavior exists and select the layer.
        pilot_rows = [p["ids"] for p in pilot]
        pilot_base = sentiment_scores(model, pilot_rows, device, positive, negative)
        pilot_alpha = float(cfg.get("pilot_alpha", 0.25))
        selection = []
        for layer in candidates:
            scale = conventions[layer]["mean_residual_norm"]
            up = sentiment_scores(model, pilot_rows, device, positive, negative,
                                  lambda: _steer(model, layer, site, directions[layer], pilot_alpha * scale))
            down = sentiment_scores(model, pilot_rows, device, positive, negative,
                                    lambda: _steer(model, layer, site, directions[layer], -pilot_alpha * scale))
            selection.append({"layer": layer, "alpha_relative": pilot_alpha,
                              "mean_shift_positive": sum(u - b for u, b in zip(up, pilot_base)) / len(up),
                              "mean_shift_negative": sum(d - b for d, b in zip(down, pilot_base)) / len(down),
                              "half_range": sum(u - d for u, d in zip(up, down)) / (2 * len(up))})
        layer = max(selection, key=lambda r: r["half_range"])["layer"] if cfg.get("layer", "auto") == "auto" else int(cfg["layer"])
        scale = conventions[layer]["mean_residual_norm"]
        # 3. Held-out evaluation.
        held_rows = [p["ids"] for p in heldout]
        capitals = T.capital_prompts(tokenizer)
        capital_base = capital_quality(model, capitals, device)
        quality_set = [c for c, r in zip(capitals, capital_base) if r["correct"]]
        neutral_rows = [tokenizer.encode(t, add_special_tokens=False) for t in T.NEUTRAL_TEXTS]
        evaluated = {"contrastive": directions[layer]}
        for value in cfg.get("random_seeds", [0, 1, 2, 3, 4]):
            evaluated[f"random_{value}"] = random_direction(hidden, 7919 * (int(value) + 1), norm=directions[layer].norm().item())
        alphas = [float(a) for a in cfg["alphas"]]
        steps = int(cfg.get("generate_tokens", 12))
        policies = ["prefill_only", "decode_time"]
        judge_rows = [tokenizer.decode(p["positive_ids"]) for p in pairs] + [tokenizer.decode(p["negative_ids"]) for p in pairs]
        judge_check = judge_scores(model, tokenizer, judge_rows, device)
        judge_accuracy = (sum(s > 0 for s in judge_check[:len(pairs)]) + sum(s < 0 for s in judge_check[len(pairs):])) / len(judge_rows)
        base_target = sentiment_scores(model, held_rows, device, positive, negative)
        base_quality = capital_quality(model, quality_set, device)
        base_nll = neutral_nll(model, neutral_rows, device)
        base_generation = generate(model, torch.tensor(held_rows, device=device), steps)
        sweep, per_example, generations = [], [], []
        for name, direction in evaluated.items():
            for alpha_relative in alphas:
                alpha = alpha_relative * scale
                factory = (lambda d=direction, a=alpha: _steer(model, layer, site, d, a))
                target = sentiment_scores(model, held_rows, device, positive, negative, factory)
                quality = capital_quality(model, quality_set, device, factory)
                nll = neutral_nll(model, neutral_rows, device, factory)
                row = {"direction": name, "alpha_relative": alpha_relative, "alpha": alpha,
                       "added_vector_norm": abs(alpha) * direction.norm().item(),
                       "target_score": summarize(target, seed),
                       "target_shift": summarize([t - b for t, b in zip(target, base_target)], seed),
                       "quality_accuracy": sum(q["correct"] for q in quality) / len(quality),
                       "quality_logprob_change": summarize([q["target_logprob"] - b["target_logprob"]
                                                            for q, b in zip(quality, base_quality)], seed),
                       "neutral_nll_change": summarize([x - y for x, y in zip(nll, base_nll)], seed), "policies": {}}
                for policy in policies:
                    tokens = generate(model, torch.tensor(held_rows, device=device), steps, factory, policy)
                    continuations = [_truncate(tokenizer, t) for t in tokens.tolist()]
                    texts = [tokenizer.decode(c) for c in continuations]
                    judged = judge_scores(model, tokenizer, [p["text"] + t for p, t in zip(heldout, texts)], device)
                    lexicon = [lexicon_score(t) for t in texts]
                    reps = [repetition(c) for c in continuations]
                    row["policies"][policy] = {
                        "judge_score": summarize(judged, seed),
                        "judge_positive_fraction": sum(j > 0 for j in judged) / len(judged),
                        "lexicon_score": summarize(lexicon, seed),
                        "repetition": summarize(reps, seed),
                        "changed_vs_unsteered_fraction": sum(not torch.equal(a, b) for a, b in zip(tokens, base_generation)) / len(tokens)}
                    if name == "contrastive" or alpha_relative in (min(alphas), max(alphas)):
                        generations.extend({"direction": name, "alpha_relative": alpha_relative, "policy": policy,
                                            "noun": p["noun"], "prompt": p["text"], "continuation": t, "judge": j,
                                            "lexicon": lx, "repetition": r}
                                           for p, t, j, lx, r in zip(heldout, texts, judged, lexicon, reps))
                sweep.append(row)
                per_example.extend({"direction": name, "alpha_relative": alpha_relative, "noun": p["noun"],
                                    "target_score": t, "target_shift": t - b}
                                   for p, t, b in zip(heldout, target, base_target))
        # 4. Controls.
        zero = next(r for r in sweep if r["direction"] == "contrastive" and r["alpha_relative"] == 0.0)
        zero_logits_equal = True
        with _steer(model, layer, site, directions[layer], 0.0):
            steered = model(input_ids=torch.tensor(held_rows, device=device), use_cache=False, logits_to_keep=1).logits
        plain = model(input_ids=torch.tensor(held_rows, device=device), use_cache=False, logits_to_keep=1).logits
        zero_logits_equal = torch.equal(steered, plain)
        policy_first_token = []
        for alpha_relative in (min(alphas), max(alphas)):
            factory = (lambda a=alpha_relative * scale: _steer(model, layer, site, directions[layer], a))
            first = [generate(model, torch.tensor(held_rows, device=device), 2, factory, policy) for policy in policies]
            policy_first_token.append(torch.equal(first[0][:, 0], first[1][:, 0]))
        repeat_generation = generate(model, torch.tensor(held_rows, device=device), steps)
        controls = [
            {"control": "zero_strength", "description": "alpha=0 steering vs plain forward (all prompt positions)",
             "passed": bool(zero_logits_equal and zero["target_shift"]["mean"] == 0.0
                            and all(p["changed_vs_unsteered_fraction"] == 0.0 for p in zero["policies"].values()))},
            {"control": "policies_share_prefill", "description": "prefill-only and decode-time give the same first token",
             "passed": all(policy_first_token)},
            {"control": "generation_reset", "description": "Unsteered generation repeated after the sweep",
             "passed": torch.equal(repeat_generation, base_generation)},
            {"control": "hook_cleanup", "hooks_before": hooks_before, "hooks_after": hook_count(model),
             "passed": hook_count(model) == hooks_before},
            {"control": "judge_validation", "description": "Unsteered judge on aligned training sentences",
             "accuracy": judge_accuracy, "passed": judge_accuracy >= 0.9},
            {"control": "random_norm_matched", "description": "Random directions have the contrastive direction's norm",
             "norms": {k: v.norm().item() for k, v in evaluated.items()},
             "passed": all(abs(v.norm().item() - directions[layer].norm().item()) < 1e-4 for v in evaluated.values())}]
    extreme_abs = max(abs(a) for a in alphas)
    failures = []
    for sign in (1, -1):
        failures += [g for g in generations if g["direction"] == "contrastive" and g["policy"] == "decode_time"
                     and g["alpha_relative"] == sign * extreme_abs][:4]
    capital_failures = []
    with torch.inference_mode():
        for extreme in (min(alphas), max(alphas)):
            quality = capital_quality(model, quality_set, device,
                                      lambda a=extreme: _steer(model, layer, site, directions[layer], a * scale))
            wrong = [(prompt, result) for prompt, result in zip(quality_set, quality) if not result["correct"]]
            capital_failures += [{"country": prompt["country"], "expected": prompt["answer"],
                                  "predicted": tokenizer.decode([result["top1_id"]]), "alpha_relative": extreme}
                                 for prompt, result in wrong[:4]]
            prediction_counts = {}
            for _, result in wrong:
                token = tokenizer.decode([result["top1_id"]])
                prediction_counts[token] = prediction_counts.get(token, 0) + 1
            capital_failures.append({"country": f"all {len(wrong)} errors", "expected": "-",
                                     "predicted": dict(sorted(prediction_counts.items(), key=lambda kv: -kv[1])[:5]),
                                     "alpha_relative": extreme})
    torch.save({"directions": {k: v for k, v in directions.items()}, "selected_layer": layer, "site": site,
                "random_directions": {k: v for k, v in evaluated.items() if k != "contrastive"}},
               details["_dir"] / "steering_directions.pt")
    details["steering_examples.json"] = per_example
    details["steering_generations.json"] = generations
    return {"question": "Does a training-split contrastive direction steer held-out prompts, and at what cost?",
            "behavior": "sentiment of a review continuation",
            "splits": {"train_pairs": len(pairs), "pilot_prompts": len(pilot), "heldout_prompts": len(heldout),
                       "quality_capitals_baseline_correct": len(quality_set), "quality_capitals_total": len(capitals),
                       "neutral_texts": len(neutral_rows)},
            "templates": {"train": T.TRAIN_TEMPLATE, "evaluation": T.EVAL_TEMPLATE, "capital": T.CAPITAL_TEMPLATE,
                          "judge": T.JUDGE_INSTRUCTION, "judge_answers": list(T.JUDGE_ANSWERS)},
            "words": {"train_positive": T.POSITIVE_TRAIN[:2], "train_negative": T.NEGATIVE_TRAIN[:2],
                      "eval_positive": T.POSITIVE_EVAL, "eval_negative": T.NEGATIVE_EVAL},
            "site": site, "selected_layer": layer, "layer_selection": "pilot split only" if cfg.get("layer", "auto") == "auto" else "fixed in config",
            "pilot_selection": selection, "pilot_baseline_score": summarize(pilot_base, seed),
            "conventions": {str(k): v for k, v in conventions.items()},
            "scale": {"alpha_unit": "mean residual L2 norm at the site on training prompts (excluding position 0)",
                      "unit_value": scale, "formula": "h' = h + alpha_relative * unit_value * v_hat, applied at every prompt position (prefill) and, for decode_time, at every generated position"},
            "baseline": {"target_score": summarize(base_target, seed), "capital_accuracy": sum(r["correct"] for r in capital_base) / len(capital_base),
                         "neutral_nll": summarize(base_nll, seed)},
            "sweep": sweep, "controls": controls, "controls_passed": all(c["passed"] for c in controls),
            "failure_examples": failures, "quality_failure_examples": capital_failures,
            "artifacts": {"per_example": "details/steering_examples.json", "generations": "details/steering_generations.json",
                          "directions": "details/steering_directions.pt"}}
