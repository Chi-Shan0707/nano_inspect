"""HTML/Markdown sections and figures for experiment runs (rendered from experiment.json)."""
from __future__ import annotations

from pathlib import Path
from ..reporting import _escape, _fmt, _md_table, _plt, _render, _table

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots, fixed order
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
NEGATIVE, NEUTRAL, POSITIVE = "#e34948", "#f0efec", "#2a78d6"


def _axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.title.set_color(INK)


def _save(fig, run, name):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(run / name, dpi=150, facecolor=SURFACE)
    _plt().close(fig)
    return name


def _figure(name, caption):
    return f"<figure><img src='{name}' alt='{_escape(caption)}'><figcaption>{_escape(caption)}</figcaption></figure>", \
        [f"![{caption}]({name})", ""]


def _ci(stats):
    if not stats or stats.get("mean") is None:
        return "n/a"
    return f"{stats['mean']:.3f} [{stats['ci95'][0]:.3f}, {stats['ci95'][1]:.3f}] (n={stats['n']})"


def _retrieval(study, run):
    plt = _plt()
    conditions = study["conditions"]
    html, md = ["<h3>Study A — retrieval across distance and distractors</h3>",
                f"<p>{_escape(study['question'])} Pilot accuracy ≥ {study['solvable_threshold']} marks a condition "
                "solvable; held-out examples are scored separately and never used for that decision.</p>"], \
        ["### Study A — retrieval across distance and distractors", ""]
    rows = [{"format": c["format"], "facts": c["n_facts"], "filler": c["filler_sentences"], "tokens": c["n_tokens"],
             "mean distance": c["mean_distance"], "pilot acc": c["pilot_accuracy"], "solvable": c["solvable"],
             "held-out acc [95% CI]": _ci(c["heldout_accuracy"]), "margin [95% CI]": _ci(c["heldout_margin"]),
             "acc by query third": c["accuracy_by_query_third"],
             "KV bytes": c["cache"]["attention_kv"], "recurrent bytes": c["cache"]["recurrent_state"],
             "conv bytes": c["cache"]["convolution_state"]} for c in conditions]
    columns = list(rows[0])
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex="col")
    for col, fmt in enumerate(sorted({c["format"] for c in conditions}, key=lambda f: f != "sentence")):
        for k, n in enumerate(sorted({c["n_facts"] for c in conditions})):
            subset = sorted([c for c in conditions if c["format"] == fmt and c["n_facts"] == n], key=lambda c: c["mean_distance"])
            x = [c["mean_distance"] for c in subset]
            for row, key in ((0, "heldout_accuracy"), (1, "heldout_margin")):
                mean = [c[key]["mean"] for c in subset]
                low = [c[key]["mean"] - c[key]["ci95"][0] for c in subset]
                high = [c[key]["ci95"][1] - c[key]["mean"] for c in subset]
                axes[row, col].errorbar(x, mean, yerr=[low, high], color=SERIES[k], marker="o", markersize=5,
                                        linewidth=2, capsize=3, label=f"{n} facts ({n - 1} distractors)")
        axes[0, col].set_title(f"{fmt} format: held-out accuracy", fontsize=10)
        axes[1, col].set_title(f"{fmt} format: target − best distractor logit", fontsize=10)
        axes[1, col].set_xlabel("mean tokens from queried value to prediction (log scale)", color=MUTED, fontsize=9)
        for row in (0, 1):
            from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
            axes[row, col].set_xscale("log")
            ticks = [t for t in (3, 10, 30, 100, 300, 1000) if min(c["mean_distance"] for c in conditions if c["format"] == fmt) / 1.5
                     <= t <= max(c["mean_distance"] for c in conditions if c["format"] == fmt) * 1.5]
            axes[row, col].xaxis.set_major_locator(FixedLocator(ticks))
            axes[row, col].xaxis.set_major_formatter(ScalarFormatter())
            axes[row, col].xaxis.set_minor_locator(NullLocator())
            _axes(axes[row, col])
        axes[0, col].set_ylim(0, 1.05)
    axes[0, 0].legend(fontsize=8, frameon=False)
    name = _save(fig, run, "retrieval-behavior.png")
    caption = "Held-out retrieval accuracy and logit margin (mean, 95% bootstrap CI) versus distance, by number of facts."
    figure_html, figure_md = _figure(name, caption)
    html += [figure_html, _table(rows, columns)]
    md += figure_md + _md_table(rows, columns)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    caches = sorted({(c["cache"]["sequence_length"], c["cache"]["attention_kv"], c["cache"]["recurrent_state"],
                      c["cache"]["convolution_state"]) for c in conditions})
    lengths = [c[0] for c in caches]
    for k, (index, label) in enumerate(((1, "full-attention KV (6 layers)"), (2, "gated-delta recurrent state (18 layers)"),
                                        (3, "gated-delta convolution history (18 layers)"))):
        ax.plot(lengths, [c[index] / 2 ** 20 for c in caches], color=SERIES[k], marker="o", markersize=5, linewidth=2, label=label)
    ax.set_xlabel("prompt tokens", color=MUTED, fontsize=9)
    ax.set_ylabel("MiB (unique tensor storage)", color=MUTED, fontsize=9)
    ax.set_title("Measured cache storage after prefill (batch 1)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    _axes(ax)
    name = _save(fig, run, "retrieval-cache.png")
    figure_html, figure_md = _figure(name, "Measured cached-state bytes versus prompt length; KV grows, recurrent state is fixed.")
    html.append(figure_html)
    md += figure_md
    if study.get("failure_examples"):
        failures = [{k: r[k] for k in ("format", "n_facts", "filler_sentences", "distance", "target", "top1", "best_distractor",
                                       "margin_vs_best_distractor")} for r in study["failure_examples"]]
        html.append("<details><summary>Held-out failure examples</summary>" + _table(failures, list(failures[0])) + "</details>")
        md += ["Held-out failure examples:", ""] + _md_table(failures, list(failures[0]))
    return html, md


def _patching(study, run):
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    plt = _plt()
    patched = study.get("patched_layers", list(range(len(study["layer_types"]))))
    layers = len(patched)
    sites, positions = study["sites"], study["positions"]
    full = [i for i, layer in enumerate(patched) if study["layer_types"][layer] == "full_attention"]  # column indices
    held = study["heldout_behavior"]
    html = ["<h3>Study B — clean-to-corrupted activation patching</h3>",
            f"<p>{_escape(study['question'])} Condition {_escape(_fmt(study['condition']))}; {held['n_tokens']} tokens, "
            f"mean distance {held['mean_distance']:.1f}. Metric: {_escape(study['metric'])}.</p>",
            f"<p>Held-out selection {_escape(_fmt(study['heldout_selection']))}; pilot selection {_escape(_fmt(study['pilot_selection']))}. "
            f"Clean LD {_escape(_ci(held['clean_logit_difference']))}; corrupted LD {_escape(_ci(held['corrupted_logit_difference']))}.</p>"]
    md = ["### Study B — clean-to-corrupted activation patching", "",
          f"Condition {_fmt(study['condition'])}; held-out selection {_fmt(study['heldout_selection'])}; "
          f"clean LD {_ci(held['clean_logit_difference'])}; corrupted LD {_ci(held['corrupted_logit_difference'])}.", "",
          f"Metric: {study['metric']}", ""]
    grid = np.zeros((len(positions), len(sites), layers))
    for row in study["summary"]:
        grid[positions.index(row["position"]), sites.index(row["site"]), patched.index(row["layer"])] = row["recovery_mean"]
    cmap = LinearSegmentedColormap.from_list("diverging", [NEGATIVE, NEUTRAL, POSITIVE])
    norm = TwoSlopeNorm(vcenter=0.0, vmin=min(-0.25, grid.min()), vmax=max(1.0, grid.max()))
    fig, axes = plt.subplots(len(positions), 1, figsize=(11, 2.0 * len(positions) + 0.6), sharex=True)
    for p, name in enumerate(positions):
        ax = axes[p]
        image = ax.imshow(grid[p], aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_yticks(range(len(sites)), sites, fontsize=8)
        ax.set_title(f"patched position: {name}", fontsize=9, loc="left")
        ax.tick_params(colors=MUTED, labelsize=8)
        for layer in full:
            ax.add_patch(plt.Rectangle((layer - 0.5, -0.5), 1, len(sites), fill=False, edgecolor=INK, linewidth=0.8))
    axes[-1].set_xticks(range(layers), [f"{layer}{'A' if i in full else ''}" for i, layer in enumerate(patched)], fontsize=8)
    axes[-1].set_xlabel("layer (A = full attention, outlined; others gated delta)", color=MUTED, fontsize=9)
    colorbar = fig.colorbar(image, ax=list(axes), shrink=0.8, pad=0.01)
    colorbar.set_label("mean recovery of logit difference", color=MUTED, fontsize=8)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(run / "patching-heatmap.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    figure_html, figure_md = _figure("patching-heatmap.png", "Mean held-out recovery when one clean activation is restored "
                                     "(layer × site, per aligned position). Patching effects, not attention weights.")
    html.append(figure_html)
    md += figure_md
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, site in zip(axes, ["residual_output", "mixer_output"]):
        if site not in sites:
            continue
        for k, name in enumerate(positions):
            cells = sorted([r for r in study["summary"] if r["site"] == site and r["position"] == name], key=lambda r: r["layer"])
            x = [r["layer"] for r in cells]
            ax.plot(x, [r["recovery_mean"] for r in cells], color=SERIES[k], linewidth=2, label=name)
            ax.fill_between(x, [r["recovery_ci95"][0] for r in cells], [r["recovery_ci95"][1] for r in cells], color=SERIES[k], alpha=0.15, linewidth=0)
        controls = [r for r in study["random_control"] if r["site"] == site]
        for name in sorted({r["position"] for r in controls}):
            cells = sorted([r for r in controls if r["position"] == name], key=lambda r: r["layer"])
            ax.plot([r["layer"] for r in cells], [r["recovery_mean"] for r in cells], color=SERIES[positions.index(name)],
                    linewidth=1.2, linestyle="--", label=f"{name}: norm-matched random")
        for column in full:
            ax.axvline(patched[column], color=GRID, linewidth=3, zorder=0)
        ax.set_title(f"{site}", fontsize=10)
        ax.set_xlabel("layer (shaded = full attention)", color=MUTED, fontsize=9)
        _axes(ax)
    axes[0].set_ylabel("recovery (mean, 95% CI)", color=MUTED, fontsize=9)
    axes[1].legend(fontsize=7, frameon=False, ncol=2)
    name = _save(fig, run, "patching-lines.png")
    figure_html, figure_md = _figure(name, "Recovery by layer for residual_output and mixer_output patches, with "
                                     "norm-matched random-displacement controls (dashed).")
    html.append(figure_html)
    md += figure_md
    type_rows = [{"position": p, "site": s, **{f"{kind} mean": v["mean_over_layers"] for kind, v in by_kind.items()},
                  **{f"{kind} max": v["max_over_layers"] for kind, v in by_kind.items()}}
                 for p, by_site in study["by_layer_type"].items() for s, by_kind in by_site.items()]
    html += ["<h4>Recovery by mixer type (averaged over layers of each type)</h4>", _table(type_rows, list(type_rows[0]))]
    md += ["Recovery by mixer type:", ""] + _md_table(type_rows, list(type_rows[0]))
    top = sorted(study["summary"], key=lambda r: -r["recovery_mean"])[:12]
    columns = ["position", "site", "layer", "layer_type", "recovery_mean", "recovery_sem", "recovery_ci95", "recovery_n", "mean_patch_norm"]
    html += ["<h4>Largest mean recoveries</h4>", _table(top, columns),
             "<details><summary>All layer × site × position cells</summary>", _table(study["summary"], columns), "</details>",
             "<details><summary>Norm-matched random controls</summary>",
             _table(study["random_control"], ["position", "site", "layer", "layer_type", "recovery_mean", "recovery_ci95",
                                              "max_abs_single_seed_mean"]), "</details>"]
    md += ["Largest mean recoveries:", ""] + _md_table(top, columns)
    control_rows = [{k: v for k, v in c.items() if k != "layers"} for c in study["controls"]]
    html += [f"<h4>Controls (all passed: {study['controls_passed']})</h4>",
             _table(control_rows, ["control", "description", "passed", "max_abs_logit_difference", "aligned_pairs",
                                   "hooks_before", "hooks_after"], status="passed")]
    isolation = next((c for c in study["controls"] if c["control"] == "position_site_isolation"), None)
    if isolation:
        html.append(_table(isolation["layers"], list(isolation["layers"][0])))
    md += [f"Controls (all passed: {study['controls_passed']}):", ""] + _md_table(control_rows, ["control", "passed", "description"])
    html.append("<details><summary>Example capture metadata</summary>" + _render(study["capture_metadata_example"]) + "</details>")
    return html, md


def _steering(study, run):
    plt = _plt()
    sweep = study["sweep"]
    layer = study["selected_layer"]
    html = ["<h3>Study C — contrastive additive steering</h3>",
            f"<p>{_escape(study['question'])} Behavior: {_escape(study['behavior'])}. Site {_escape(study['site'])} of layer "
            f"{layer} (selected by {_escape(study['layer_selection'])}). {_escape(study['scale']['formula'])}; one alpha unit = "
            f"{study['scale']['unit_value']:.3f}.</p>", f"<p>Splits: {_escape(_fmt(study['splits']))}</p>"]
    md = ["### Study C — contrastive additive steering", "",
          f"Layer {layer} {study['site']} ({study['layer_selection']}); {study['scale']['formula']}; unit {study['scale']['unit_value']:.3f}.",
          f"Splits: {_fmt(study['splits'])}.", ""]
    html += ["<h4>Pilot layer selection</h4>", _table(study["pilot_selection"], list(study["pilot_selection"][0])),
             "<h4>Direction conventions (training split)</h4>",
             _table([{"layer": k, **v} for k, v in study["conventions"].items()],
                    ["layer", "raw_direction_norm", "mean_residual_norm", "pair_cosine_with_direction", "position"])]
    md += ["Pilot layer selection:", ""] + _md_table(study["pilot_selection"], list(study["pilot_selection"][0]))
    alphas = sorted({r["alpha_relative"] for r in sweep})
    randoms = sorted({r["direction"] for r in sweep if r["direction"] != "contrastive"})

    def series(direction, getter):
        cells = sorted([r for r in sweep if r["direction"] == direction], key=lambda r: r["alpha_relative"])
        return [r["alpha_relative"] for r in cells], [getter(r) for r in cells]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [("target sentiment log-odds shift", lambda r: r["target_shift"]["mean"]),
              ("capital-city accuracy (quality)", lambda r: r["quality_accuracy"]),
              ("neutral-text NLL change (nats/token)", lambda r: r["neutral_nll_change"]["mean"])]
    for ax, (title, getter) in zip(axes, panels):
        for k, name in enumerate(randoms):
            x, y = series(name, getter)
            ax.plot(x, y, color=SERIES[1], linewidth=1, alpha=0.6, label="norm-matched random" if k == 0 else None)
        x, y = series("contrastive", getter)
        ax.plot(x, y, color=SERIES[0], linewidth=2, marker="o", markersize=5, label="contrastive direction")
        if title.startswith("target"):
            cells = sorted([r for r in sweep if r["direction"] == "contrastive"], key=lambda r: r["alpha_relative"])
            ax.fill_between(x, [r["target_shift"]["ci95"][0] for r in cells], [r["target_shift"]["ci95"][1] for r in cells],
                            color=SERIES[0], alpha=0.15, linewidth=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("alpha (× mean residual norm)", color=MUTED, fontsize=9)
        _axes(ax)
    axes[0].legend(fontsize=8, frameon=False)
    name = _save(fig, run, "steering-sweep.png")
    figure_html, figure_md = _figure(name, "Held-out target effect (mean, 95% CI) and unrelated-task quality versus signed "
                                     "strength, contrastive direction vs five norm-matched random directions.")
    html.append(figure_html)
    md += figure_md
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (title, getter) in zip(axes, [("judge: positive − negative logit", lambda r, p: r["policies"][p]["judge_score"]["mean"]),
                                          ("token repetition rate", lambda r, p: r["policies"][p]["repetition"]["mean"])]):
        for k, policy in enumerate(["prefill_only", "decode_time"]):
            x, y = series("contrastive", lambda r: getter(r, policy))
            ax.plot(x, y, color=SERIES[k * 2], linewidth=2, marker="o", markersize=5, label=f"contrastive, {policy}")
            ys = [series(name, lambda r: getter(r, policy))[1] for name in randoms]
            ax.plot(x, [sum(v) / len(v) for v in zip(*ys)], color=SERIES[k * 2], linewidth=1.2, linestyle="--",
                    label=f"random mean, {policy}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("alpha (× mean residual norm)", color=MUTED, fontsize=9)
        _axes(ax)
    axes[0].legend(fontsize=7, frameon=False)
    name = _save(fig, run, "steering-policies.png")
    figure_html, figure_md = _figure(name, "Generated-continuation sentiment (unsteered judge) and repetition for prefill-only "
                                     "versus decode-time steering.")
    html.append(figure_html)
    md += figure_md
    rows = []
    for r in sweep:
        if r["direction"] != "contrastive" and r["alpha_relative"] not in (min(alphas), max(alphas)):
            continue
        rows.append({"direction": r["direction"], "alpha": r["alpha_relative"], "target shift [CI]": _ci(r["target_shift"]),
                     "capital acc": r["quality_accuracy"], "capital logprob change": _ci(r["quality_logprob_change"]),
                     "neutral NLL change": _ci(r["neutral_nll_change"]),
                     "judge prefill-only": _ci(r["policies"]["prefill_only"]["judge_score"]),
                     "judge decode-time": _ci(r["policies"]["decode_time"]["judge_score"]),
                     "repetition decode-time": r["policies"]["decode_time"]["repetition"]["mean"]})
    html += ["<h4>Sweep (contrastive at all strengths; random directions at the extremes)</h4>", _table(rows, list(rows[0]))]
    md += ["Sweep:", ""] + _md_table(rows, list(rows[0]))
    html += [f"<h4>Controls (all passed: {study['controls_passed']})</h4>",
             _table(study["controls"], ["control", "description", "passed", "accuracy", "hooks_before", "hooks_after"], status="passed")]
    md += [f"Controls (all passed: {study['controls_passed']}):", ""] + _md_table(study["controls"], ["control", "passed", "description"])
    failures = study["failure_examples"] + [{"prompt": f"capital of {q['country']}", "continuation": f"expected {q['expected']}, got {q['predicted']}",
                                              "alpha_relative": q["alpha_relative"], "policy": "prefill"} for q in study["quality_failure_examples"]]
    if failures:
        columns = ["alpha_relative", "policy", "prompt", "continuation", "judge", "repetition"]
        html += ["<h4>Failure examples at the largest |alpha|</h4>", _table(failures, columns)]
        md += ["Failure examples at the largest |alpha|:", ""] + _md_table(failures, columns)
    return html, md


def experiment_section(experiment: dict, run: Path):
    html = ["<section><h2>Experiments</h2>",
            f"<p>Model revision <code>{_escape(experiment.get('model_revision'))}</code>, {_escape(experiment.get('dtype'))} on "
            f"{_escape(experiment.get('device'))}. Config <code>{_escape(experiment.get('config_path'))}</code>. "
            f"Timing (s): {_escape(_fmt(experiment.get('timing_seconds', {})))}. Per-example data: <code>details/</code>.</p>"]
    md = ["## Experiments", "", f"Model revision `{experiment.get('model_revision')}`; {experiment.get('dtype')}; "
          f"summary {_fmt(experiment.get('summary', {}))}.", ""]
    studies = experiment.get("studies", {})
    for key, render in (("retrieval", _retrieval), ("patching", _patching), ("steering", _steering)):
        if key in studies:
            part_html, part_md = render(studies[key], run)
            html += part_html
            md += part_md
    html.append("</section>")
    return "".join(html), md
