"""Offline, artifact-linked HTML and Markdown evidence reports.

The renderer deliberately displays recorded measurements without inventing
architecture facts or converting recurrent state into attention maps.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _render(value: Any, title: str = "Evidence") -> str:
    if isinstance(value, dict):
        rows = "".join(
            f"<tr><th>{_escape(key)}</th><td>{_render(item, str(key))}</td></tr>"
            for key, item in value.items()
        )
        return f"<table>{rows}</table>"
    if isinstance(value, list):
        if not value:
            return "<code>[]</code>"
        if all(isinstance(item, dict) for item in value):
            return "".join(
                f"<details><summary>{_escape(title)} {i}: "
                f"{_escape(item.get('name', item.get('site', item.get('path', 'record'))))}"
                f"</summary>{_render(item)}</details>"
                for i, item in enumerate(value)
            )
        return f"<pre>{_escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"
    if isinstance(value, str) and value.startswith(("https://", "http://")):
        return f'<a href="{_escape(value)}">{_escape(value)}</a>'
    return f"<code>{_escape(value)}</code>"


def _numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(_numeric_leaves(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = float(value)
    return result


def _atlas(inventory: dict, run: Path) -> str:
    """Summarize measured layer weights; never assume residual/head widths match."""
    layers = inventory.get("layers", [])
    accounting = inventory.get("accounting", {})
    if not layers:
        return ""
    stack = " → ".join(f"{layer['index']}: {layer['type']}" for layer in layers)
    content = ["<section><h2>Architecture atlas</h2>",
               f"<p>{len(layers)} measured text blocks. Layer schedule:</p><p>{_escape(stack)}</p>",
               _render(accounting, "Parameter accounting")]
    parameters = inventory.get("parameters", [])
    embedding = next((p for p in parameters if p["name"].endswith("embed_tokens.weight")), None)
    if embedding:
        vocab, width = embedding["shape"]
        content.insert(1, f"<p>Text execution: token IDs → embedding [{vocab}, {width}] → "
                       f"{len(layers)} residual blocks → final normalization → vocabulary projection. "
                       "Embedding dimensions are measured from the loaded weight; tied references are listed below.</p>")
    vision = [m for m in inventory.get("modules", []) if m.get("class") == "Qwen3_5VisionBlock"]
    content.append(f"<p>Loaded vision tower: {len(vision)} observed module instances of Qwen3_5VisionBlock. "
                   "See module records for complete paths and text-execution flags.</p>")
    selected = {}
    for layer in layers:
        selected.setdefault(layer["type"], layer)
    for kind, layer in selected.items():
        weights = layer["weight_shapes"]
        projections = [{"weight": name, "input_width": shape[1], "output_width": shape[0], "weight_shape": shape}
                       for name, shape in weights.items() if len(shape) == 2]
        label = f"Layer {layer['index']}: {kind}"
        diagram = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 230" role="img" aria-label="{_escape(label)} residual block">
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#163b60"/></marker></defs>
<g fill="#edf3f8" stroke="#163b60"><rect x="90" y="80" width="130" height="55"/><rect x="250" y="80" width="170" height="55"/><rect x="490" y="80" width="140" height="55"/><rect x="660" y="80" width="120" height="55"/></g>
<g font-family="sans-serif" font-size="16" text-anchor="middle" fill="#182333"><text x="155" y="112">Input RMSNorm</text><text x="335" y="112">{_escape(kind)}</text><text x="560" y="104">Post-mixer</text><text x="560" y="123">RMSNorm</text><text x="720" y="112">Gated MLP</text><text x="455" y="114">+</text><text x="820" y="114">+</text><text x="480" y="210">{_escape(label)} — projection dimensions below come from loaded weights</text></g>
<g fill="none" stroke="#163b60" stroke-width="2" marker-end="url(#arrow)"><path d="M10 108 H90"/><path d="M220 108 H250"/><path d="M420 108 H440"/><path d="M470 108 H490"/><path d="M630 108 H660"/><path d="M780 108 H805"/><path d="M835 108 H940"/><path d="M45 108 V35 H455 V92"/><path d="M475 108 V35 H820 V92"/></g></svg>'''
        filename = f"block-{layer['index']}.svg"
        (run / filename).write_text(diagram)
        content.extend([f"<h3>{_escape(label)}</h3><img src='{filename}' alt='{_escape(label)}'>",
                        _render(projections, "Measured projection")])
        def weight_width(suffix):
            shape = next((s for n, s in weights.items() if n.endswith(suffix)), None)
            return f"{shape[1]} → {shape[0]}" if shape and len(shape) == 2 else "see weight table"
        if kind == "full_attention":
            stages = [
                ("Q + output gate projection", weight_width("self_attn.q_proj.weight")),
                ("K / V projections", weight_width("self_attn.k_proj.weight") + " each"),
                ("Q/K RMSNorm → partial RoPE", "Keep gate separate; repeat KV across query groups"),
                ("Q × Kᵀ × scale + causal/padding mask", "Softmax over key positions → probabilities × V"),
                ("Aggregated values × sigmoid(gate)", "Concatenate heads; preserve residual separately"),
                ("Output projection → residual addition", weight_width("self_attn.o_proj.weight")),
            ]
        else:
            stages = [
                ("QKV projection → depthwise causal convolution + SiLU", weight_width("linear_attn.in_proj_qkv.weight")),
                ("Split Q/K/V; normalize Q/K and scale Q", "Control projections a/b; output gate projection z"),
                ("Decay state S ← exp(g) × S", "g = −exp(A_log) × softplus(a + dt_bias)"),
                ("Update S ← S + k ⊗ [beta × (v − kᵀS)]", "beta = sigmoid(b); separate matrix state per value head"),
                ("Readout qᵀS → gated RMSNorm", "SiLU(z) output gate; concatenate value heads"),
                ("Output projection → residual addition", weight_width("linear_attn.out_proj.weight")),
            ]
        stages.extend([
            ("Post-mixer RMSNorm → MLP gate/up projections", weight_width("mlp.gate_proj.weight") + " each"),
            ("SiLU(gate) × up → down projection → residual addition", weight_width("mlp.down_proj.weight")),
        ])
        parts = []
        for index, (operation, dimensions) in enumerate(stages):
            y = 15 + 90 * index
            parts.append(f'<rect x="30" y="{y}" width="900" height="65" rx="4" fill="#edf3f8" stroke="#163b60"/>'
                         f'<text x="480" y="{y+25}" text-anchor="middle" font-family="sans-serif" font-size="18">{_escape(operation)}</text>'
                         f'<text x="480" y="{y+49}" text-anchor="middle" font-family="sans-serif" font-size="15">{_escape(dimensions)}</text>')
            if index < len(stages) - 1:
                parts.append(f'<path d="M480 {y+65} V{y+87}" stroke="#163b60" stroke-width="2" marker-end="url(#arrow)"/>')
        expanded = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 {len(stages)*90}"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#163b60"/></marker></defs>{"".join(parts)}</svg>'
        expanded_name = f"operations-{layer['index']}.svg"
        (run / expanded_name).write_text(expanded)
        content.append(f'<details><summary>Expanded {_escape(kind)} operations</summary><img src="{expanded_name}" alt="Expanded {_escape(kind)} computation"></details>')
    components = accounting.get("by_component", {})
    if components:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh(list(components), [v["parameters"] / 1e6 for v in components.values()])
        ax.set_xlabel("Million parameters (unique loaded storage)")
        fig.tight_layout()
        fig.savefig(run / "parameter-allocation.png", dpi=160)
        plt.close(fig)
        content.append('<img src="parameter-allocation.png" alt="Measured parameter allocation">')
    caches = inventory.get("cache_measurements", [])
    content.extend(["<h3>Measured cache storage</h3>", _render(caches, "Cache measurement"), "</section>"])
    return "".join(content)


def generate_report(run_dir: str | Path, compare: str | Path | None = None) -> Path:
    """Render all JSON evidence in a retained run, optionally comparing scalars."""
    run = Path(run_dir).resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    artifacts = sorted(run.glob("*.json"))
    if not artifacts:
        raise ValueError(f"No JSON evidence found in {run}")
    evidence = {path.name: json.loads(path.read_text()) for path in artifacts}
    sections = []
    markdown = [f"# NanoInterpretability evidence: {run.name}", "",
                "Recorded artifacts are the source of truth. Expand the HTML tables for measurements, selections, controls, and source locations.", ""]
    figures: list[Path] = []
    for obj in evidence.values():
        if isinstance(obj, dict) and obj.get("kind") == "inventory":
            sections.append(_atlas(obj, run))
    # Plot measured capture statistics only; axes retain semantic site names.
    captures = [record for obj in evidence.values() if isinstance(obj, dict)
                for record in obj.get("captures", []) if isinstance(record, dict)
                and isinstance(record.get("mean"), (int, float))]
    if captures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shown = captures[:40]
        fig, ax = plt.subplots(figsize=(10, max(3, len(shown) * .25)))
        labels = [f"L{r.get('layer', '?')} {r.get('site', '?')} {r.get('phase', '')}" for r in shown]
        ax.barh(range(len(shown)), [r["mean"] for r in shown])
        ax.set_yticks(range(len(shown)), labels)
        ax.set_xlabel("Mean of selected captured tensor values")
        ax.invert_yaxis()
        fig.tight_layout()
        figure = run / "capture-means.png"
        fig.savefig(figure, dpi=160)
        plt.close(fig)
        figures.append(figure)
    for name, obj in evidence.items():
        sections.append(f'<section><h2>{_escape(name)}</h2><p><a href="{_escape(name)}">Raw artifact</a></p>{_render(obj)}</section>')
        markdown.extend([f"## {name}", "", f"[Raw artifact]({name})", ""])
        if isinstance(obj, dict):
            for key in ("findings", "limitations", "summary", "parity"):
                if key in obj:
                    markdown.extend([f"### {key}", "", "```json", json.dumps(obj[key], indent=2, ensure_ascii=False), "```", ""])
    if compare is not None:
        other = Path(compare).resolve()
        comparisons = []
        for name, obj in evidence.items():
            path = other / name
            if not path.is_file():
                continue
            left = _numeric_leaves(obj)
            right = _numeric_leaves(json.loads(path.read_text()))
            comparisons.extend({"artifact": name, "metric": key, "current": left[key],
                                "comparison": right[key], "difference": left[key] - right[key]}
                               for key in sorted(left.keys() & right.keys()))
        sections.append(f"<section><h2>Comparison with {_escape(other.name)}</h2>"
                        "<p>Scalar differences only. Check model, prompt, selection and generation metadata before interpreting differences.</p>"
                        f"{_render(comparisons)}</section>")
        markdown.extend(["## Comparison", "", f"Compared scalar fields with `{other}`; see HTML for differences.", ""])
    downloads = [path for path in sorted(run.rglob("*")) if path.is_file()
                 and path.name not in {"report.html", "report.md"}]
    links = "".join(f'<li><a href="{_escape(path.relative_to(run).as_posix())}">{_escape(path.relative_to(run))}</a></li>' for path in downloads)
    pictures = "".join(f'<figure><img src="{p.name}" alt="Measured captured tensor means"></figure>' for p in figures)
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>NanoInterpretability — {_escape(run.name)}</title>
<style>body{{font:16px system-ui,sans-serif;margin:2rem auto;max-width:1200px;padding:0 1rem;color:#182333;background:#fafbfc}}h1,h2{{color:#163b60}}section{{margin:2rem 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;vertical-align:top;padding:.45rem;border:1px solid #ccd5df}}th{{width:15%}}pre{{white-space:pre-wrap;max-height:24rem;overflow:auto}}code{{overflow-wrap:anywhere}}details{{padding:.4rem;border-bottom:1px solid #dde3ea}}summary{{cursor:pointer}}img{{max-width:100%}}a{{color:#075d9c}}</style></head>
<body><h1>NanoInterpretability: {_escape(run.name)}</h1><p>Measured evidence and reproducible run artifacts. Expand records for tensor values, controls and source operations.</p>
{pictures}{''.join(sections)}<section><h2>All run artifacts</h2><ul>{links}</ul></section></body></html>'''
    (run / "report.html").write_text(document)
    (run / "report.md").write_text("\n".join(markdown))
    return run / "report.html"
