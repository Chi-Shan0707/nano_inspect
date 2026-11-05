// Architecture: config, layer schedule, per-layer pipeline with shapes, parameters, cache scaling, notes.
import { $, api, concept, el, esc, fmtBytes, fmtInt, observations, section, state, term, TYPE_LABEL, typeColor } from "./core.js";
import { lineChart } from "./charts.js";
import { markdown } from "./markdown.js";

// Plain-language meaning of each observed module, keyed by its path inside the block.
const MODULE_NOTES = {
  input_layernorm: ["Input RMSNorm", "Rescales the residual stream to unit RMS, then applies a learned per-feature gain (1 + w). The mixer sees this normalized copy; the residual itself is untouched."],
  "self_attn.q_proj": ["Query + gate projection", "One matrix produces, per head, a 256-d query and a 256-d output-gate logit (8 heads × 2 × 256 = 4,096)."],
  "self_attn.q_norm": ["Query RMSNorm (QK-norm)", "Normalizes each head's query separately; keeps attention logits in a stable range."],
  "self_attn.k_proj": ["Key projection", "Only 2 key heads (GQA): 2 × 256 = 512 features."],
  "self_attn.k_norm": ["Key RMSNorm", "Per-head normalization of keys, matching the queries."],
  "self_attn.v_proj": ["Value projection", "2 value heads × 256 = 512 features: what each position offers when attended to."],
  "self_attn.o_proj": ["Output projection", "Concatenated gated head outputs (8 × 256 = 2,048) back to the 1,024-d residual width."],
  self_attn: ["Full-attention mixer (output)", "Its output is added to the residual stream."],
  "linear_attn.in_proj_qkv": ["q/k/v projection", "One matrix gives queries, keys and values for 16 heads × 128 each (3 × 2,048 = 6,144 features)."],
  "linear_attn.in_proj_z": ["Output-gate projection", "z (16 heads × 128): the SiLU(z) gate applied after the memory readout."],
  "linear_attn.in_proj_b": ["Write-strength logits", "One number per head and token; β = sigmoid(b) decides how strongly this token writes."],
  "linear_attn.in_proj_a": ["Decay logits", "One number per head and token; sets the retention exp(g) of the memory."],
  "linear_attn.norm": ["Gated RMSNorm", "Normalizes each head's 128-d readout and multiplies by SiLU(z). Runs on (tokens × heads) rows, hence the flattened shape."],
  "linear_attn.out_proj": ["Output projection", "16 heads × 128 = 2,048 → 1,024 residual width."],
  linear_attn: ["Gated-delta mixer (output)", "Its output is added to the residual stream."],
  post_attention_layernorm: ["Post-mixer RMSNorm", "Normalizes h = x + mixer output before the MLP."],
  "mlp.gate_proj": ["MLP gate branch", "1,024 → 3,584; passed through SiLU."],
  "mlp.act_fn": ["SiLU", "x · sigmoid(x): a smooth gate."],
  "mlp.up_proj": ["MLP up branch", "1,024 → 3,584; multiplied elementwise by SiLU(gate)."],
  "mlp.down_proj": ["MLP down projection", "3,584 → 1,024, added to the residual stream."],
  mlp: ["SwiGLU MLP (output)", "Per-position transformation; no mixing between tokens."],
};

// Operations that run inside the mixer without their own module (so the inventory hook cannot observe them).
const HIDDEN_STEPS = {
  full_attention: {
    after: "self_attn.v_proj",
    steps: c => [
      ["Partial RoPE", `Rotates the first ${Math.round(c.head_dim * (c.rope_parameters?.partial_rotary_factor ?? 1))} of ${c.head_dim} features of each q and k by position-dependent angles (θ = ${fmtInt(c.rope_parameters?.rope_theta)}).`, "[token, head, 256] → same"],
      ["Grouped scores + causal softmax", `Each of ${c.num_attention_heads} query heads uses KV head ⌊h / ${c.num_attention_heads / c.num_key_value_heads}⌋. score = q·k/√${c.head_dim}; future positions masked; softmax over keys.`, "→ [head 8, query, key]"],
      ["Weighted sum of values", "o = Σⱼ pⱼ vⱼ for every head.", "→ [token, 8 × 256]"],
      ["Output gate", "o ⊙ sigmoid(gate): per feature, input-dependent.", "→ [token, 2,048]"],
    ],
  },
  linear_attention: {
    after: "linear_attn.in_proj_a",
    steps: c => [
      ["Causal depthwise conv + SiLU", `Each of the 6,144 q/k/v channels mixes the last ${c.linear_conv_kernel_dim} tokens (functional call on conv1d.weight).`, "[token, 6,144] → same"],
      ["Normalize q, k", `L2-normalize per head; scale q by 1/√${c.linear_key_head_dim}.`, "→ 16 heads × 128"],
      ["Gates", "β = sigmoid(b) ∈ (0,1); g = −exp(A_log)·softplus(a + dt_bias) ≤ 0; retention exp(g).", "→ [token, 16]"],
      ["Delta-rule memory", "Per head: S ← exp(g)·S; S ← S + β·k·(v − kᵀS)ᵀ; read o = qᵀS. S is a 128×128 matrix carried across tokens.", "→ [token, 16, 128]"],
    ],
  },
};

export async function renderArchitecture() {
  const root = $("#architecture");
  const a = await api("/api/architecture");
  state.arch = a;
  state.layerTypes = (a.layers || []).map(l => l.type);
  const c = a.config || {};
  root.innerHTML = "";
  root.appendChild(el("p", "lede", `<b>Question:</b> what is inside the model, how big is each part, and how does the data change shape on its way through one block?`));
  root.appendChild(concept("arch", "Concept: a hybrid stack of residual blocks", `
    <p>The text model is <code>embedding → 24 blocks → final ${term("RMSNorm")} → ${term("unembedding")}</code>. Every block adds a ${term("token mixer")} update and an MLP update to the ${term("residual stream")}. Blocks differ only in their mixer:</p>
    <ul>
      <li><b style="color:${typeColor("linear_attention")}">Gated delta</b> (layers ≠ 3 mod 4): a recurrent ${term("delta rule")} memory per head. Fixed size, so long contexts cost the same per token.</li>
      <li><b style="color:${typeColor("full_attention")}">Full attention</b> (layers 3, 7, …, 23): softmax attention with ${term("GQA")}, ${term("RoPE")} and an ${term("output gate")}. Exact lookup, with a ${term("KV cache")} that grows.</li>
    </ul>
    <p>Weight shapes are stored as <code>[out, in]</code>: a <code>[4096, 1024]</code> matrix maps 1,024 inputs to 4,096 outputs. Shapes below in the pipeline are <i>observed</i> on a 5-token input, so the token axis reads 5.</p>`));

  const cards = [
    ["Text layers", c.num_hidden_layers, "residual blocks"], ["Residual width", fmtInt(c.hidden_size), "numbers per token"],
    ["MLP width", fmtInt(c.intermediate_size), "hidden units per MLP"], ["Vocabulary", fmtInt(c.vocab_size), "tokens"],
    ["Attention heads", `${c.num_attention_heads} q / ${c.num_key_value_heads} kv`, `${c.head_dim}-d each (GQA ${c.num_attention_heads / c.num_key_value_heads}:1)`],
    ["Delta heads", `${c.linear_num_value_heads}`, `${c.linear_key_head_dim}×${c.linear_value_head_dim} memory each`],
    ["Conv kernel", c.linear_conv_kernel_dim, "tokens of local mixing"],
    ["RoPE", `${(c.rope_parameters?.partial_rotary_factor ?? 1) * 100}% dims`, `θ = ${fmtInt(c.rope_parameters?.rope_theta)}`],
    ["Embeddings", c.tie_word_embeddings ? "tied" : "untied", "input = output matrix"],
  ];
  root.appendChild(el("div", "cards", cards.map(([k, v, s]) => `<div class="card stat"><div class="k">${esc(k)}</div><div class="v">${esc(v ?? "–")}</div><div class="s">${esc(s)}</div></div>`).join("")));
  if (!a.inventory) root.appendChild(el("p", "hint", "No inventory found; showing the config only. Run <code>nanoinspect inventory</code> for weights, dataflow and accounting."));

  // Schedule
  const sched = section("Layer schedule", "Click a layer to see its pipeline. Colour marks the mixer type; the pattern repeats every 4 layers.");
  const strip = el("div", "schedule");
  (a.layers || []).forEach(layer => {
    const b = el("button", "", String(layer.index));
    b.type = "button"; b.style.background = typeColor(layer.type);
    b.title = `layer ${layer.index}: ${TYPE_LABEL[layer.type] || layer.type}`;
    b.onclick = () => selectLayer(layer.index);
    strip.appendChild(b);
  });
  sched.appendChild(strip);
  sched.appendChild(el("div", "legend", `<span><i style="background:${typeColor("linear_attention")}"></i>gated delta: fixed-size recurrent memory</span>` +
    `<span><i style="background:${typeColor("full_attention")}"></i>full attention: KV cache grows with length</span>`));
  sched.appendChild(el("div", "", "")).id = "arch-layer";
  root.appendChild(sched);

  if (a.accounting) root.appendChild(renderParams(a));
  if (a.caches) renderCache(a, root);
  if (a.svgs?.length) {
    const diagrams = section("Block diagrams", "Generated by <code>nanoinspect inventory</code> from the loaded weights. Click to open full size.");
    diagrams.appendChild(el("div", "svgs", a.svgs.map(s => `<a href="${s}" target="_blank"><img src="${s}" alt="${esc(s)}"></a>`).join("")));
    root.appendChild(diagrams);
  }
  if (a.notes) {
    const notes = section("Reading notes", "From <code>docs/architecture.md</code>.");
    notes.appendChild(el("article", "notes prose", markdown(a.notes)));
    root.appendChild(notes);
  }
  if (a.layers?.[0]?.weight_shapes) selectLayer(3);
}

function selectLayer(index) {
  const layer = state.arch.layers[index], c = state.arch.config;
  document.querySelectorAll("#architecture .schedule button").forEach((b, i) => b.classList.toggle("selected", i === index));
  const target = $("#arch-layer");
  if (!layer.weight_shapes) { target.innerHTML = ""; return; }
  const params = name => Object.entries(layer.weight_shapes).filter(([w]) => w.startsWith(name + ".")).reduce((s, [, sh]) => s + sh.reduce((p, x) => p * x, 1), 0);
  const shape = s => s ? `[${s.join(", ")}]` : "–";
  const hidden = HIDDEN_STEPS[layer.type];
  const rows = [];
  for (const f of layer.dataflow) {
    if (f.module.startsWith("model.")) continue;  // the block itself
    const [title, note] = MODULE_NOTES[f.module] || [f.module, ""];
    const p = params(f.module);
    const isSummary = !f.module.includes(".") && ["self_attn", "linear_attn", "mlp"].includes(f.module);
    rows.push(`<div class="pstep${isSummary ? " summary" : ""}"><div class="pname">${esc(title)} <code>${esc(f.module)}</code></div>
      <div class="pshape">${f.input.map(shape).join(" + ") || "–"} → ${shape(f.output)}</div>
      <div class="pnote">${esc(note)}${p ? ` <span class="muted">· ${fmtInt(p)} params</span>` : ""}</div></div>`);
    if (hidden && f.module === hidden.after) {
      for (const [t, n, s] of hidden.steps(c)) rows.push(`<div class="pstep hidden-op"><div class="pname">${esc(t)} <span class="tag">inside the mixer</span></div><div class="pshape">${esc(s)}</div><div class="pnote">${esc(n)}</div></div>`);
    }
  }
  const mixer = layer.type === "full_attention" ? "self_attn" : "linear_attn";
  const weights = Object.entries(layer.weight_shapes).map(([name, s]) =>
    `<tr><td class="mono">${esc(name)}</td><td class="mono">[${s.join(", ")}]</td><td class="num">${fmtInt(s.reduce((p, x) => p * x, 1))}</td></tr>`).join("");
  target.innerHTML = `
    <div class="layer-head"><span class="dot" style="background:${typeColor(layer.type)}"></span>
      <b>Layer ${index}</b> · ${esc(TYPE_LABEL[layer.type])} · <code>${esc(layer.mixer_class)}</code> ·
      ${fmtInt(layer.parameters)} params (mixer ${fmtInt(params(mixer))}, MLP ${fmtInt(params("mlp"))}, norms ${fmtInt(params("input_layernorm") + params("post_attention_layernorm"))})</div>
    <p class="caption">Execution order, top to bottom. Shapes are [batch, tokens, features] from a 5-token probe. Grey steps happen inside the mixer without a separate module.</p>
    <div class="pipeline">${rows.join("")}</div>
    <details class="raw"><summary>All weight tensors of this layer</summary>
      <div class="scroll"><table><tr><th>Tensor</th><th>Shape [out, in]</th><th class="num">Elements</th></tr>${weights}</table></div></details>`;
}

function renderParams(a) {
  const acc = a.accounting, c = a.config;
  const comps = Object.entries(acc.by_component || {}).sort((x, y) => y[1].parameters - x[1].parameters);
  const total = acc.loaded_unique_parameters;
  const label = { embedding: "Embedding (tied with output)", text_mixer: "Token mixers (24)", text_mlp: "MLPs (24)", vision: "Vision tower (not used for text)", other: "Norms & other" };
  const s = section("Where the parameters are", `Unique loaded parameters: ${fmtInt(total)}. The tied embedding is counted once.`);
  const max = Math.max(...comps.map(([, v]) => v.parameters));
  s.appendChild(el("div", "barlist", comps.map(([k, v]) =>
    `<div class="barrow"><span class="bl">${esc(label[k] || k)}</span><span class="bt"><span class="bf" style="left:0;width:${(v.parameters / max * 100).toFixed(1)}%"></span></span>` +
    `<span class="bv">${fmtInt(v.parameters)} <span class="muted">${(v.parameters / total * 100).toFixed(1)}%</span></span></div>`).join("")));
  const layers = a.layers || [];
  const attn = layers.find(l => l.type === "full_attention"), delta = layers.find(l => l.type === "linear_attention");
  const textActive = acc.active_text_unique_parameters;
  const emb = acc.by_component.embedding.parameters;
  s.appendChild(observations([
    `The embedding matrix (${fmtInt(c.vocab_size)} × ${fmtInt(c.hidden_size)}) is <b>${(emb / textActive * 100).toFixed(0)}%</b> of the text path. The vocabulary is huge relative to the width; in large models this share is small.`,
    attn && delta ? `A gated-delta block has <b>${fmtInt(delta.parameters)}</b> parameters, a full-attention block <b>${fmtInt(attn.parameters)}</b>. The difference is in the mixer; every MLP is identical (${fmtInt(3 * c.hidden_size * c.intermediate_size)}).` : null,
    `The checkpoint holds ${fmtInt(acc.checkpoint_tensor_parameters)} parameters. ${fmtInt(acc.reconciliation?.checkpoint_only_parameters)} belong to a multi-token-prediction head that the loader skips, so they never run.`,
    `Text-only inference executes ${fmtInt(textActive)} parameters; the ${fmtInt(acc.inactive_unique_parameters)}-parameter vision tower stays idle.`,
  ]));
  return s;
}

function renderCache(a, root) {
  const caches = a.caches;
  const s = section("Cache: what generation must remember", `Measured tensor storage after a prefill of each length (BF16). Not peak GPU memory.`);
  root.appendChild(s);
  s.appendChild(el("div", "scroll", `<table><tr><th class="num">Tokens</th><th class="num">KV (6 attention layers)</th><th class="num">Recurrent state (18 layers)</th><th class="num">Conv state</th><th class="num">Total</th></tr>${caches.map(c =>
    `<tr><td class="num">${c.sequence_length}</td><td class="num">${fmtBytes(c.bytes_by_kind.kv || 0)}</td><td class="num">${fmtBytes(c.bytes_by_kind.recurrent || 0)}</td><td class="num">${fmtBytes(c.bytes_by_kind.conv || 0)}</td><td class="num">${fmtBytes(c.measured_storage_bytes)}</td></tr>`).join("")}</table>`));
  if (caches.length >= 2) {
    const first = caches[0], last = caches[caches.length - 1];
    const perToken = ((last.bytes_by_kind.kv || 0) - (first.bytes_by_kind.kv || 0)) / (last.sequence_length - first.sequence_length);
    const fixed = (first.bytes_by_kind.recurrent || 0) + (first.bytes_by_kind.conv || 0);
    const attnLayers = state.layerTypes.filter(t => t === "full_attention").length || 6;
    const allAttention = perToken / attnLayers * state.layerTypes.length;
    const lengths = Array.from({ length: 17 }, (_, i) => i * 256);  // evenly spaced: the x axis is linear
    const mib = v => v / 2 ** 20;
    lineChart(s, {
      labels: lengths.map(String), xLabel: "context length (tokens)",
      yLabel: "Cache size (MiB), extrapolated from the measured per-token slope",
      series: [{ name: "this hybrid (6 attention + 18 gated delta)", values: lengths.map(n => mib(fixed + perToken * n)) },
               { name: "hypothetical: all 24 layers full attention", values: lengths.map(n => mib(allAttention * n)), dashed: true }],
      format: v => v.toFixed(0), domain: [0, mib(allAttention * lengths[lengths.length - 1]) * 1.05],
      tip: i => `${fmtInt(lengths[i])} tokens\nhybrid: ${fmtBytes(fixed + perToken * lengths[i])}\nall attention: ${fmtBytes(allAttention * lengths[i])}`,
    });
    const crossover = Math.round(fixed / (allAttention - perToken));
    s.appendChild(observations([
      `KV cache grows by <b>${fmtInt(perToken)} bytes per token</b> = 2 (K and V) × ${state.arch.config.num_key_value_heads} KV heads × ${state.arch.config.head_dim} dims × 2 bytes × ${attnLayers} layers.`,
      `The gated-delta state is a constant <b>${fmtBytes(fixed)}</b> however long the input is.`,
      `The hybrid needs more cache than an all-attention model below about <b>${fmtInt(crossover)} tokens</b>. Beyond that it needs less. At 32k tokens it is ${fmtBytes(fixed + perToken * 32768)} versus ${fmtBytes(allAttention * 32768)}.`,
    ], "The straight-line extrapolation uses the measured slope; the hypothetical model is a formula, not a measurement."));
  }
}
