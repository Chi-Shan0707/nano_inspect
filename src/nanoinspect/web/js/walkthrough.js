// Step through one mixer for one token and head: every recorded operation, then the hand-recomputed example.
import { $, concept, el, esc, layerOptions, num, observations, persist, pct, presets, run, section, state, term, tok,
         tokenStrip, typeColor } from "./core.js";
import { barList, lineChart, sparkbars } from "./charts.js";

const RETRIEVAL = "Facts:\nThe red box contains the apple.\nThe blue box contains the coin.\nQuestion: What does the blue box contain?\nAnswer: The blue box contains the";

// [friendly name, formula, what it means]
const OPS = {
  attention: {
    mixer_input: ["Mixer input", "h = RMSNorm(x)", "The normalized residual stream entering the mixer. Every projection below reads this."],
    query_projection: ["Queries", "q = W_q h", "What each position is looking for: 8 heads × 256 features. Comes from the first half of each head's slice of q_proj."],
    output_gate_logits: ["Gate logits", "g = W_g h", "Second half of q_proj: one logit per output feature, used at the very end to gate the head outputs."],
    key_projection: ["Keys", "k = W_k h", "What each position offers to be found by: only 2 heads (GQA)."],
    value_projection: ["Values", "v = W_v h", "What each position hands over when attended to: 2 heads × 256."],
    query_normalized: ["QK-norm on queries", "q ← RMSNorm(q)·(1+w)", "Per-head normalization so attention logits cannot blow up."],
    key_normalized: ["QK-norm on keys", "k ← RMSNorm(k)·(1+w)", "Same for keys."],
    value: ["Values per head", "", "Values reshaped to [head, token, feature]."],
    rotary_cos: ["RoPE cosines", "cos(pos·θᵢ)", "Rotation angles per position. Only the first quarter of each head's features rotate."],
    rotary_sin: ["RoPE sines", "sin(pos·θᵢ)", "Paired with the cosines."],
    query_rotary: ["Queries after RoPE", "q′ = rotate(q, pos)", "The rotated features make q·k depend on the distance between positions; the other 192 features stay position-free."],
    key_rotary: ["Keys after RoPE", "k′ = rotate(k, pos)", "Keys rotated by their own positions."],
    key_grouped: ["Keys shared across heads (GQA)", "repeat_kv", "Each KV head is copied to its 4 query heads: heads 0–3 read KV head 0, heads 4–7 read KV head 1."],
    value_grouped: ["Values shared across heads", "repeat_kv", "Same grouping for values."],
    scores: ["Attention scores", "sᵢⱼ = q′ᵢ·k′ⱼ / √256", "How well query i matches key j, for every pair."],
    mask: ["Causal mask", "−∞ where j > i", "A token may not look at the future."],
    masked_scores: ["Masked scores", "s + mask", "Future positions become −∞ so softmax gives them 0."],
    probabilities: ["Attention probabilities", "pᵢⱼ = softmaxⱼ(sᵢⱼ)", "Each row sums to 1: how query i splits its attention across earlier keys."],
    value_aggregation: ["Weighted values", "oᵢ = Σⱼ pᵢⱼ vⱼ", "The information actually copied from the attended positions."],
    output_gate: ["Output gate", "σ(g) ∈ (0, 1)", "Input-dependent, per-feature switch on each head's output."],
    gated_output: ["Gated head outputs", "o ⊙ σ(g)", "Concatenated over heads: 8 × 256 = 2,048 features."],
    projection: ["Output projection", "Δx = W_o (o ⊙ σ(g))", "Back to 1,024 features. This vector is added to the residual stream."],
  },
  delta: {
    mixer_input: ["Mixer input", "h = RMSNorm(x)", "The normalized residual stream entering the mixer."],
    padding_masked_input: ["Padding mask", "h · mask", "Zeroes padded positions so they cannot write to memory (a no-op without padding)."],
    pre_convolution_qkv: ["q/k/v projection", "[q; k; v] = W h", "One projection for all three: 16 heads × 128 each, laid out as channels."],
    output_gate_logits: ["Output-gate logits", "z = W_z h", "Used after the readout as SiLU(z)."],
    beta_logits: ["Write-strength logits", "b = W_b h", "One per head."],
    decay_logits: ["Decay logits", "a = W_a h", "One per head."],
    convolution_history: ["Convolution cache", "", "During decoding: the last 3 inputs kept for the causal convolution."],
    post_convolution_qkv: ["After causal convolution", "SiLU(Σᵢ wᵢ x_{t−3+i})", "Each channel blends the last 4 tokens: cheap local context before the memory step."],
    beta: ["Write strength β", "β = σ(b)", "How much this token corrects the memory (0 = ignore, 1 = full correction)."],
    log_decay: ["Log retention g", "g = −exp(A_log)·softplus(a + dt_bias)", "Always ≤ 0. exp(g) is the fraction of memory kept; it depends on the token."],
    value: ["Values", "", "The 128-d value each head wants associated with its key."],
    kernel_readout: ["Kernel readout (used by the model)", "o = qᵀS (chunked kernel)", "The upstream implementation computes the recurrence in chunks of 64 tokens; the explicit steps below must agree with it."],
    query_normalized_scaled: ["Normalized query", "q / ‖q‖ / √128", "Unit-length queries scaled like attention."],
    key_normalized: ["Normalized key", "k / ‖k‖", "Unit-length keys keep the delta rule stable: writing k then reading k returns what was written."],
    state_before: ["Memory before this token", "S_{t−1}  (128 × 128)", "Everything this head has stored about the prefix."],
    state_memory_readout: ["What memory predicts", "m = kᵀ(exp(g)·S_{t−1})", "The value currently stored under this token's key, after decay."],
    state_delta: ["Prediction error", "δ = β·(v − m)", "Only the difference between the wanted value and the stored one gets written."],
    state_after: ["Memory after this token", "S_t = exp(g)S_{t−1} + k δᵀ", "A rank-one correction. The memory never grows; it is overwritten in place."],
    readout: ["Readout", "o = qᵀ S_t", "The query reads memory: an associative lookup, like attention without storing every key."],
    gated_output: ["Gated RMSNorm", "RMSNorm(o)·w ⊙ SiLU(z)", "Per-head normalization, then the output gate."],
    projection: ["Output projection", "Δx = W_out(…)", "Back to 1,024 features; added to the residual stream."],
  },
};

const AXIS_LABEL = { batch: "batch", token: "token", hidden: "feature", head: "head", kv_head: "kv head", head_feature: "head×feature",
  feature: "feature", channel: "channel", key_feature: "key feature", value_feature: "value feature", rotary_feature: "rotary feature",
  key_token: "key token", history: "history", mask_head: "mask head" };

export function initWalkthrough() {
  const root = $("#walkthrough");
  root.innerHTML = "";
  root.appendChild(el("p", "lede", "<b>Question:</b> what exactly happens to one token inside one mixer? Every intermediate tensor, with real numbers, linked to the line of upstream code that produced it."));
  root.appendChild(concept("walk", "Concept: reading a mixer as a sequence of tensor operations", `
    <p>Pick a prompt, a layer, a token position and a ${term("head")}. The mixer is run once with recording hooks. Each card shows:</p>
    <ul><li>the <b>formula</b> and its meaning;</li>
      <li>the tensor <b>shape with named axes</b>, e.g. <code>[batch 1, head 8, query 37, key 37]</code>;</li>
      <li>summary <b>statistics over the whole tensor</b>, and the first 8 values of the slice for your token and head (bars: blue positive, red negative);</li>
      <li>the exact <b>source line</b> in the pinned Transformers version.</li></ul>
    <p>Below the steps, the <b>worked example</b> recomputes the key steps by hand (FP32) and compares them with what the model recorded. Small differences are BF16 rounding.</p>
    <p><i>Head = auto</i> chooses the head whose output most raises the predicted token over the <i>compare</i> token (${term("direct logit attribution")}).</p>`));
  const form = el("form", "panel");
  form.id = "walk-form";
  form.innerHTML = `
    <label>Prompt<textarea name="prompt" rows="4"></textarea></label>
    <div class="row">
      <label>Layer <select name="layer"></select></label>
      <label>Token position <input name="position" type="number" value="-1" title="-1 = last token; click a token after running"></label>
      <label>Head <input name="head" value="auto" title="auto, or a head number"></label>
      <label>Compare token <input name="compare" placeholder="optional, e.g. ·apple"></label>
      <label class="check"><input name="chat_template" type="checkbox"> chat template</label>
      <button type="submit">Run</button>
    </div>
    <div class="presets"></div>`;
  root.appendChild(form);
  const out = el("div", "out");
  root.appendChild(out);
  form.elements.layer.innerHTML = layerOptions();
  form.elements.prompt.value = RETRIEVAL;
  form.elements.layer.value = "15";
  form.elements.compare.value = " apple";
  persist(form);
  presets($(".presets", form), form, [
    { label: "Attention: who reads ‘coin’? (layer 15)", values: { prompt: RETRIEVAL, layer: "15", position: -1, head: "auto", compare: " apple" } },
    { label: "Gated delta memory (layer 14)", values: { prompt: RETRIEVAL, layer: "14", position: -1, head: "auto", compare: " apple" } },
    { label: "Early attention layer (layer 3)", values: { prompt: "The quick brown fox jumps over the lazy dog. The quick brown fox", layer: "3", position: -1, head: "auto", compare: "" } },
  ]);
  form.onsubmit = event => {
    event.preventDefault();
    const body = Object.fromEntries(new FormData(form));
    body.chat_template = form.elements.chat_template.checked;
    run(form, out, "/api/walkthrough", body, data => render(data, form, out));
  };
}

function shapeText(op) {
  return `[${op.shape.map((n, i) => `${AXIS_LABEL[op.axes[i]] || op.axes[i]} ${n}`).join(", ")}]`;
}

function render(d, form, out) {
  const kind = d.kind === "full_attention" ? "attention" : "delta";
  const ex = d.example;
  const sel = d.selected;
  out.appendChild(el("div", "result-head", `<span class="dot" style="background:${typeColor(d.kind)}"></span>
    <b>Layer ${d.layer}</b> · ${d.kind === "full_attention" ? "full attention" : "gated delta"} · head <b>${d.head}</b> of ${d.heads}
    ${d.head_chosen === "auto" ? `<span class="muted">(auto: largest direct effect${sel[1] ? ` on ${tok(sel[0].token)} vs ${tok(sel[1].token)}` : ""})</span>` : ""}
    · token <b>${d.position}</b> ${tok(d.tokens[d.position])} · model predicts ${tok(sel[0].token)} · ${d.elapsed_seconds}s`));
  out.appendChild(el("p", "caption", "Click a token to step through the same layer at that position."));
  out.appendChild(tokenStrip(d.tokens, { selected: d.position, onPick: i => { form.elements.position.value = i; form.requestSubmit(); } }));

  out.appendChild(observations(summarize(d, kind)));

  const steps = section("Operations, in execution order", `Shapes show named axes. Stats cover the whole tensor; the bars show the first 8 values of the slice for token ${d.position}${kind === "attention" ? `, head ${d.head}` : `, head ${d.head}`}.`);
  const list = el("ol", "ops");
  d.operations.forEach(op => {
    const [name, formula, meaning] = OPS[kind][op.name] || [op.name, "", ""];
    const li = el("li", "op");
    li.innerHTML = `
      <div class="op-head"><span class="op-name">${esc(name)}</span> <code class="muted">${esc(op.name)}</code>
        ${formula ? `<code class="formula-inline">${esc(formula)}</code>` : ""}
        <a class="src" href="${esc(op.source_url)}" target="_blank" rel="noopener" title="Pinned upstream source">line ${op.source_line} ↗</a></div>
      <div class="op-body">
        <div>${esc(meaning)}</div>
        <div class="op-meta"><span class="mono">${esc(shapeText(op))}</span>
          <span class="muted">mean ${num(op.mean)} · std ${num(op.std)} · min ${num(op.min)} · max ${num(op.max)}</span></div>
        <div class="op-slice">${sparkbars(op.slice)} <span class="mono muted">${op.slice.slice(0, 4).map(v => num(v, 3)).join(", ")}…</span>
          ${op.inputs.length ? `<span class="muted">← ${op.inputs.map(esc).join(", ")}</span>` : ""}</div>
      </div>`;
    list.appendChild(li);
  });
  steps.appendChild(list);
  out.appendChild(steps);

  const worked = section("Worked example: the key steps recomputed by hand", "Manual FP32 recomputation from recorded inputs next to the recorded value. Agreement shows the formula above is what the model ran.");
  worked.appendChild(el("div", "worked", ex.steps.map(step => `<div class="wstep"><div class="wtitle">${esc(step.step)}</div>${stepBody(step)}</div>`).join("")));
  out.appendChild(worked);

  if (kind === "attention") {
    const probs = ex.probabilities_over_keys.slice().sort((a, b) => b.probability - a.probability).slice(0, 10);
    const s = section(`Where head ${d.head} looks from token ${d.position}`, "Top 10 key positions by attention probability.");
    s.insertAdjacentHTML("beforeend", barList(probs.map(p => ({ label: `<span class="mono">${p.key_position} ${tok(p.key_token)}</span>`, value: p.probability })), { format: pct, max: 1 }));
    out.appendChild(s);
  } else {
    const s = section(`Head ${d.head}: memory over the prompt`, "Frobenius norm of the state after each token, and the per-token gates. Drops in retention flush older content.");
    out.appendChild(s);
    const labels = ex.controls_over_tokens.map(r => String(r.position));
    lineChart(s, { labels, xLabel: "token position", yLabel: "State norm ‖S‖ after each token", series: [{ name: "‖S‖", values: ex.state_trajectory.map(r => r.state_frobenius_norm) }],
      format: v => v.toFixed(2), tip: i => `token ${i} ${JSON.stringify(ex.controls_over_tokens[i].token)}\n‖S‖ = ${ex.state_trajectory[i].state_frobenius_norm.toFixed(3)}` });
    lineChart(s, { labels, xLabel: "token position", yLabel: "Gates per token (both in 0–1)", domain: [0, 1],
      series: [{ name: "retention exp(g)", values: ex.controls_over_tokens.map(r => r.decay) }, { name: "write strength β", values: ex.controls_over_tokens.map(r => r.beta) }],
      tip: i => `token ${i} ${JSON.stringify(ex.controls_over_tokens[i].token)}\nretention ${ex.controls_over_tokens[i].decay.toFixed(4)}\nβ ${ex.controls_over_tokens[i].beta.toFixed(4)}` });
  }

  const heads = ex.head_contributions;
  if (heads?.length) {
    const hasDiff = heads[0].direct_logit_difference != null;
    const s = section("Every head's direct effect on the prediction", hasDiff
      ? `Direct logit attribution of each head's output (through the frozen final norm): how much it raises ${tok(sel[0].token)} over ${tok(sel[1].token)}. Positive bars favour the prediction.`
      : `Size of each head's contribution to the residual stream (give a compare token to see a signed effect on the prediction).`);
    s.insertAdjacentHTML("beforeend", barList(heads.map(h => ({ label: `head ${h.head}${h.head === d.head ? " ◀" : ""}`,
      value: hasDiff ? h.direct_logit_difference : h.contribution_norm, note: hasDiff ? `‖out‖ ${h.contribution_norm.toFixed(2)}` : "" })),
      { format: v => (hasDiff && v > 0 ? "+" : "") + v.toFixed(3) }));
    out.appendChild(s);
  }
}

function summarize(d, kind) {
  const ex = d.example, items = [];
  const heads = ex.head_contributions || [];
  if (kind === "attention") {
    const top = ex.probabilities_over_keys.slice().sort((a, b) => b.probability - a.probability);
    items.push(`Head ${d.head} at token ${d.position} puts <b>${pct(top[0].probability)}</b> of its attention on position ${top[0].key_position} ${tok(top[0].key_token)}` +
      (top[1] ? `, then ${pct(top[1].probability)} on ${top[1].key_position} ${tok(top[1].key_token)}.` : "."));
    const sink = ex.probabilities_over_keys[0]?.probability ?? 0;
    if (sink > 0.3 && top[0].key_position !== 0) items.push(`${pct(sink)} goes to position 0, the ${term("attention sink")}: a default target when nothing else matches.`);
    if (top[0].key_position === 0) items.push(`Most attention lands on position 0 (an ${term("attention sink")}). This head probably has little to contribute at this token.`);
    const gate = ex.steps.find(s => s.mean_gate != null);
    if (gate) items.push(`Its output gate averages <b>${gate.mean_gate.toFixed(2)}</b> (1 = fully open). The gate scales what the head copies before it reaches the residual stream.`);
  } else {
    const ctl = ex.steps.find(s => s.decay != null);
    if (ctl) {
      const half = ctl.decay < 1 ? Math.log(0.5) / Math.log(ctl.decay) : Infinity;
      items.push(`At token ${d.position}, head ${d.head} keeps <b>${(ctl.decay * 100).toFixed(2)}%</b> of its memory (half-life ≈ ${Number.isFinite(half) ? half.toFixed(0) : "∞"} tokens) and writes with β = <b>${ctl.beta.toFixed(3)}</b>.`);
    }
    const traj = ex.state_trajectory;
    if (traj?.length > 1) {
      const peak = traj.reduce((m, r) => (r.state_frobenius_norm > m.state_frobenius_norm ? r : m));
      items.push(`The memory norm goes from ${traj[0].state_frobenius_norm.toFixed(2)} after token 0 to ${traj[traj.length - 1].state_frobenius_norm.toFixed(2)}; peak ${peak.state_frobenius_norm.toFixed(2)} at token ${peak.position}.`);
    }
    const err = ex.steps.find(s => s.recurrence_vs_kernel_max_abs_error != null);
    if (err) items.push(`The explicit token-by-token recurrence matches the model's chunked kernel to within ${err.recurrence_vs_kernel_max_abs_error.toExponential(1)}.`);
  }
  if (heads.length && heads[0].direct_logit_difference != null) {
    const ranked = heads.slice().sort((a, b) => b.direct_logit_difference - a.direct_logit_difference);
    items.push(`Largest direct push toward ${tok(d.selected[0].token)}: head ${ranked[0].head} (+${ranked[0].direct_logit_difference.toFixed(2)} logits); strongest push the other way: head ${ranked[ranked.length - 1].head} (${ranked[ranked.length - 1].direct_logit_difference.toFixed(2)}).`);
  }
  return items;
}

function stepBody(step) {
  const skip = new Set(["step"]);
  const parts = [];
  for (const [k, v] of Object.entries(step)) {
    if (skip.has(k)) continue;
    const label = k.replace(/_/g, " ");
    if (Array.isArray(v) && v.length && typeof v[0] === "object") {
      const cols = Object.keys(v[0]);
      parts.push(`<table class="mini"><tr>${cols.map(c => `<th>${esc(c.replace(/_/g, " "))}</th>`).join("")}</tr>${v.map(row =>
        `<tr>${cols.map(c => `<td class="${typeof row[c] === "number" ? "num" : "mono"}">${typeof row[c] === "number" ? num(row[c], 4) : tok(row[c])}</td>`).join("")}</tr>`).join("")}</table>`);
    } else if (Array.isArray(v)) {
      parts.push(`<div class="kv"><span>${esc(label)}</span><span class="mono">[${v.map(x => typeof x === "number" ? num(x, 4) : esc(x)).join(", ")}]</span></div>`);
    } else {
      parts.push(`<div class="kv"><span>${esc(label)}</span><span class="mono">${typeof v === "number" ? num(v, 5) : esc(v)}</span></div>`);
    }
  }
  return parts.join("");
}
