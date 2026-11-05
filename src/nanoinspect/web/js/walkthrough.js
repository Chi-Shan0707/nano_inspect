// Step through one mixer for one token and head: every recorded operation, then the hand-recomputed example.
import { $, concept, el, esc, layerOptions, num, observations, persist, pct, presets, run, section, state, term, tok,
         tokenStrip, tr, typeColor } from "./core.js";
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

// Chinese versions of the "what it means" column above.
const OPS_ZH = {
  attention: {
    mixer_input: "进入 mixer 的、归一化后的 residual stream。下面的每个投影都读取它。",
    query_projection: "每个位置在寻找什么：8 个 head × 256 个特征。来自 q_proj 中每个 head 切片的前一半。",
    output_gate_logits: "q_proj 的后一半：每个输出特征一个 logit，在最后用于给 head 输出加门控。",
    key_projection: "每个位置提供什么以便被找到：只有 2 个 head（GQA）。",
    value_projection: "每个位置被关注时交出什么：2 个 head × 256。",
    query_normalized: "逐 head 归一化，使 attention logit 不会爆炸。",
    key_normalized: "对 key 做同样的处理。",
    value: "把 value 重排为 [head, token, feature]。",
    rotary_cos: "每个位置的旋转角度。每个 head 只有前四分之一的特征参与旋转。",
    rotary_sin: "与 cosine 配对使用。",
    query_rotary: "旋转后的特征使 q·k 取决于位置之间的距离；其余 192 个特征与位置无关。",
    key_rotary: "key 按各自的位置旋转。",
    key_grouped: "每个 KV head 被复制给它的 4 个 query head：head 0–3 读取 KV head 0，head 4–7 读取 KV head 1。",
    value_grouped: "value 也做同样的分组。",
    scores: "query i 与 key j 的匹配程度，对所有位置对计算。",
    mask: "token 不能看向未来。",
    masked_scores: "未来位置变为 −∞，softmax 后为 0。",
    probabilities: "每一行和为 1：query i 如何把 attention 分配到更早的 key 上。",
    value_aggregation: "真正从被关注位置复制过来的信息。",
    output_gate: "依赖输入、逐特征的开关，作用于每个 head 的输出。",
    gated_output: "在 head 维度上拼接：8 × 256 = 2,048 个特征。",
    projection: "映射回 1,024 个特征。这个向量被加到 residual stream 上。",
  },
  delta: {
    mixer_input: "进入 mixer 的、归一化后的 residual stream。",
    padding_masked_input: "把 padding 位置清零，使其无法写入记忆（没有 padding 时不起作用）。",
    pre_convolution_qkv: "三者共用一个投影：每个 16 个 head × 128，按通道排列。",
    output_gate_logits: "在读出之后以 SiLU(z) 的形式使用。",
    beta_logits: "每个 head 一个。",
    decay_logits: "每个 head 一个。",
    convolution_history: "解码时：为 causal convolution 保留最近 3 个输入。",
    post_convolution_qkv: "每个通道混合最近 4 个 token：在记忆步骤之前廉价地引入局部上下文。",
    beta: "这个 token 对记忆的修正程度（0 = 忽略，1 = 完全修正）。",
    log_decay: "始终 ≤ 0。exp(g) 是保留下来的记忆比例；它依赖于 token。",
    value: "每个 head 希望与其 key 关联的 128 维 value。",
    kernel_readout: "上游实现以 64 个 token 为一块分块计算循环；下面的显式步骤必须与之一致。",
    query_normalized_scaled: "单位长度的 query，像 attention 一样缩放。",
    key_normalized: "单位长度的 key 使 delta rule 保持稳定：写入 k 后再读取 k，就能得到写入的内容。",
    state_before: "这个 head 关于前缀所存储的一切。",
    state_memory_readout: "衰减之后，当前存储在这个 token 的 key 下的 value。",
    state_delta: "只有期望的 value 与已存储 value 之间的差会被写入。",
    state_after: "一个秩一修正。记忆从不增长，而是原地被覆写。",
    readout: "query 读取记忆：一种联想查找，就像不存储每个 key 的 attention。",
    gated_output: "逐 head 归一化，然后施加 output gate。",
    projection: "映射回 1,024 个特征；加到 residual stream 上。",
  },
};

const AXIS_LABEL = { batch: "batch", token: "token", hidden: "feature", head: "head", kv_head: "kv head", head_feature: "head×feature",
  feature: "feature", channel: "channel", key_feature: "key feature", value_feature: "value feature", rotary_feature: "rotary feature",
  key_token: "key token", history: "history", mask_head: "mask head" };

export function initWalkthrough() {
  const root = $("#walkthrough");
  root.innerHTML = "";
  root.appendChild(el("p", "lede", tr("<b>Question:</b> what exactly happens to one token inside one mixer? Every intermediate tensor, with real numbers, linked to the line of upstream code that produced it.",
    "<b>问题：</b>一个 token 在一个 mixer 内部究竟经历了什么？每个中间张量都给出真实数值，并链接到产生它的那一行上游代码。")));
  root.appendChild(concept("walk", tr("Concept: reading a mixer as a sequence of tensor operations", "概念：把 mixer 读成一串张量操作"), tr(`
    <p>Pick a prompt, a layer, a token position and a ${term("head")}. The mixer is run once with recording hooks. Each card shows:</p>
    <ul><li>the <b>formula</b> and its meaning;</li>
      <li>the tensor <b>shape with named axes</b>, e.g. <code>[batch 1, head 8, query 37, key 37]</code>;</li>
      <li>summary <b>statistics over the whole tensor</b>, and the first 8 values of the slice for your token and head (bars: blue positive, red negative);</li>
      <li>the exact <b>source line</b> in the pinned Transformers version.</li></ul>
    <p>Below the steps, the <b>worked example</b> recomputes the key steps by hand (FP32) and compares them with what the model recorded. Small differences are BF16 rounding.</p>
    <p><i>Head = auto</i> chooses the head whose output most raises the predicted token over the <i>compare</i> token (${term("direct logit attribution")}).</p>`, `
    <p>选择一个 prompt、一层、一个 token 位置和一个 ${term("head")}。mixer 会在记录 hook 下运行一次。每张卡片显示：</p>
    <ul><li><b>公式</b>及其含义；</li>
      <li>带<b>命名轴的张量形状</b>，例如 <code>[batch 1, head 8, query 37, key 37]</code>；</li>
      <li><b>整个张量的统计量</b>，以及你所选 token 和 head 对应切片的前 8 个值（柱形：蓝色为正，红色为负）；</li>
      <li>在固定版本 Transformers 中对应的确切<b>源码行</b>。</li></ul>
    <p>在这些步骤下方，<b>手算示例</b>用 FP32 手动重算关键步骤，并与模型记录的值对比。微小差异来自 BF16 舍入。</p>
    <p><i>Head = auto</i> 会选择其输出最能把预测 token 相对 <i>compare</i> token 抬高的 head（${term("direct logit attribution")}）。</p>`)));
  const form = el("form", "panel");
  form.id = "walk-form";
  form.innerHTML = `
    <label>Prompt<textarea name="prompt" rows="4"></textarea></label>
    <div class="row">
      <label>${tr("Layer", "层")} <select name="layer"></select></label>
      <label>${tr("Token position", "Token 位置")} <input name="position" type="number" value="-1" title="${tr("-1 = last token; click a token after running", "-1 = 最后一个 token；运行后可点击 token")}"></label>
      <label>Head <input name="head" value="auto" title="${tr("auto, or a head number", "auto，或 head 编号")}"></label>
      <label>Compare token <input name="compare" placeholder="${tr("optional, e.g. ·apple", "可选，例如 ·apple")}"></label>
      <label class="check"><input name="chat_template" type="checkbox"> chat template</label>
      <button type="submit">${tr("Run", "运行")}</button>
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
    { label: tr("Attention: who reads ‘coin’? (layer 15)", "Attention：谁读取了 ‘coin’？（第 15 层）"), values: { prompt: RETRIEVAL, layer: "15", position: -1, head: "auto", compare: " apple" } },
    { label: tr("Gated delta memory (layer 14)", "Gated delta 记忆（第 14 层）"), values: { prompt: RETRIEVAL, layer: "14", position: -1, head: "auto", compare: " apple" } },
    { label: tr("Early attention layer (layer 3)", "浅层 attention（第 3 层）"), values: { prompt: "The quick brown fox jumps over the lazy dog. The quick brown fox", layer: "3", position: -1, head: "auto", compare: "" } },
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
  const vs = sel[1] ? tr(` on ${tok(sel[0].token)} vs ${tok(sel[1].token)}`, `，${tok(sel[0].token)} 对比 ${tok(sel[1].token)}`) : "";
  out.appendChild(el("div", "result-head", `<span class="dot" style="background:${typeColor(d.kind)}"></span>
    <b>${tr(`Layer ${d.layer}`, `第 ${d.layer} 层`)}</b> · ${d.kind === "full_attention" ? "full attention" : "gated delta"} · head <b>${d.head}</b> ${tr(`of ${d.heads}`, `/ ${d.heads}`)}
    ${d.head_chosen === "auto" ? `<span class="muted">(${tr(`auto: largest direct effect${vs}`, `auto：直接效应最大${vs}`)})</span>` : ""}
    · token <b>${d.position}</b> ${tok(d.tokens[d.position])} · ${tr("model predicts", "模型预测")} ${tok(sel[0].token)} · ${d.elapsed_seconds}s`));
  out.appendChild(el("p", "caption", tr("Click a token to step through the same layer at that position.", "点击某个 token，在该位置上逐步查看同一层。")));
  out.appendChild(tokenStrip(d.tokens, { selected: d.position, onPick: i => { form.elements.position.value = i; form.requestSubmit(); } }));

  out.appendChild(observations(summarize(d, kind)));

  const steps = section(tr("Operations, in execution order", "按执行顺序排列的操作"), tr(`Shapes show named axes. Stats cover the whole tensor; the bars show the first 8 values of the slice for token ${d.position}, head ${d.head}.`,
    `形状带有命名轴。统计量覆盖整个张量；柱形显示 token ${d.position}、head ${d.head} 对应切片的前 8 个值。`));
  const list = el("ol", "ops");
  d.operations.forEach(op => {
    const [name, formula, en] = OPS[kind][op.name] || [op.name, "", ""];
    const meaning = tr(en, OPS_ZH[kind][op.name] || en);
    const li = el("li", "op");
    li.innerHTML = `
      <div class="op-head"><span class="op-name">${esc(name)}</span> <code class="muted">${esc(op.name)}</code>
        ${formula ? `<code class="formula-inline">${esc(formula)}</code>` : ""}
        <a class="src" href="${esc(op.source_url)}" target="_blank" rel="noopener" title="${tr("Pinned upstream source", "固定版本的上游源码")}">${tr("line", "第")} ${op.source_line}${tr("", " 行")} ↗</a></div>
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

  const worked = section(tr("Worked example: the key steps recomputed by hand", "手算示例：手动重算关键步骤"), tr("Manual FP32 recomputation from recorded inputs next to the recorded value. Agreement shows the formula above is what the model ran.",
    "根据记录的输入用 FP32 手动重算，并与记录值并列对比。两者一致说明上面的公式正是模型实际运行的计算。"));
  worked.appendChild(el("div", "worked", ex.steps.map(step => `<div class="wstep"><div class="wtitle">${esc(step.step)}</div>${stepBody(step)}</div>`).join("")));
  out.appendChild(worked);

  if (kind === "attention") {
    const probs = ex.probabilities_over_keys.slice().sort((a, b) => b.probability - a.probability).slice(0, 10);
    const s = section(tr(`Where head ${d.head} looks from token ${d.position}`, `head ${d.head} 从 token ${d.position} 看向哪里`), tr("Top 10 key positions by attention probability.", "按 attention 概率排序的前 10 个 key 位置。"));
    s.insertAdjacentHTML("beforeend", barList(probs.map(p => ({ label: `<span class="mono">${p.key_position} ${tok(p.key_token)}</span>`, value: p.probability })), { format: pct, max: 1 }));
    out.appendChild(s);
  } else {
    const s = section(tr(`Head ${d.head}: memory over the prompt`, `Head ${d.head}：整个 prompt 上的记忆`), tr("Frobenius norm of the state after each token, and the per-token gates. Drops in retention flush older content.",
      "每个 token 之后状态的 Frobenius 范数，以及每个 token 的门控值。retention 的下降会冲刷掉较早的内容。"));
    out.appendChild(s);
    const labels = ex.controls_over_tokens.map(r => String(r.position));
    lineChart(s, { labels, xLabel: tr("token position", "token 位置"), yLabel: tr("State norm ‖S‖ after each token", "每个 token 之后的状态范数 ‖S‖"), series: [{ name: "‖S‖", values: ex.state_trajectory.map(r => r.state_frobenius_norm) }],
      format: v => v.toFixed(2), tip: i => `token ${i} ${JSON.stringify(ex.controls_over_tokens[i].token)}\n‖S‖ = ${ex.state_trajectory[i].state_frobenius_norm.toFixed(3)}` });
    lineChart(s, { labels, xLabel: tr("token position", "token 位置"), yLabel: tr("Gates per token (both in 0–1)", "每个 token 的门控值（均在 0–1 之间）"), domain: [0, 1],
      series: [{ name: "retention exp(g)", values: ex.controls_over_tokens.map(r => r.decay) }, { name: "write strength β", values: ex.controls_over_tokens.map(r => r.beta) }],
      tip: i => `token ${i} ${JSON.stringify(ex.controls_over_tokens[i].token)}\nretention ${ex.controls_over_tokens[i].decay.toFixed(4)}\nβ ${ex.controls_over_tokens[i].beta.toFixed(4)}` });
  }

  const heads = ex.head_contributions;
  if (heads?.length) {
    const hasDiff = heads[0].direct_logit_difference != null;
    const s = section(tr("Every head's direct effect on the prediction", "每个 head 对预测的直接效应"), hasDiff
      ? tr(`Direct logit attribution of each head's output (through the frozen final norm): how much it raises ${tok(sel[0].token)} over ${tok(sel[1].token)}. Positive bars favour the prediction.`,
           `每个 head 输出的 direct logit attribution（经过冻结的 final norm）：它把 ${tok(sel[0].token)} 相对 ${tok(sel[1].token)} 抬高了多少。正值柱形支持该预测。`)
      : tr(`Size of each head's contribution to the residual stream (give a compare token to see a signed effect on the prediction).`,
           `每个 head 对 residual stream 贡献的大小（填写 compare token 可看到对预测的带符号效应）。`));
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
    const first = `${top[0].key_position} ${tok(top[0].key_token)}`, second = top[1] ? `${top[1].key_position} ${tok(top[1].key_token)}` : null;
    items.push(tr(`Head ${d.head} at token ${d.position} puts <b>${pct(top[0].probability)}</b> of its attention on position ${first}` +
      (second ? `, then ${pct(top[1].probability)} on ${second}.` : "."),
      `Head ${d.head} 在 token ${d.position} 处把 <b>${pct(top[0].probability)}</b> 的 attention 放在位置 ${first} 上` +
      (second ? `，其次是 ${second}（${pct(top[1].probability)}）。` : "。")));
    const sink = ex.probabilities_over_keys[0]?.probability ?? 0;
    if (sink > 0.3 && top[0].key_position !== 0) items.push(tr(`${pct(sink)} goes to position 0, the ${term("attention sink")}: a default target when nothing else matches.`,
      `${pct(sink)} 流向位置 0，即 ${term("attention sink")}：没有其他匹配时的默认目标。`));
    if (top[0].key_position === 0) items.push(tr(`Most attention lands on position 0 (an ${term("attention sink")}). This head probably has little to contribute at this token.`,
      `大部分 attention 落在位置 0（${term("attention sink")}）。这个 head 在该 token 上可能没有多少贡献。`));
    const gate = ex.steps.find(s => s.mean_gate != null);
    if (gate) items.push(tr(`Its output gate averages <b>${gate.mean_gate.toFixed(2)}</b> (1 = fully open). The gate scales what the head copies before it reaches the residual stream.`,
      `它的 output gate 平均为 <b>${gate.mean_gate.toFixed(2)}</b>（1 = 完全打开）。在 head 复制的内容进入 residual stream 之前，门会对其进行缩放。`));
  } else {
    const ctl = ex.steps.find(s => s.decay != null);
    if (ctl) {
      const half = ctl.decay < 1 ? Math.log(0.5) / Math.log(ctl.decay) : Infinity;
      const kept = (ctl.decay * 100).toFixed(2), hl = Number.isFinite(half) ? half.toFixed(0) : "∞";
      items.push(tr(`At token ${d.position}, head ${d.head} keeps <b>${kept}%</b> of its memory (half-life ≈ ${hl} tokens) and writes with β = <b>${ctl.beta.toFixed(3)}</b>.`,
        `在 token ${d.position} 处，head ${d.head} 保留 <b>${kept}%</b> 的记忆（half-life ≈ ${hl} 个 token），并以 β = <b>${ctl.beta.toFixed(3)}</b> 写入。`));
    }
    const traj = ex.state_trajectory;
    if (traj?.length > 1) {
      const peak = traj.reduce((m, r) => (r.state_frobenius_norm > m.state_frobenius_norm ? r : m));
      const a = traj[0].state_frobenius_norm.toFixed(2), b = traj[traj.length - 1].state_frobenius_norm.toFixed(2), m = peak.state_frobenius_norm.toFixed(2);
      items.push(tr(`The memory norm goes from ${a} after token 0 to ${b}; peak ${m} at token ${peak.position}.`,
        `记忆范数从 token 0 之后的 ${a} 变化到 ${b}；峰值 ${m} 出现在 token ${peak.position}。`));
    }
    const err = ex.steps.find(s => s.recurrence_vs_kernel_max_abs_error != null);
    if (err) items.push(tr(`The explicit token-by-token recurrence matches the model's chunked kernel to within ${err.recurrence_vs_kernel_max_abs_error.toExponential(1)}.`,
      `显式的逐 token 循环计算与模型的分块 kernel 一致，误差在 ${err.recurrence_vs_kernel_max_abs_error.toExponential(1)} 以内。`));
  }
  if (heads.length && heads[0].direct_logit_difference != null) {
    const ranked = heads.slice().sort((a, b) => b.direct_logit_difference - a.direct_logit_difference);
    const lo = ranked[ranked.length - 1];
    items.push(tr(`Largest direct push toward ${tok(d.selected[0].token)}: head ${ranked[0].head} (+${ranked[0].direct_logit_difference.toFixed(2)} logits); strongest push the other way: head ${lo.head} (${lo.direct_logit_difference.toFixed(2)}).`,
      `最直接推向 ${tok(d.selected[0].token)} 的是 head ${ranked[0].head}（+${ranked[0].direct_logit_difference.toFixed(2)} logits）；反方向推得最强的是 head ${lo.head}（${lo.direct_logit_difference.toFixed(2)}）。`));
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
