// Mixer maps: all heads of one layer. Full attention -> softmax maps; gated delta -> gates and memory over tokens.
import { $, concept, el, esc, layerOptions, observations, persist, pct, plainTok, presets, run, section, segmented, term, tok, tr, typeColor } from "./core.js";
import { barList, heatmap, lineChart } from "./charts.js";

const RETRIEVAL = "Facts:\nThe red box contains the apple.\nThe blue box contains the coin.\nQuestion: What does the blue box contain?\nAnswer: The blue box contains the";
const REPEAT = "The violet giraffe ate seven lemons. Then the violet giraffe ate seven";

export function initMixer() {
  const root = $("#mixer");
  root.innerHTML = "";
  root.appendChild(el("p", "lede", tr("<b>Question:</b> across all heads of a layer, which earlier tokens does each position draw on, and how does recurrent memory evolve?",
    "<b>问题：</b>在一层的所有 head 中，每个位置从哪些更早的 token 取信息？循环记忆又是如何演化的？")));
  root.appendChild(concept("mixer", tr("Concept: attention patterns vs recurrent memory", "概念：attention pattern 与循环记忆"), tr(`
    <div class="grid2"><div>
      <h4 style="color:${typeColor("full_attention")}">Full-attention layers</h4>
      <p>Each ${term("head")} produces a matrix p[query, key]: row i shows how token i splits its attention over tokens 0…i (rows sum to 1). Common patterns:</p>
      <ul><li><b>${term("attention sink")}</b>: a bright first column; the head is mostly idle.</li>
        <li><b>previous-token</b>: a bright diagonal just below the main one; it builds bigram context.</li>
        <li><b>content / retrieval</b>: isolated bright cells far from the diagonal; it fetches a specific earlier token.</li>
        <li><b>induction</b>: on repeated text, token i attends to the token <i>after</i> the earlier copy of itself.</li></ul>
    </div><div>
      <h4 style="color:${typeColor("linear_attention")}">Gated-delta layers</h4>
      <p>There is <b>no attention matrix</b>: the past is compressed into a 128×128 memory per head. What you can see is how that memory is <i>managed</i>, token by token:</p>
      <ul><li><b>retention exp(g)</b>: fraction of memory kept (its ${term("half-life")} says how far back the head can see);</li>
        <li><b>write strength β</b>: how much this token corrects the memory;</li>
        <li><b>‖S‖</b>: how much the memory holds; <b>write size</b>: how big this token's correction is.</li></ul>
      <p>Values come from an explicit FP32 recurrence run alongside the model's kernel.</p>
    </div></div>
    <p class="muted">An attention weight or a large write shows where information <i>could</i> flow, not that it matters for the output. Test importance with patching.</p>`, `
    <div class="grid2"><div>
      <h4 style="color:${typeColor("full_attention")}">Full attention 层</h4>
      <p>每个 ${term("head")} 产生一个矩阵 p[query, key]：第 i 行表示 token i 如何把 attention 分配到 token 0…i 上（每行和为 1）。常见的 pattern：</p>
      <ul><li><b>${term("attention sink")}</b>：第一列很亮；这个 head 大多处于空闲。</li>
        <li><b>previous-token</b>：主对角线正下方的一条亮对角线；它构建 bigram 上下文。</li>
        <li><b>content / retrieval</b>：远离对角线的孤立亮格；它取回某个特定的更早 token。</li>
        <li><b>induction</b>：在重复文本上，token i 关注它自己早先副本<i>之后</i>的那个 token。</li></ul>
    </div><div>
      <h4 style="color:${typeColor("linear_attention")}">Gated delta 层</h4>
      <p>这里<b>没有 attention 矩阵</b>：过去被压缩进每个 head 一个 128×128 的记忆。你能看到的是这个记忆如何被逐 token 地<i>管理</i>：</p>
      <ul><li><b>retention exp(g)</b>：保留的记忆比例（它的 ${term("half-life")} 表示这个 head 能看多远）；</li>
        <li><b>write strength β</b>：这个 token 对记忆的修正程度；</li>
        <li><b>‖S‖</b>：记忆中存了多少内容；<b>write size</b>：这个 token 的修正有多大。</li></ul>
      <p>这些数值来自与模型 kernel 同时运行的显式 FP32 循环计算。</p>
    </div></div>
    <p class="muted">attention 权重或大的写入只说明信息<i>可能</i>在哪里流动，并不说明它对输出重要。请用 patching 检验重要性。</p>`)));
  const form = el("form", "panel");
  form.id = "mixer-form";
  form.innerHTML = `
    <label>Prompt<textarea name="prompt" rows="4"></textarea></label>
    <div class="row"><label>${tr("Layer", "层")} <select name="layer"></select></label>
      <label class="check"><input name="chat_template" type="checkbox"> chat template</label>
      <button type="submit">${tr("Run", "运行")}</button></div>
    <div class="presets"></div>`;
  root.appendChild(form);
  const out = el("div", "out");
  root.appendChild(out);
  form.elements.layer.innerHTML = layerOptions();
  form.elements.prompt.value = RETRIEVAL;
  form.elements.layer.value = "15";
  persist(form);
  presets($(".presets", form), form, [
    { label: tr("Retrieval, attention layer 15", "Retrieval，attention 第 15 层"), values: { prompt: RETRIEVAL, layer: "15" } },
    { label: tr("Repeated text (look for induction), layer 7", "重复文本（寻找 induction），第 7 层"), values: { prompt: REPEAT, layer: "7" } },
    { label: tr("Retrieval, gated-delta layer 14", "Retrieval，gated delta 第 14 层"), values: { prompt: RETRIEVAL, layer: "14" } },
    { label: tr("Repeated text, gated-delta layer 2", "重复文本，gated delta 第 2 层"), values: { prompt: REPEAT, layer: "2" } },
  ]);
  form.onsubmit = event => {
    event.preventDefault();
    run(form, out, "/api/mixer", { prompt: form.elements.prompt.value, layer: Number(form.elements.layer.value),
      chat_template: form.elements.chat_template.checked }, d => (d.kind === "full_attention" ? renderAttention : renderDelta)(d, out));
  };
}

const labelsOf = tokens => tokens.map((t, i) => `${i} ${plainTok(t)}`);

// Per-head pattern statistics over query rows 1..T-1.
function headStats(p) {
  const T = p.length;
  let sink = 0, prev = 0, self = 0, entropy = 0, n = 0;
  for (let i = 1; i < T; i++) {
    sink += p[i][0]; prev += p[i][i - 1]; self += p[i][i]; n++;
    entropy += -p[i].slice(0, i + 1).reduce((s, x) => s + (x > 0 ? x * Math.log2(x) : 0), 0);
  }
  return { sink: sink / n, prev: prev / n, self: self / n, entropy: entropy / n };
}
function induction(p, tokens) {
  // Mean attention from token i to the position right after an earlier occurrence of token i.
  let total = 0, n = 0;
  for (let i = 1; i < tokens.length; i++) {
    for (let j = 0; j < i - 1; j++) if (tokens[j] === tokens[i] && j + 1 < i) { total += p[i][j + 1]; n++; }
  }
  return n ? total / n : null;
}
function classify(s, ind) {
  if (ind != null && ind > 0.3) return "induction-like";
  if (s.sink > 0.6) return "mostly sink";
  if (s.prev > 0.4) return "previous-token";
  if (s.self > 0.4) return "self";
  return s.entropy > 3 ? "broad" : "content";
}

function renderAttention(d, out) {
  const labels = labelsOf(d.tokens), group = d.heads / d.kv_heads;
  const stats = d.probabilities.map(p => ({ ...headStats(p), induction: induction(p, d.tokens) }));
  out.appendChild(el("div", "result-head", `<span class="dot" style="background:${typeColor(d.kind)}"></span><b>${tr(`Layer ${d.layer}`, `第 ${d.layer} 层`)}</b> · full attention · ` +
    tr(`${d.heads} query heads, ${d.kv_heads} KV heads (heads 0–${group - 1} share KV head 0, ${group}–${2 * group - 1} share KV head 1)`,
       `${d.heads} 个 query head，${d.kv_heads} 个 KV head（head 0–${group - 1} 共享 KV head 0，${group}–${2 * group - 1} 共享 KV head 1）`) + ` · ${d.elapsed_seconds}s`));
  const last = d.tokens.length - 1;
  const bestLast = d.probabilities.map((p, h) => {
    const row = p[last].slice(1); const j = row.indexOf(Math.max(...row)) + 1; return { h, j, v: p[last][j] };
  }).sort((a, b) => b.v - a.v)[0];
  const sinks = stats.map((s, h) => `h${h} ${pct(s.sink)}`).join(", "), target = `${bestLast.j} ${tok(d.tokens[bestLast.j])}`;
  const indHead = stats.reduce((b, s, h) => (s.induction ?? -1) > (stats[b].induction ?? -1) ? h : b, 0), indMax = pct(Math.max(...stats.map(s => s.induction ?? 0)));
  out.appendChild(observations([
    tr(`Mean mass on position 0 (sink) per head: ${sinks}.`, `每个 head 在位置 0（sink）上的平均概率：${sinks}。`),
    tr(`From the last token, the strongest non-sink attention is head ${bestLast.h} → position ${target} (${pct(bestLast.v)}).`,
       `从最后一个 token 出发，最强的非 sink attention 是 head ${bestLast.h} → 位置 ${target}（${pct(bestLast.v)}）。`),
    stats.some(s => s.induction != null) ? tr(`Repeated tokens found; induction score (attention to the token after the earlier copy) is highest for head ${indHead} (${indMax}).`,
      `发现重复 token；induction 分数（对早先副本之后那个 token 的 attention）最高的是 head ${indHead}（${indMax}）。`) : null,
  ], tr("Pattern labels are simple heuristics on these statistics, meant as reading aids.", "Pattern 标签只是基于这些统计量的简单启发式规则，仅作阅读辅助。")));

  const grid = section(tr("All heads at a glance", "所有 head 一览"), tr("Each thumbnail is one head's query × key map. Click one to enlarge it below.", "每个缩略图是一个 head 的 query × key 图。点击可在下方放大。"));
  const thumbs = el("div", "thumbs-grid");
  grid.appendChild(thumbs);
  out.appendChild(grid);
  const detail = el("div", "");
  out.appendChild(detail);
  const size = Math.max(2, Math.min(5, Math.floor(150 / d.tokens.length)));
  d.probabilities.forEach((p, h) => {
    const card = el("div", "thumb");
    card.innerHTML = `<div class="thumb-title">head ${h} <span class="muted">· kv ${Math.floor(h / group)}</span></div>`;
    heatmap(card, { rows: labels, cols: labels, values: p, domain: [0, 1], compact: true, cell: size, skip: (r, c) => c > r,
      tip: (r, c, v) => `head ${h}\nquery ${labels[r]} → key ${labels[c]}\np = ${v.toFixed(3)}`, onClick: () => showHead(h) });
    const s = stats[h];
    card.insertAdjacentHTML("beforeend", `<div class="thumb-tag">${classify(s, s.induction)}</div><div class="thumb-stats">sink ${pct(s.sink)} · prev ${pct(s.prev)}</div>`);
    card.onclick = () => showHead(h);
    thumbs.appendChild(card);
  });

  function showHead(h, row = last) {
    [...thumbs.children].forEach((c, i) => c.classList.toggle("selected", i === h));
    detail.innerHTML = "";
    const s = section(`Head ${h}`, tr(`Rows are queries (the token doing the looking), columns are keys. Click a row to list where that token looks.`,
      `行是 query（发起查看的 token），列是 key。点击某一行可列出该 token 看向哪里。`));
    const holder = el("div", "grid-detail");
    const left = el("div", ""), right = el("div", "");
    holder.append(left, right);
    s.appendChild(holder);
    detail.appendChild(s);
    const draw = r => {
      left.innerHTML = "";
      heatmap(left, { rows: labels, cols: labels, values: d.probabilities[h], domain: [0, 1], skip: (rr, c) => c > rr, rowLabelWidth: 92,
        selected: { r }, format: v => v.toFixed(2), legend: tr("attention probability", "attention 概率"), maxCell: 22,
        tip: (rr, c, v) => `query ${labels[rr]}\nkey ${labels[c]}\np = ${v.toFixed(4)}`, onClick: rr => draw(rr) });
      const ranked = d.probabilities[h][r].map((v, j) => ({ j, v })).slice(0, r + 1).sort((a, b) => b.v - a.v).slice(0, 8);
      right.innerHTML = `<div class="chart-title">${tr(`Token ${r} ${tok(d.tokens[r])} attends to`, `Token ${r} ${tok(d.tokens[r])} 关注的位置`)}</div>` +
        barList(ranked.map(x => ({ label: `<span class="mono">${x.j} ${tok(d.tokens[x.j])}</span>`, value: x.v })), { format: pct, max: 1 }) +
        `<div class="caption">Pattern: <b>${classify(stats[h], stats[h].induction)}</b>. ${tr("Mean over rows", "各行平均")}: sink ${pct(stats[h].sink)}, previous token ${pct(stats[h].prev)}, self ${pct(stats[h].self)}, entropy ${stats[h].entropy.toFixed(2)} bits${stats[h].induction != null ? `, induction ${pct(stats[h].induction)}` : ""}.</div>`;
    };
    draw(row);
  }
  const auto = bestLast.h;
  showHead(auto);
}

function renderDelta(d, out) {
  const labels = labelsOf(d.tokens), T = d.tokens.length, H = d.heads;
  const byHead = key => Array.from({ length: H }, (_, h) => d[key].map(row => row[h]));
  const decay = byHead("decay"), beta = byHead("beta"), state = byHead("state_norm"), write = byHead("write_norm");
  const median = xs => { const s = xs.slice().sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
  const halfLife = r => r >= 1 ? Infinity : r <= 0 ? 0 : Math.log(0.5) / Math.log(r);
  const summary = Array.from({ length: H }, (_, h) => ({
    h, retention: median(decay[h]), half: halfLife(median(decay[h])), beta: beta[h].reduce((a, b) => a + b, 0) / T,
    finalNorm: state[h][T - 1],
  }));
  out.appendChild(el("div", "result-head", `<span class="dot" style="background:${typeColor(d.kind)}"></span><b>${tr(`Layer ${d.layer}`, `第 ${d.layer} 层`)}</b> · gated delta · ${tr(`${H} heads, 128×128 memory each`, `${H} 个 head，每个 128×128 记忆`)} · ${d.elapsed_seconds}s`));
  const sortedHalf = summary.slice().sort((a, b) => b.half - a.half);
  const meanRet = Array.from({ length: T }, (_, t) => decay.reduce((s, row) => s + row[t], 0) / H);
  const flush = meanRet.map((v, t) => ({ t, v })).slice(1).sort((a, b) => a.v - b.v)[0];
  const fmtHalf = x => Number.isFinite(x) ? (x >= 1000 ? "≥1000" : x.toFixed(x < 10 ? 1 : 0)) : "∞";
  const longest = sortedHalf[0], shortest = sortedHalf[H - 1];
  out.appendChild(observations([
    tr(`Longest memory: head ${longest.h} (median retention ${longest.retention.toFixed(4)}, half-life ≈ ${fmtHalf(longest.half)} tokens). Shortest: head ${shortest.h} (half-life ≈ ${fmtHalf(shortest.half)}).`,
       `记忆最长：head ${longest.h}（retention 中位数 ${longest.retention.toFixed(4)}，half-life ≈ ${fmtHalf(longest.half)} 个 token）。最短：head ${shortest.h}（half-life ≈ ${fmtHalf(shortest.half)}）。`),
    flush ? tr(`Averaged over heads, memory is flushed most at token ${flush.t} ${tok(d.tokens[flush.t])} (mean retention ${flush.v.toFixed(3)}).`,
               `对所有 head 取平均，记忆在 token ${flush.t} ${tok(d.tokens[flush.t])} 处被冲刷得最厉害（平均 retention ${flush.v.toFixed(3)}）。`) : null,
    tr(`Heads differ in time scale: a mix of short-memory heads (local syntax) and long-memory heads (facts across the prompt) is typical for gated linear attention.`,
       `各 head 的时间尺度不同：短记忆 head（局部句法）与长记忆 head（跨越整个 prompt 的事实）混合存在，这在 gated linear attention 中很典型。`),
  ], tr("Half-life uses each head's median retention over this prompt; retention changes token by token.",
        "Half-life 使用每个 head 在该 prompt 上的 retention 中位数；retention 会逐 token 变化。")));

  const table = section(tr("Per-head memory profile", "各 head 的记忆概况"), tr(`Sorted by half-life. ${term("half-life")} = ln 0.5 / ln(median retention).`,
    `按 half-life 排序。${term("half-life")} = ln 0.5 / ln（retention 中位数）。`));
  table.appendChild(el("div", "scroll", `<table><tr><th>Head</th><th class="num">${tr("Median retention", "Retention 中位数")}</th><th class="num">Half-life (tokens)</th><th class="num">${tr("Mean β", "平均 β")}</th><th class="num">${tr("Final ‖S‖", "最终 ‖S‖")}</th></tr>${
    sortedHalf.map(s => `<tr><td>head ${s.h}</td><td class="num">${s.retention.toFixed(4)}</td><td class="num">${fmtHalf(s.half)}</td><td class="num">${s.beta.toFixed(3)}</td><td class="num">${s.finalNorm.toFixed(2)}</td></tr>`).join("")}</table>`));
  out.appendChild(table);

  const metrics = [
    ["decay", "Retention exp(g)", decay, tr("fraction kept (colour range fit to data)", "保留比例（颜色范围按数据拟合）"), true],
    ["beta", "Write strength β", beta, "β", false],
    ["state", tr("Memory norm ‖S‖", "记忆范数 ‖S‖"), state, tr("Frobenius norm after the token", "该 token 之后的 Frobenius 范数"), true],
    ["write", "Write size ‖β(v − kᵀS)‖", write, tr("size of the correction", "修正的大小"), true],
  ];
  const maps = section(tr("Over tokens, per head", "逐 token、逐 head"), tr("Rows are heads, columns are prompt tokens. Click a head row for its curves.", "行是 head，列是 prompt 中的 token。点击某个 head 所在行可查看其曲线。"));
  const holder = el("div", "");
  const curves = el("div", "");
  let current = "decay", head = sortedHalf[0].h;
  const draw = () => {
    holder.innerHTML = "";
    const [, title, values, legend, fit] = metrics.find(m => m[0] === current);
    const flat = values.flat();
    heatmap(holder, { rows: values.map((_, h) => `head ${h}`), cols: labels, values, rowLabelWidth: 64, selected: { r: head },
      domain: fit ? [Math.min(...flat), Math.max(...flat)] : [0, 1], format: v => v.toFixed(3), legend,
      tip: (r, c, v) => `head ${r} · token ${labels[c]}\n${title} = ${v.toFixed(4)}`, onClick: r => { head = r; draw(); drawCurves(); } });
  };
  const drawCurves = () => {
    curves.innerHTML = "";
    const tl = d.tokens.map((_, i) => String(i));
    lineChart(curves, { labels: tl, xLabel: tr("token position", "token 位置"), yLabel: tr(`Head ${head}: gates per token`, `Head ${head}：每个 token 的门控值`), domain: [0, 1],
      series: [{ name: "retention exp(g)", values: decay[head] }, { name: "write strength β", values: beta[head] }],
      tip: i => `token ${labels[i]}\nretention ${decay[head][i].toFixed(4)}\nβ ${beta[head][i].toFixed(4)}` });
    lineChart(curves, { labels: tl, xLabel: tr("token position", "token 位置"), yLabel: tr(`Head ${head}: memory norm ‖S‖`, `Head ${head}：记忆范数 ‖S‖`), series: [{ name: "‖S‖", values: state[head] }],
      tip: i => `token ${labels[i]}\n‖S‖ ${state[head][i].toFixed(3)}\nwrite ${write[head][i].toFixed(3)}` });
  };
  maps.appendChild(segmented(metrics.map(m => [m[0], m[1]]), current, v => { current = v; draw(); }));
  maps.appendChild(holder);
  maps.appendChild(curves);
  out.appendChild(maps);
  draw(); drawCurves();
}

