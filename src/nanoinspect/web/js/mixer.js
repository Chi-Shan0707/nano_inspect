// Mixer maps: all heads of one layer. Full attention -> softmax maps; gated delta -> gates and memory over tokens.
import { $, concept, el, esc, layerOptions, observations, persist, pct, plainTok, presets, run, section, segmented, term, tok, typeColor } from "./core.js";
import { barList, heatmap, lineChart } from "./charts.js";

const RETRIEVAL = "Facts:\nThe red box contains the apple.\nThe blue box contains the coin.\nQuestion: What does the blue box contain?\nAnswer: The blue box contains the";
const REPEAT = "The violet giraffe ate seven lemons. Then the violet giraffe ate seven";

export function initMixer() {
  const root = $("#mixer");
  root.innerHTML = "";
  root.appendChild(el("p", "lede", "<b>Question:</b> across all heads of a layer, which earlier tokens does each position draw on, and how does recurrent memory evolve?"));
  root.appendChild(concept("mixer", "Concept: attention patterns vs recurrent memory", `
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
    <p class="muted">An attention weight or a large write shows where information <i>could</i> flow, not that it matters for the output. Test importance with patching.</p>`));
  const form = el("form", "panel");
  form.id = "mixer-form";
  form.innerHTML = `
    <label>Prompt<textarea name="prompt" rows="4"></textarea></label>
    <div class="row"><label>Layer <select name="layer"></select></label>
      <label class="check"><input name="chat_template" type="checkbox"> chat template</label>
      <button type="submit">Run</button></div>
    <div class="presets"></div>`;
  root.appendChild(form);
  const out = el("div", "out");
  root.appendChild(out);
  form.elements.layer.innerHTML = layerOptions();
  form.elements.prompt.value = RETRIEVAL;
  form.elements.layer.value = "15";
  persist(form);
  presets($(".presets", form), form, [
    { label: "Retrieval, attention layer 15", values: { prompt: RETRIEVAL, layer: "15" } },
    { label: "Repeated text (look for induction), layer 7", values: { prompt: REPEAT, layer: "7" } },
    { label: "Retrieval, gated-delta layer 14", values: { prompt: RETRIEVAL, layer: "14" } },
    { label: "Repeated text, gated-delta layer 2", values: { prompt: REPEAT, layer: "2" } },
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
  out.appendChild(el("div", "result-head", `<span class="dot" style="background:${typeColor(d.kind)}"></span><b>Layer ${d.layer}</b> · full attention · ${d.heads} query heads, ${d.kv_heads} KV heads (heads ${0}–${group - 1} share KV head 0, ${group}–${2 * group - 1} share KV head 1) · ${d.elapsed_seconds}s`));
  const last = d.tokens.length - 1;
  const bestLast = d.probabilities.map((p, h) => {
    const row = p[last].slice(1); const j = row.indexOf(Math.max(...row)) + 1; return { h, j, v: p[last][j] };
  }).sort((a, b) => b.v - a.v)[0];
  out.appendChild(observations([
    `Mean mass on position 0 (sink) per head: ${stats.map((s, h) => `h${h} ${pct(s.sink)}`).join(", ")}.`,
    `From the last token, the strongest non-sink attention is head ${bestLast.h} → position ${bestLast.j} ${tok(d.tokens[bestLast.j])} (${pct(bestLast.v)}).`,
    stats.some(s => s.induction != null) ? `Repeated tokens found; induction score (attention to the token after the earlier copy) is highest for head ${stats.reduce((b, s, h) => (s.induction ?? -1) > (stats[b].induction ?? -1) ? h : b, 0)} (${pct(Math.max(...stats.map(s => s.induction ?? 0)))}).` : null,
  ], "Pattern labels are simple heuristics on these statistics, meant as reading aids."));

  const grid = section("All heads at a glance", "Each thumbnail is one head's query × key map. Click one to enlarge it below.");
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
    const s = section(`Head ${h}`, `Rows are queries (the token doing the looking), columns are keys. Click a row to list where that token looks.`);
    const holder = el("div", "grid-detail");
    const left = el("div", ""), right = el("div", "");
    holder.append(left, right);
    s.appendChild(holder);
    detail.appendChild(s);
    const draw = r => {
      left.innerHTML = "";
      heatmap(left, { rows: labels, cols: labels, values: d.probabilities[h], domain: [0, 1], skip: (rr, c) => c > rr, rowLabelWidth: 92,
        selected: { r }, format: v => v.toFixed(2), legend: "attention probability", maxCell: 22,
        tip: (rr, c, v) => `query ${labels[rr]}\nkey ${labels[c]}\np = ${v.toFixed(4)}`, onClick: rr => draw(rr) });
      const ranked = d.probabilities[h][r].map((v, j) => ({ j, v })).slice(0, r + 1).sort((a, b) => b.v - a.v).slice(0, 8);
      right.innerHTML = `<div class="chart-title">Token ${r} ${tok(d.tokens[r])} attends to</div>` +
        barList(ranked.map(x => ({ label: `<span class="mono">${x.j} ${tok(d.tokens[x.j])}</span>`, value: x.v })), { format: pct, max: 1 }) +
        `<div class="caption">Pattern: <b>${classify(stats[h], stats[h].induction)}</b>. Mean over rows: sink ${pct(stats[h].sink)}, previous token ${pct(stats[h].prev)}, self ${pct(stats[h].self)}, entropy ${stats[h].entropy.toFixed(2)} bits${stats[h].induction != null ? `, induction ${pct(stats[h].induction)}` : ""}.</div>`;
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
  out.appendChild(el("div", "result-head", `<span class="dot" style="background:${typeColor(d.kind)}"></span><b>Layer ${d.layer}</b> · gated delta · ${H} heads, 128×128 memory each · ${d.elapsed_seconds}s`));
  const sortedHalf = summary.slice().sort((a, b) => b.half - a.half);
  const meanRet = Array.from({ length: T }, (_, t) => decay.reduce((s, row) => s + row[t], 0) / H);
  const flush = meanRet.map((v, t) => ({ t, v })).slice(1).sort((a, b) => a.v - b.v)[0];
  const fmtHalf = x => Number.isFinite(x) ? (x >= 1000 ? "≥1000" : x.toFixed(x < 10 ? 1 : 0)) : "∞";
  out.appendChild(observations([
    `Longest memory: head ${sortedHalf[0].h} (median retention ${sortedHalf[0].retention.toFixed(4)}, half-life ≈ ${fmtHalf(sortedHalf[0].half)} tokens). Shortest: head ${sortedHalf[H - 1].h} (half-life ≈ ${fmtHalf(sortedHalf[H - 1].half)}).`,
    flush ? `Averaged over heads, memory is flushed most at token ${flush.t} ${tok(d.tokens[flush.t])} (mean retention ${flush.v.toFixed(3)}).` : null,
    `Heads differ in time scale: a mix of short-memory heads (local syntax) and long-memory heads (facts across the prompt) is typical for gated linear attention.`,
  ], "Half-life uses each head's median retention over this prompt; retention changes token by token."));

  const table = section("Per-head memory profile", `Sorted by half-life. ${term("half-life")} = ln 0.5 / ln(median retention).`);
  table.appendChild(el("div", "scroll", `<table><tr><th>Head</th><th class="num">Median retention</th><th class="num">Half-life (tokens)</th><th class="num">Mean β</th><th class="num">Final ‖S‖</th></tr>${
    sortedHalf.map(s => `<tr><td>head ${s.h}</td><td class="num">${s.retention.toFixed(4)}</td><td class="num">${fmtHalf(s.half)}</td><td class="num">${s.beta.toFixed(3)}</td><td class="num">${s.finalNorm.toFixed(2)}</td></tr>`).join("")}</table>`));
  out.appendChild(table);

  const metrics = [
    ["decay", "Retention exp(g)", decay, "fraction kept (colour range fit to data)", true],
    ["beta", "Write strength β", beta, "β", false],
    ["state", "Memory norm ‖S‖", state, "Frobenius norm after the token", true],
    ["write", "Write size ‖β(v − kᵀS)‖", write, "size of the correction", true],
  ];
  const maps = section("Over tokens, per head", "Rows are heads, columns are prompt tokens. Click a head row for its curves.");
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
    lineChart(curves, { labels: tl, xLabel: "token position", yLabel: `Head ${head}: gates per token`, domain: [0, 1],
      series: [{ name: "retention exp(g)", values: decay[head] }, { name: "write strength β", values: beta[head] }],
      tip: i => `token ${labels[i]}\nretention ${decay[head][i].toFixed(4)}\nβ ${beta[head][i].toFixed(4)}` });
    lineChart(curves, { labels: tl, xLabel: "token position", yLabel: `Head ${head}: memory norm ‖S‖`, series: [{ name: "‖S‖", values: state[head] }],
      tip: i => `token ${labels[i]}\n‖S‖ ${state[head][i].toFixed(3)}\nwrite ${write[head][i].toFixed(3)}` });
  };
  maps.appendChild(segmented(metrics.map(m => [m[0], m[1]]), current, v => { current = v; draw(); }));
  maps.appendChild(holder);
  maps.appendChild(curves);
  out.appendChild(maps);
  draw(); drawCurves();
}

