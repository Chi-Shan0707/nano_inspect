// Logit lens: read the residual stream at every (layer, position); component norms.
import { $, concept, el, esc, fmtInt, observations, persist, pct, plainTok, presets, run, section, segmented, term, tok,
         tokenStrip, tr, typeColor } from "./core.js";
import { barList, heatmap, lineChart } from "./charts.js";

const RETRIEVAL = "Facts:\nThe red box contains the apple.\nThe blue box contains the coin.\nQuestion: What does the blue box contain?\nAnswer: The blue box contains the";

export function initLens() {
  const root = $("#lens");
  root.innerHTML = "";
  root.appendChild(el("p", "lede", tr("<b>Question:</b> at which depth does the model's prediction take shape, and what does each intermediate layer “think” comes next?",
    "<b>问题：</b>模型的预测在哪个深度成形？每个中间层“认为”接下来是什么？")));
  root.appendChild(concept("lens", tr("Concept: the logit lens", "概念：logit lens"), tr(`
    <p>Because every block <i>adds</i> to the ${term("residual stream")}, we can stop anywhere and apply the model's own final norm and ${term("unembedding")}:</p>
    <pre class="formula">logits_ℓ = W_U · RMSNorm(x_ℓ)        x_ℓ = residual stream after layer ℓ</pre>
    <p>Row <b>E</b> reads the raw token embedding; row 23 is exactly the model's output. In the grid, each cell shows the lens's <b>top-1 token</b> at that layer and position, coloured by its probability.
    Cells whose label starts with ▪ predict the <b>actual next token</b> of your prompt.</p>
    <p>Read it as a probe: early layers often decode to junk because they were never trained to be read this way (their basis differs from the output's). What matters is <i>when</i> the right answer rises, and how sharply.</p>`, `
    <p>由于每个 block 都是往 ${term("residual stream")} 上<i>相加</i>，我们可以在任意位置停下，套用模型自己的 final norm 和 ${term("unembedding")}：</p>
    <pre class="formula">logits_ℓ = W_U · RMSNorm(x_ℓ)        x_ℓ = 第 ℓ 层之后的 residual stream</pre>
    <p><b>E</b> 行读取原始的 token embedding；第 23 行正好就是模型的输出。网格中每个格子显示该层、该位置上 lens 的 <b>top-1 token</b>，颜色表示其概率。
    标签以 ▪ 开头的格子预测的是你的 prompt 中<b>实际的下一个 token</b>。</p>
    <p>把它当作探针来读：浅层常常解码出无意义的 token，因为它们从未被训练成可以这样读取（它们的基与输出层不同）。重要的是正确答案<i>何时</i>上升，以及上升得有多陡。</p>`)));
  const form = el("form", "panel");
  form.id = "lens-form";
  form.innerHTML = `
    <label>Prompt<textarea name="prompt" rows="4"></textarea></label>
    <div class="row">
      <label>Target token <input name="target" placeholder="${tr("default: model's prediction", "默认：模型的预测")}"></label>
      <label>Top-k <input name="top_k" type="number" value="5" min="1" max="10"></label>
      <label class="check"><input name="chat_template" type="checkbox"> chat template</label>
      <button type="submit">${tr("Run", "运行")}</button></div>
    <div class="presets"></div>`;
  root.appendChild(form);
  const out = el("div", "out");
  root.appendChild(out);
  form.elements.prompt.value = RETRIEVAL;
  form.elements.target.value = " coin";
  persist(form);
  presets($(".presets", form), form, [
    { label: tr("Fact retrieval (target ·coin)", "事实检索（target ·coin）"), values: { prompt: RETRIEVAL, target: " coin" } },
    { label: tr("Capital city", "首都城市"), values: { prompt: "The Eiffel Tower is located in the city of", target: " Paris" } },
    { label: tr("Arithmetic", "算术"), values: { prompt: "Question: What is 7 plus 5?\nAnswer:", target: " 12" } },
    { label: tr("Multilingual", "多语言"), values: { prompt: "English: cat\nFrench: chat\nEnglish: dog\nFrench:", target: " chien" } },
  ]);
  form.onsubmit = event => {
    event.preventDefault();
    run(form, out, "/api/lens", { prompt: form.elements.prompt.value, target: form.elements.target.value,
      top_k: Number(form.elements.top_k.value), chat_template: form.elements.chat_template.checked }, d => render(d, out));
  };
}

function render(d, out) {
  const T = d.tokens.length, L = d.lens.length, last = T - 1;
  const rowLabel = r => (r === 0 ? "E" : String(r - 1));
  const target = d.target.token;
  // Actual next token of the prompt at each position (for the "already predicts the text" marks).
  const nextTok = i => (i + 1 < T ? d.tokens[i + 1] : null);
  const matches = d.lens.map(row => row.cells.map((cell, i) => nextTok(i) != null && cell.top[0][0] === nextTok(i)));

  // Observations at the last position
  const cellsLast = d.lens.map(r => r.cells[last]);
  const firstTop5 = cellsLast.findIndex(c => c.target_rank <= 5);
  const firstTop1 = cellsLast.findIndex(c => c.target_rank === 1);
  const stays = firstTop1 >= 0 && cellsLast.slice(firstTop1).every(c => c.target_rank === 1);
  const accByLayer = matches.map(row => row.slice(0, -1).filter(Boolean).length / Math.max(1, T - 1));
  const accFinal = accByLayer[L - 1];
  const halfway = accByLayer.findIndex(a => a >= accFinal / 2);
  const chosen = d.target.chosen === "given" ? tr("given", "指定") : tr(d.target.chosen, "模型 top-1");
  out.appendChild(el("div", "result-head", `Target ${tok(target)} (${chosen}) · ${tr(`model's top-${d.model_top.length} at the last position`, `模型在最后一个位置的 top-${d.model_top.length}`)}: ${d.model_top.map(x => `${tok(x.token)} ${pct(x.prob)}`).join(" · ")} · ${d.elapsed_seconds}s`));
  const T5 = rowLabel(firstTop5), T1 = rowLabel(firstTop1), finalCell = cellsLast[L - 1], half = rowLabel(Math.max(0, halfway));
  out.appendChild(observations([
    firstTop5 < 0 ? tr(`${tok(target)} never enters the lens's top-5 at the last position.`, `在最后一个位置，${tok(target)} 从未进入 lens 的 top-5。`) :
      tr(`At the last position, ${tok(target)} first enters the top-5 at layer <b>${T5}</b>` +
           (firstTop1 >= 0 ? ` and becomes top-1 at layer <b>${T1}</b>${stays ? " (and stays there)" : " (but loses it again later)"}.` : `, but never becomes top-1.`),
         `在最后一个位置，${tok(target)} 在第 <b>${T5}</b> 层首次进入 top-5` +
           (firstTop1 >= 0 ? `，并在第 <b>${T1}</b> 层成为 top-1${stays ? "（此后一直保持）" : "（但之后又失去了）"}。` : `，但从未成为 top-1。`)),
    tr(`Final probability of ${tok(target)}: <b>${pct(finalCell.target_prob)}</b> (rank ${fmtInt(finalCell.target_rank)}).`,
       `${tok(target)} 的最终概率：<b>${pct(finalCell.target_prob)}</b>（排名 ${fmtInt(finalCell.target_rank)}）。`),
    tr(`Over the whole prompt, the final layer's top-1 equals the actual next token at ${pct(accFinal)} of positions. Half of that accuracy is already reached at layer ${half}.`,
       `在整个 prompt 上，最后一层的 top-1 在 ${pct(accFinal)} 的位置上等于实际的下一个 token。在第 ${half} 层就已达到这一准确率的一半。`),
  ], tr("Early-layer lens readouts are often meaningless tokens; read trends, not individual early cells.",
        "浅层的 lens 读出常常是无意义的 token；请看趋势，而不是单个浅层格子。")));

  // Grid
  const grid = section(tr("Lens grid: top-1 prediction at every layer and position", "Lens 网格：每层、每个位置的 top-1 预测"),
    tr("Colour = probability of the shown token (or of the target, see toggle). ▪ in a cell's label = matches the actual next prompt token. Click any cell for its top-k.",
       "颜色 = 所显示 token 的概率（或 target 的概率，见切换按钮）。格子标签中的 ▪ = 与 prompt 中实际的下一个 token 一致。点击任意格子查看其 top-k。"));
  out.appendChild(grid);
  const gridHolder = el("div", "");
  const detail = el("div", "grid-detail");
  let mode = "top1", selected = { r: L - 1, c: last };
  grid.appendChild(segmented([["top1", tr("colour: top-1 probability", "颜色：top-1 概率")], ["target", tr(`colour: P(${plainTok(target)})`, `颜色：P(${plainTok(target)})`)]], mode, m => { mode = m; draw(); }));
  grid.appendChild(gridHolder);
  grid.appendChild(detail);
  const draw = () => {
    gridHolder.innerHTML = "";
    const values = d.lens.map(row => row.cells.map(c => (mode === "top1" ? c.top[0][1] : c.target_prob)));
    const wide = T <= 64;  // label cells with tokens; the grid scrolls sideways when needed
    heatmap(gridHolder, {
      rows: d.lens.map((_, r) => rowLabel(r)), cols: d.tokens.map((t, i) => `${i} ${plainTok(t)}`), values, domain: [0, 1],
      rowColor: r => typeColor(d.lens[r].type), cell: wide ? Math.max(52, Math.min(72, Math.floor(1220 / T))) : undefined,
      minCell: 8, cellHeight: wide ? 21 : 14, selected, rowLabelWidth: 40,
      text: wide ? (r, c) => (matches[r][c] ? "▪" : "") + plainTok(d.lens[r].cells[c].top[0][0]).trim().slice(0, 10) : null,
      legend: mode === "top1" ? tr("probability of the top-1 token", "top-1 token 的概率") : `P(${plainTok(target)})`, format: v => v.toFixed(1),
      tip: (r, c) => {
        const cell = d.lens[r].cells[c];
        return tr(`layer ${rowLabel(r)} · position ${c}`, `第 ${rowLabel(r)} 层 · 位置 ${c}`) + ` ${JSON.stringify(d.tokens[c])}\n` + cell.top.map(([t, p], k) => `${k + 1}. ${JSON.stringify(t)} ${pct(p)}`).join("\n") +
          `\nP(${JSON.stringify(target)}) = ${pct(cell.target_prob)} (${tr("rank", "排名")} ${fmtInt(cell.target_rank)})` + (nextTok(c) != null ? `\n${tr("actual next token", "实际的下一个 token")}: ${JSON.stringify(nextTok(c))}` : "");
      },
      onClick: (r, c) => { selected = { r, c }; draw(); },
    });
    showCell(selected.r, selected.c);
  };
  const showCell = (r, c) => {
    detail.innerHTML = "";
    const cell = d.lens[r].cells[c];
    const left = el("div", ""), right = el("div", "");
    left.innerHTML = `<div class="chart-title">${tr(`Layer ${rowLabel(r)}, position ${c}`, `第 ${rowLabel(r)} 层，位置 ${c}`)} ${tok(d.tokens[c])}: top-${cell.top.length}</div>` +
      barList(cell.top.map(([t, p]) => ({ label: `<span class="mono">${tok(t)}</span>${nextTok(c) === t ? " ✓" : ""}`, value: p })), { format: pct, max: 1 }) +
      `<div class="caption">${nextTok(c) != null ? tr(`Actual next token: ${tok(nextTok(c))}. `, `实际的下一个 token：${tok(nextTok(c))}。`) : tr("Last position: this is the model's answer. ", "最后一个位置：这就是模型的答案。")}P(${tok(target)}) = ${pct(cell.target_prob)}, ${tr("rank", "排名")} ${fmtInt(cell.target_rank)}.</div>`;
    detail.append(left, right);
    const probs = d.lens.map(row => row.cells[c].target_prob), ranks = d.lens.map(row => row.cells[c].target_rank);
    const which = el("div", "");
    right.appendChild(segmented([["prob", tr("probability", "概率")], ["rank", tr("rank (log scale)", "排名（对数坐标）")]], "prob", m => plot(m)));
    right.appendChild(which);
    const plot = m => {
      which.innerHTML = "";
      lineChart(which, { labels: d.lens.map((_, k) => rowLabel(k)), xLabel: tr("layer", "层"), refLine: r,
        yLabel: m === "prob" ? tr(`P(${plainTok(target)}) by layer at position ${c}`, `位置 ${c} 上 P(${plainTok(target)}) 随层的变化`)
                             : tr(`Rank of ${plainTok(target)} by layer at position ${c} (1 = top)`, `位置 ${c} 上 ${plainTok(target)} 的排名随层的变化（1 = 最高）`),
        series: [{ name: m, values: m === "prob" ? probs : ranks }], log: m === "rank",
        domain: m === "prob" ? [0, Math.max(0.05, Math.min(1, Math.max(...probs) * 1.1))] : [1, Math.max(10, ...ranks)],
        format: m === "prob" ? v => v.toFixed(2) : v => fmtInt(Math.round(v)),
        tip: k => `${tr(`layer ${rowLabel(k)}`, `第 ${rowLabel(k)} 层`)}\nP = ${pct(probs[k])}\n${tr("rank", "排名")} ${fmtInt(ranks[k])}\ntop-1: ${JSON.stringify(d.lens[k].cells[c].top[0][0])}` });
    };
    plot("prob");
  };
  draw();

  // Norms
  const norms = section(tr("How big is each update?", "每次更新有多大？"), tr(`L2 norm of the residual stream and of each block's mixer and MLP outputs, per layer and token. Size is not importance: use patching for causal claims.`,
    `residual stream 以及每个 block 的 mixer 输出和 MLP 输出的 L2 范数，逐层、逐 token。大小不等于重要性：因果结论请用 patching。`));
  const holder = el("div", "");
  let site = "residual_output", skipFirst = true;
  const sites = [["residual_output", "residual stream"], ["mixer_output", tr("mixer output", "mixer 输出")], ["mlp_output", tr("MLP output", "MLP 输出")]];
  const drawNorms = () => {
    holder.innerHTML = "";
    const values = d.norms[site];
    const flat = values.flatMap(row => row.slice(skipFirst ? 1 : 0)).sort((a, b) => a - b);
    heatmap(holder, { rows: values.map((_, i) => i), cols: d.tokens.map((t, i) => `${i} ${plainTok(t)}`), values, domain: [0, flat[Math.floor(flat.length * 0.99)] || 1],
      rowColor: r => typeColor(d.layer_types[r]), format: v => v.toFixed(1),
      legend: tr(`L2 norm (colour clipped at the 99th percentile${skipFirst ? ", position 0 excluded from the range" : ""})`, `L2 范数（颜色在第 99 百分位截断${skipFirst ? "，范围不含位置 0" : ""}）`),
      tip: (r, c, v) => `${tr(`layer ${r} · position ${c}`, `第 ${r} 层 · 位置 ${c}`)} ${JSON.stringify(d.tokens[c])}\n‖${site}‖ = ${v.toFixed(3)}` });
  };
  norms.appendChild(segmented(sites, site, v => { site = v; drawNorms(); }));
  const toggle = el("label", "check inline", `<input type="checkbox" checked> ${tr("exclude position 0 from the colour range", "颜色范围不含位置 0")}`);
  toggle.querySelector("input").onchange = e => { skipFirst = e.target.checked; drawNorms(); };
  norms.appendChild(toggle);
  norms.appendChild(holder);
  const res = d.norms.residual_output;
  const medianRest = row => { const r = row.slice(1).sort((a, b) => a - b); return r[Math.floor(r.length / 2)] ?? row[0]; };
  const ratios = res.map(row => row[0] / medianRest(row));
  const peak = ratios.indexOf(Math.max(...ratios));
  const mid = Math.floor(res.length / 2);
  const ratio = ratios[peak].toFixed(0), n0 = medianRest(res[0]).toFixed(1), nMid = medianRest(res[mid]).toFixed(1), nEnd = medianRest(res[res.length - 1]).toFixed(1), lastLayer = res.length - 1;
  norms.appendChild(observations([
    T > 1 && ratios[peak] > 3 ? tr(`Position 0 ${tok(d.tokens[0])} reaches a residual norm <b>${ratio}×</b> the median of the other positions (at layer ${peak}): a “massive activation”, typical of the token heads use as an ${term("attention sink")}.`,
      `位置 0 ${tok(d.tokens[0])} 的 residual 范数达到其他位置中位数的 <b>${ratio} 倍</b>（在第 ${peak} 层）：这是一种“massive activation”，常见于被 head 用作 ${term("attention sink")} 的 token。`) : null,
    T > 1 ? tr(`The residual norm grows with depth: median ${n0} after layer 0, ${nMid} after layer ${mid}, ${nEnd} after layer ${lastLayer}. Each block adds to the stream, and RMSNorm makes the blocks indifferent to its overall scale.`,
      `residual 范数随深度增长：第 0 层之后中位数为 ${n0}，第 ${mid} 层之后为 ${nMid}，第 ${lastLayer} 层之后为 ${nEnd}。每个 block 都往 stream 上相加，而 RMSNorm 让各 block 不在意它的整体尺度。`) : null,
  ]));
  out.appendChild(norms);
  drawNorms();
}
