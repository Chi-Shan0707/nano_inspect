// Logit lens: read the residual stream at every (layer, position); component norms.
import { $, concept, el, esc, fmtInt, observations, persist, pct, plainTok, presets, run, section, segmented, term, tok,
         tokenStrip, typeColor } from "./core.js";
import { barList, heatmap, lineChart } from "./charts.js";

const RETRIEVAL = "Facts:\nThe red box contains the apple.\nThe blue box contains the coin.\nQuestion: What does the blue box contain?\nAnswer: The blue box contains the";

export function initLens() {
  const root = $("#lens");
  root.innerHTML = "";
  root.appendChild(el("p", "lede", "<b>Question:</b> at which depth does the model's prediction take shape, and what does each intermediate layer “think” comes next?"));
  root.appendChild(concept("lens", "Concept: the logit lens", `
    <p>Because every block <i>adds</i> to the ${term("residual stream")}, we can stop anywhere and apply the model's own final norm and ${term("unembedding")}:</p>
    <pre class="formula">logits_ℓ = W_U · RMSNorm(x_ℓ)        x_ℓ = residual stream after layer ℓ</pre>
    <p>Row <b>E</b> reads the raw token embedding; row 23 is exactly the model's output. In the grid, each cell shows the lens's <b>top-1 token</b> at that layer and position, coloured by its probability.
    Cells whose label starts with ▪ predict the <b>actual next token</b> of your prompt.</p>
    <p>Read it as a probe: early layers often decode to junk because they were never trained to be read this way (their basis differs from the output's). What matters is <i>when</i> the right answer rises, and how sharply.</p>`));
  const form = el("form", "panel");
  form.id = "lens-form";
  form.innerHTML = `
    <label>Prompt<textarea name="prompt" rows="4"></textarea></label>
    <div class="row">
      <label>Target token <input name="target" placeholder="default: model's prediction"></label>
      <label>Top-k <input name="top_k" type="number" value="5" min="1" max="10"></label>
      <label class="check"><input name="chat_template" type="checkbox"> chat template</label>
      <button type="submit">Run</button></div>
    <div class="presets"></div>`;
  root.appendChild(form);
  const out = el("div", "out");
  root.appendChild(out);
  form.elements.prompt.value = RETRIEVAL;
  form.elements.target.value = " coin";
  persist(form);
  presets($(".presets", form), form, [
    { label: "Fact retrieval (target ·coin)", values: { prompt: RETRIEVAL, target: " coin" } },
    { label: "Capital city", values: { prompt: "The Eiffel Tower is located in the city of", target: " Paris" } },
    { label: "Arithmetic", values: { prompt: "Question: What is 7 plus 5?\nAnswer:", target: " 12" } },
    { label: "Multilingual", values: { prompt: "English: cat\nFrench: chat\nEnglish: dog\nFrench:", target: " chien" } },
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
  out.appendChild(el("div", "result-head", `Target ${tok(target)} (${d.target.chosen}) · model's top-${d.model_top.length} at the last position: ${d.model_top.map(x => `${tok(x.token)} ${pct(x.prob)}`).join(" · ")} · ${d.elapsed_seconds}s`));
  out.appendChild(observations([
    firstTop5 < 0 ? `${tok(target)} never enters the lens's top-5 at the last position.` :
      `At the last position, ${tok(target)} first enters the top-5 at layer <b>${rowLabel(firstTop5)}</b>` +
      (firstTop1 >= 0 ? ` and becomes top-1 at layer <b>${rowLabel(firstTop1)}</b>${stays ? " (and stays there)" : " (but loses it again later)"}.` : `, but never becomes top-1.`),
    `Final probability of ${tok(target)}: <b>${pct(cellsLast[L - 1].target_prob)}</b> (rank ${fmtInt(cellsLast[L - 1].target_rank)}).`,
    `Over the whole prompt, the final layer's top-1 equals the actual next token at ${pct(accFinal)} of positions. Half of that accuracy is already reached at layer ${rowLabel(Math.max(0, halfway))}.`,
  ], "Early-layer lens readouts are often meaningless tokens; read trends, not individual early cells."));

  // Grid
  const grid = section("Lens grid: top-1 prediction at every layer and position",
    "Colour = probability of the shown token (or of the target, see toggle). ▪ in a cell's label = matches the actual next prompt token. Click any cell for its top-k.");
  out.appendChild(grid);
  const gridHolder = el("div", "");
  const detail = el("div", "grid-detail");
  let mode = "top1", selected = { r: L - 1, c: last };
  grid.appendChild(segmented([["top1", "colour: top-1 probability"], ["target", `colour: P(${plainTok(target)})`]], mode, m => { mode = m; draw(); }));
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
      legend: mode === "top1" ? "probability of the top-1 token" : `P(${plainTok(target)})`, format: v => v.toFixed(1),
      tip: (r, c) => {
        const cell = d.lens[r].cells[c];
        return `layer ${rowLabel(r)} · position ${c} ${JSON.stringify(d.tokens[c])}\n` + cell.top.map(([t, p], k) => `${k + 1}. ${JSON.stringify(t)} ${pct(p)}`).join("\n") +
          `\nP(${JSON.stringify(target)}) = ${pct(cell.target_prob)} (rank ${fmtInt(cell.target_rank)})` + (nextTok(c) != null ? `\nactual next token: ${JSON.stringify(nextTok(c))}` : "");
      },
      onClick: (r, c) => { selected = { r, c }; draw(); },
    });
    showCell(selected.r, selected.c);
  };
  const showCell = (r, c) => {
    detail.innerHTML = "";
    const cell = d.lens[r].cells[c];
    const left = el("div", ""), right = el("div", "");
    left.innerHTML = `<div class="chart-title">Layer ${rowLabel(r)}, position ${c} ${tok(d.tokens[c])}: top-${cell.top.length}</div>` +
      barList(cell.top.map(([t, p]) => ({ label: `<span class="mono">${tok(t)}</span>${nextTok(c) === t ? " ✓" : ""}`, value: p })), { format: pct, max: 1 }) +
      `<div class="caption">${nextTok(c) != null ? `Actual next token: ${tok(nextTok(c))}. ` : "Last position: this is the model's answer. "}P(${tok(target)}) = ${pct(cell.target_prob)}, rank ${fmtInt(cell.target_rank)}.</div>`;
    detail.append(left, right);
    const probs = d.lens.map(row => row.cells[c].target_prob), ranks = d.lens.map(row => row.cells[c].target_rank);
    const which = el("div", "");
    right.appendChild(segmented([["prob", "probability"], ["rank", "rank (log scale)"]], "prob", m => plot(m)));
    right.appendChild(which);
    const plot = m => {
      which.innerHTML = "";
      lineChart(which, { labels: d.lens.map((_, k) => rowLabel(k)), xLabel: "layer", refLine: r,
        yLabel: m === "prob" ? `P(${plainTok(target)}) by layer at position ${c}` : `Rank of ${plainTok(target)} by layer at position ${c} (1 = top)`,
        series: [{ name: m, values: m === "prob" ? probs : ranks }], log: m === "rank",
        domain: m === "prob" ? [0, Math.max(0.05, Math.min(1, Math.max(...probs) * 1.1))] : [1, Math.max(10, ...ranks)],
        format: m === "prob" ? v => v.toFixed(2) : v => fmtInt(Math.round(v)),
        tip: k => `layer ${rowLabel(k)}\nP = ${pct(probs[k])}\nrank ${fmtInt(ranks[k])}\ntop-1: ${JSON.stringify(d.lens[k].cells[c].top[0][0])}` });
    };
    plot("prob");
  };
  draw();

  // Norms
  const norms = section("How big is each update?", `L2 norm of the residual stream and of each block's mixer and MLP outputs, per layer and token. Size is not importance: use patching for causal claims.`);
  const holder = el("div", "");
  let site = "residual_output", skipFirst = true;
  const sites = [["residual_output", "residual stream"], ["mixer_output", "mixer output"], ["mlp_output", "MLP output"]];
  const drawNorms = () => {
    holder.innerHTML = "";
    const values = d.norms[site];
    const flat = values.flatMap(row => row.slice(skipFirst ? 1 : 0)).sort((a, b) => a - b);
    heatmap(holder, { rows: values.map((_, i) => i), cols: d.tokens.map((t, i) => `${i} ${plainTok(t)}`), values, domain: [0, flat[Math.floor(flat.length * 0.99)] || 1],
      rowColor: r => typeColor(d.layer_types[r]), format: v => v.toFixed(1), legend: `L2 norm (colour clipped at the 99th percentile${skipFirst ? ", position 0 excluded from the range" : ""})`,
      tip: (r, c, v) => `layer ${r} · position ${c} ${JSON.stringify(d.tokens[c])}\n‖${site}‖ = ${v.toFixed(3)}` });
  };
  norms.appendChild(segmented(sites, site, v => { site = v; drawNorms(); }));
  const toggle = el("label", "check inline", `<input type="checkbox" checked> exclude position 0 from the colour range`);
  toggle.querySelector("input").onchange = e => { skipFirst = e.target.checked; drawNorms(); };
  norms.appendChild(toggle);
  norms.appendChild(holder);
  const res = d.norms.residual_output;
  const medianRest = row => { const r = row.slice(1).sort((a, b) => a - b); return r[Math.floor(r.length / 2)] ?? row[0]; };
  const ratios = res.map(row => row[0] / medianRest(row));
  const peak = ratios.indexOf(Math.max(...ratios));
  const mid = Math.floor(res.length / 2);
  norms.appendChild(observations([
    T > 1 && ratios[peak] > 3 ? `Position 0 ${tok(d.tokens[0])} reaches a residual norm <b>${ratios[peak].toFixed(0)}×</b> the median of the other positions (at layer ${peak}): a “massive activation”, typical of the token heads use as an ${term("attention sink")}.` : null,
    T > 1 ? `The residual norm grows with depth: median ${medianRest(res[0]).toFixed(1)} after layer 0, ${medianRest(res[mid]).toFixed(1)} after layer ${mid}, ${medianRest(res[res.length - 1]).toFixed(1)} after layer ${res.length - 1}. Each block adds to the stream, and RMSNorm makes the blocks indifferent to its overall scale.` : null,
  ]));
  out.appendChild(norms);
  drawNorms();
}
