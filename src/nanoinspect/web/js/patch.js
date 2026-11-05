// Activation patching: clean -> corrupted, over (site, layer, position).
import { $, concept, el, esc, observations, persist, pct, plainTok, presets, run, section, state, term, tok, tokenStrip,
         tr, typeColor, cssVar } from "./core.js";
import { heatmap, lineChart } from "./charts.js";

const RETRIEVAL = "Facts:\nThe red box contains the apple.\nThe blue box contains the coin.\nQuestion: What does the blue box contain?\nAnswer: The blue box contains the";
// site -> [[label, note] in English, [label, note] in Chinese]
const SITES = {
  residual_output: [["residual stream (after the block)", "Everything the model has computed at that position up to this layer. Patching it shows <i>where</i> the information lives."],
                    ["residual stream（block 之后）", "模型截至该层在该位置计算出的一切。patch 它可以看出信息<i>在哪里</i>。"]],
  residual_input: [["residual stream (before the block)", "Same as the previous layer's output."],
                   ["residual stream（block 之前）", "与上一层的输出相同。"]],
  mixer_output: [["mixer output", "Only what this layer's attention or gated-delta mixer added. Shows <i>which mixers move</i> the information."],
                 ["mixer 输出", "只包含本层 attention 或 gated delta mixer 加上的部分。可以看出<i>哪些 mixer 搬运</i>了信息。"]],
  post_mixer_residual: [["mid-block residual", "Residual stream after the mixer, before the MLP."],
                        ["block 中间的 residual", "mixer 之后、MLP 之前的 residual stream。"]],
  mlp_output: [["MLP output", "Only what this layer's MLP added at that position. Shows <i>which MLPs compute</i> something that matters."],
               ["MLP 输出", "只包含本层 MLP 在该位置加上的部分。可以看出<i>哪些 MLP 计算</i>了重要的东西。"]],
};
const siteInfo = site => tr(...SITES[site]);

export function initPatch() {
  const root = $("#patch");
  root.innerHTML = "";
  root.appendChild(el("p", "lede", tr("<b>Question:</b> which internal activations actually <i>cause</i> the answer? Not just correlate with it.",
    "<b>问题：</b>哪些内部 activation 真正<i>导致</i>了答案，而不只是与之相关？")));
  root.appendChild(concept("patch", tr("Concept: activation patching (causal tracing)", "概念：activation patching（causal tracing）"), tr(`
    <ol>
      <li>Write a <b>clean</b> prompt and a <b>corrupted</b> one that differ in a single detail and tokenize to the same length (e.g. <i>coin</i> → <i>bell</i>). The model answers differently: <b>target</b> vs <b>foil</b>.</li>
      <li>Measure the ${term("logit difference")} LD = logit(target) − logit(foil) on both.</li>
      <li>Rerun the corrupted prompt many times. Each time, overwrite <b>one activation</b> (one site, layer and position) with its value from the clean run.</li>
      <li>Report ${term("restoration")} = (LD_patched − LD_corrupted) / (LD_clean − LD_corrupted).</li>
    </ol>
    <p>On the <b>residual stream</b> you usually see a path. Early layers carry the information at the position where the prompts differ; late layers carry it at the final position. The layer where it jumps is where a ${term("token mixer")} moves the fact to where the answer is produced. Patching <b>mixer</b> or <b>MLP</b> outputs then identifies which components do the moving and computing.</p>
    <p class="muted">Restoration measures sufficiency of one activation for this input pair. Redundant components can each look unimportant alone.</p>`, `
    <ol>
      <li>写一个 <b>clean</b> prompt 和一个 <b>corrupted</b> prompt，两者只有一处细节不同，且分词后长度相同（例如 <i>coin</i> → <i>bell</i>）。模型给出不同的答案：<b>target</b> 与 <b>foil</b>。</li>
      <li>在两者上分别测量 ${term("logit difference")} LD = logit(target) − logit(foil)。</li>
      <li>多次重新运行 corrupted prompt。每次把<b>一个 activation</b>（一个 site、一层、一个位置）覆盖为它在 clean 运行中的值。</li>
      <li>报告 ${term("restoration")} = (LD_patched − LD_corrupted) / (LD_clean − LD_corrupted)。</li>
    </ol>
    <p>在 <b>residual stream</b> 上通常能看到一条路径：浅层在两个 prompt 不同的位置携带信息，深层则在最后一个位置携带信息。信息发生跳转的那一层，就是 ${term("token mixer")} 把事实搬运到产生答案位置的地方。再对 <b>mixer</b> 或 <b>MLP</b> 输出做 patching，就能确定是哪些组件在搬运和计算。</p>
    <p class="muted">Restoration 衡量的是在这对输入上单个 activation 的充分性。冗余的组件单独看可能都显得不重要。</p>`)));
  const form = el("form", "panel");
  form.id = "patch-form";
  form.innerHTML = `
    <div class="grid2">
      <label>Clean prompt<textarea name="clean" rows="5"></textarea></label>
      <label>Corrupted prompt <span class="muted">${tr("(same number of tokens)", "（token 数相同）")}</span><textarea name="corrupted" rows="5"></textarea></label>
    </div>
    <div class="row">
      <label>Target <input name="target" placeholder="${tr("default: clean answer", "默认：clean 答案")}"></label>
      <label>Foil <input name="foil" placeholder="${tr("default: corrupted answer", "默认：corrupted 答案")}"></label>
      <fieldset class="sites"><legend>${tr("Sites to patch", "要 patch 的 site")}</legend>
        ${Object.keys(SITES).map(k => `<label class="check"><input type="checkbox" name="${k}" data-group="sites" ${k === "residual_output" ? "checked" : ""}> ${esc(siteInfo(k)[0])}</label>`).join("")}
      </fieldset>
      <button type="submit">${tr("Run", "运行")}</button></div>
    <div class="presets"></div>`;
  root.appendChild(form);
  const out = el("div", "out");
  root.appendChild(out);
  form.elements.clean.value = RETRIEVAL;
  form.elements.corrupted.value = RETRIEVAL.replace("contains the coin.", "contains the bell.");
  persist(form);
  presets($(".presets", form), form, [
    { label: tr("Fact retrieval: coin → bell", "事实检索：coin → bell"), values: { clean: RETRIEVAL, corrupted: RETRIEVAL.replace("contains the coin.", "contains the bell."), target: "", foil: "", residual_output: true, mixer_output: true, mlp_output: true } },
    { label: tr("Capital: France → Italy", "首都：France → Italy"), values: { clean: "The capital of France is the city of", corrupted: "The capital of Italy is the city of", target: " Paris", foil: " Rome", residual_output: true, mixer_output: false, mlp_output: true } },
    { label: tr("Who gets the drink? (name mover)", "谁拿到了饮料？（name mover）"), values: { clean: "When Mary and John went to the store, John gave a drink to", corrupted: "When Mary and John went to the store, Mary gave a drink to", target: " Mary", foil: " John", residual_output: true, mixer_output: true, mlp_output: false } },
  ]);
  form.onsubmit = event => {
    event.preventDefault();
    const sites = Object.keys(SITES).filter(k => form.elements[k].checked);
    run(form, out, "/api/patch", { clean: form.elements.clean.value, corrupted: form.elements.corrupted.value,
      target: form.elements.target.value, foil: form.elements.foil.value, sites }, d => render(d, out));
  };
}

function render(d, out) {
  const T = d.tokens.length, last = T - 1;
  const diff = new Set(d.differing_positions);
  const labels = d.tokens.map((t, i) => `${i} ${plainTok(t)}`);
  out.appendChild(el("div", "result-head",
    `LD = logit(${tok(d.target.token)}) − logit(${tok(d.foil.token)}) · clean <b>${d.ld_clean.toFixed(2)}</b> (P(target) ${pct(d.clean_probs.target)}) · corrupted <b>${d.ld_corrupted.toFixed(2)}</b> (P(target) ${pct(d.corrupted_probs.target)}) · ${d.elapsed_seconds}s`));
  out.appendChild(el("p", "caption", tr("Tokens that differ between the prompts are outlined in red; the clean token is shown on hover.", "两个 prompt 之间不同的 token 用红框标出；悬停可查看 clean token。")));
  out.appendChild(tokenStrip(d.tokens, { marks: Object.fromEntries([...diff].map(i => [i, "diff"])), title: i => diff.has(i) ? `clean: ${JSON.stringify(d.clean_tokens[i])} → corrupted: ${JSON.stringify(d.tokens[i])}` : tr(`position ${i}`, `位置 ${i}`) }));

  const items = [];
  if (d.ld_clean <= 0) items.push(tr(`The clean prompt does not prefer the target (LD ${d.ld_clean.toFixed(2)}): restoration is still defined, but check your target/foil.`,
    `clean prompt 并不偏好 target（LD ${d.ld_clean.toFixed(2)}）：restoration 仍然有定义，但请检查你的 target/foil。`));
  const res = d.grids.residual_output?.restoration;
  if (res) items.push(flowSummary(res, d, diff));
  for (const site of d.sites.filter(s => s === "mixer_output" || s === "mlp_output")) {
    const g = d.grids[site].restoration;
    const top = g.flatMap((row, l) => row.map((v, p) => ({ l, p, v }))).sort((a, b) => b.v - a.v).slice(0, 3);
    const attn = l => (state.layerTypes[l] === "full_attention" ? tr(" (attention)", "（attention）") : "");
    items.push(tr(`Strongest single ${siteInfo(site)[0]} patches: ${top.map(x => `layer ${x.l}${attn(x.l)} at ${x.p} ${tok(d.tokens[x.p])} → ${x.v.toFixed(2)}`).join("; ")}.`,
                  `最强的单个 ${siteInfo(site)[0]} patch：${top.map(x => `第 ${x.l} 层${attn(x.l)}，位置 ${x.p} ${tok(d.tokens[x.p])} → ${x.v.toFixed(2)}`).join("；")}。`));
  }
  out.appendChild(observations(items, tr("Restoration above 1 or below 0 is possible: a patch can overshoot or push toward the foil.",
    "Restoration 可能大于 1 或小于 0：patch 可能矫枉过正，也可能把模型推向 foil。")));

  const curves = el("div", "");
  let column = res ? bestColumn(res, diff, last) : last;
  for (const site of d.sites) {
    const [name, note] = siteInfo(site);
    const s = section(tr(`Patching the ${name}`, `对 ${name} 做 patching`), note + tr(" Rows are layers, columns positions. Click a column to plot it across layers.", "行是层，列是位置。点击某一列可绘制它随层的变化。"));
    const holder = el("div", "");
    s.appendChild(holder);
    out.appendChild(s);
    const g = d.grids[site];
    const draw = () => {
      holder.innerHTML = "";
      heatmap(holder, { rows: g.restoration.map((_, i) => i), cols: labels, values: g.restoration, scale: "div", domain: [-1, 1],
        rowColor: r => typeColor(d.layer_types[r]), colMark: c => (diff.has(c) ? cssVar("--critical") : null), selected: { c: column },
        format: v => v.toFixed(1), legend: tr("restoration (−1 … 1, clipped)", "restoration（−1 … 1，截断）"),
        tip: (r, c, v) => `${name}, ${tr(`layer ${r}`, `第 ${r} 层`)} (${d.layer_types[r] === "full_attention" ? "attention" : "gated delta"}), ${tr(`position ${c}`, `位置 ${c}`)}\nclean ${JSON.stringify(d.clean_tokens[c])} → corrupted ${JSON.stringify(d.tokens[c])}\nrestoration ${v.toFixed(3)}\npatched LD ${g.logit_difference[r][c].toFixed(3)}`,
        onClick: (_, c) => { column = c; redrawAll(); } });
    };
    s.draw = draw;
    draw();
  }
  const curveSection = section(tr("One position across layers", "单个位置随层的变化"), tr("Restoration at the selected position for each patched site.", "所选位置上每个被 patch 的 site 的 restoration。"));
  curveSection.appendChild(curves);
  out.appendChild(curveSection);
  const drawCurves = () => {
    curves.innerHTML = "";
    lineChart(curves, { labels: d.layer_types.map((_, i) => String(i)), xLabel: tr("layer", "层"), zeroLine: true,
      yLabel: tr(`Restoration when patching position ${column} ${plainTok(d.tokens[column])}`, `patch 位置 ${column} ${plainTok(d.tokens[column])} 时的 restoration`),
      series: d.sites.map(site => ({ name: siteInfo(site)[0], values: d.grids[site].restoration.map(row => row[column]) })),
      tip: i => `${tr(`layer ${i}`, `第 ${i} 层`)} (${d.layer_types[i] === "full_attention" ? "attention" : "gated delta"})\n` + d.sites.map(site => `${siteInfo(site)[0]}: ${d.grids[site].restoration[i][column].toFixed(3)}`).join("\n") });
  };
  const redrawAll = () => { out.querySelectorAll(".result-section").forEach(s => s.draw && s.draw()); drawCurves(); };
  drawCurves();
}

function bestColumn(grid, diff, last) {
  const first = [...diff][0];
  return first != null && first !== last ? first : last;
}

// Describe where the information sits by layer: argmax position of residual restoration.
function flowSummary(grid, d, diff) {
  const last = d.tokens.length - 1;
  const where = grid.map(row => {
    const best = row.reduce((b, v, p) => (v > row[b] ? p : b), 0);
    return { best, v: row[best], last: row[last], diffMax: Math.max(...[...diff].map(p => row[p] ?? -Infinity)) };
  });
  const atDiff = where.map(w => w.diffMax > 0.5);
  const atLast = where.map(w => w.last > 0.5);
  const lastDiffLayer = atDiff.lastIndexOf(true);
  const firstLastLayer = atLast.indexOf(true);
  const parts = [];
  if (lastDiffLayer >= 0) parts.push(tr(`patching the residual stream at the changed token restores the answer through layer <b>${lastDiffLayer}</b>`,
    `在被改动的 token 处 patch residual stream，直到第 <b>${lastDiffLayer}</b> 层都能恢复答案`));
  if (firstLastLayer >= 0) parts.push(tr(`patching it at the final token works from layer <b>${firstLastLayer}</b> on`,
    `在最后一个 token 处 patch，从第 <b>${firstLastLayer}</b> 层起有效`));
  let text = parts.length ? parts.join(tr("; ", "；")) + tr(".", "。") : tr("No single residual-stream patch restores more than half of the clean preference.",
    "没有任何单个 residual stream patch 能恢复超过一半的 clean 偏好。");
  if (lastDiffLayer >= 0 && firstLastLayer >= 0 && firstLastLayer <= lastDiffLayer + 2) {
    const movers = [];
    for (let l = Math.max(0, firstLastLayer - 1); l <= Math.min(lastDiffLayer + 1, d.layer_types.length - 1); l++) if (d.layer_types[l] === "full_attention") movers.push(l);
    const many = movers.length > 1;
    text += tr(` The information moves from the changed token to the final position around layers ${firstLastLayer - 1}–${firstLastLayer}` +
      (movers.length ? `, where attention layer${many ? "s" : ""} ${movers.join(", ")} sit${many ? "" : "s"}. Patch <i>mixer outputs</i> to test whether ${many ? "they" : "it"} move${many ? "" : "s"} it.` : "."),
      `信息大约在第 ${firstLastLayer - 1}–${firstLastLayer} 层从被改动的 token 移动到最后一个位置` +
      (movers.length ? `，attention 层 ${movers.join("、")} 正位于此处。对 <i>mixer 输出</i>做 patching，检验是否是${many ? "它们" : "它"}搬运了信息。` : "。"));
  }
  return text.charAt(0).toUpperCase() + text.slice(1);
}
