// Activation patching: clean -> corrupted, over (site, layer, position).
import { $, concept, el, esc, observations, persist, pct, plainTok, presets, run, section, state, term, tok, tokenStrip,
         typeColor, cssVar } from "./core.js";
import { heatmap, lineChart } from "./charts.js";

const RETRIEVAL = "Facts:\nThe red box contains the apple.\nThe blue box contains the coin.\nQuestion: What does the blue box contain?\nAnswer: The blue box contains the";
const SITE_INFO = {
  residual_output: ["residual stream (after the block)", "Everything the model has computed at that position up to this layer. Patching it shows <i>where</i> the information lives."],
  residual_input: ["residual stream (before the block)", "Same as the previous layer's output."],
  mixer_output: ["mixer output", "Only what this layer's attention or gated-delta mixer added. Shows <i>which mixers move</i> the information."],
  post_mixer_residual: ["mid-block residual", "Residual stream after the mixer, before the MLP."],
  mlp_output: ["MLP output", "Only what this layer's MLP added at that position. Shows <i>which MLPs compute</i> something that matters."],
};

export function initPatch() {
  const root = $("#patch");
  root.innerHTML = "";
  root.appendChild(el("p", "lede", "<b>Question:</b> which internal activations actually <i>cause</i> the answer? Not just correlate with it."));
  root.appendChild(concept("patch", "Concept: activation patching (causal tracing)", `
    <ol>
      <li>Write a <b>clean</b> prompt and a <b>corrupted</b> one that differ in a single detail and tokenize to the same length (e.g. <i>coin</i> → <i>bell</i>). The model answers differently: <b>target</b> vs <b>foil</b>.</li>
      <li>Measure the ${term("logit difference")} LD = logit(target) − logit(foil) on both.</li>
      <li>Rerun the corrupted prompt many times. Each time, overwrite <b>one activation</b> (one site, layer and position) with its value from the clean run.</li>
      <li>Report ${term("restoration")} = (LD_patched − LD_corrupted) / (LD_clean − LD_corrupted).</li>
    </ol>
    <p>On the <b>residual stream</b> you usually see a path. Early layers carry the information at the position where the prompts differ; late layers carry it at the final position. The layer where it jumps is where a ${term("token mixer")} moves the fact to where the answer is produced. Patching <b>mixer</b> or <b>MLP</b> outputs then identifies which components do the moving and computing.</p>
    <p class="muted">Restoration measures sufficiency of one activation for this input pair. Redundant components can each look unimportant alone.</p>`));
  const form = el("form", "panel");
  form.id = "patch-form";
  form.innerHTML = `
    <div class="grid2">
      <label>Clean prompt<textarea name="clean" rows="5"></textarea></label>
      <label>Corrupted prompt <span class="muted">(same number of tokens)</span><textarea name="corrupted" rows="5"></textarea></label>
    </div>
    <div class="row">
      <label>Target <input name="target" placeholder="default: clean answer"></label>
      <label>Foil <input name="foil" placeholder="default: corrupted answer"></label>
      <fieldset class="sites"><legend>Sites to patch</legend>
        ${Object.entries(SITE_INFO).map(([k, [label]]) => `<label class="check"><input type="checkbox" name="${k}" data-group="sites" ${k === "residual_output" ? "checked" : ""}> ${esc(label)}</label>`).join("")}
      </fieldset>
      <button type="submit">Run</button></div>
    <div class="presets"></div>`;
  root.appendChild(form);
  const out = el("div", "out");
  root.appendChild(out);
  form.elements.clean.value = RETRIEVAL;
  form.elements.corrupted.value = RETRIEVAL.replace("contains the coin.", "contains the bell.");
  persist(form);
  presets($(".presets", form), form, [
    { label: "Fact retrieval: coin → bell", values: { clean: RETRIEVAL, corrupted: RETRIEVAL.replace("contains the coin.", "contains the bell."), target: "", foil: "", residual_output: true, mixer_output: true, mlp_output: true } },
    { label: "Capital: France → Italy", values: { clean: "The capital of France is the city of", corrupted: "The capital of Italy is the city of", target: " Paris", foil: " Rome", residual_output: true, mixer_output: false, mlp_output: true } },
    { label: "Who gets the drink? (name mover)", values: { clean: "When Mary and John went to the store, John gave a drink to", corrupted: "When Mary and John went to the store, Mary gave a drink to", target: " Mary", foil: " John", residual_output: true, mixer_output: true, mlp_output: false } },
  ]);
  form.onsubmit = event => {
    event.preventDefault();
    const sites = Object.keys(SITE_INFO).filter(k => form.elements[k].checked);
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
  out.appendChild(el("p", "caption", "Tokens that differ between the prompts are outlined in red; the clean token is shown on hover."));
  out.appendChild(tokenStrip(d.tokens, { marks: Object.fromEntries([...diff].map(i => [i, "diff"])), title: i => diff.has(i) ? `clean: ${JSON.stringify(d.clean_tokens[i])} → corrupted: ${JSON.stringify(d.tokens[i])}` : `position ${i}` }));

  const items = [];
  if (d.ld_clean <= 0) items.push(`The clean prompt does not prefer the target (LD ${d.ld_clean.toFixed(2)}): restoration is still defined, but check your target/foil.`);
  const res = d.grids.residual_output?.restoration;
  if (res) items.push(flowSummary(res, d, diff));
  for (const site of d.sites.filter(s => s === "mixer_output" || s === "mlp_output")) {
    const g = d.grids[site].restoration;
    const top = g.flatMap((row, l) => row.map((v, p) => ({ l, p, v }))).sort((a, b) => b.v - a.v).slice(0, 3);
    items.push(`Strongest single ${SITE_INFO[site][0]} patches: ${top.map(x => `layer ${x.l}${state.layerTypes[x.l] === "full_attention" ? " (attention)" : ""} at ${x.p} ${tok(d.tokens[x.p])} → ${x.v.toFixed(2)}`).join("; ")}.`);
  }
  out.appendChild(observations(items, "Restoration above 1 or below 0 is possible: a patch can overshoot or push toward the foil."));

  const curves = el("div", "");
  let column = res ? bestColumn(res, diff, last) : last;
  for (const site of d.sites) {
    const [name, note] = SITE_INFO[site];
    const s = section(`Patching the ${name}`, note + " Rows are layers, columns positions. Click a column to plot it across layers.");
    const holder = el("div", "");
    s.appendChild(holder);
    out.appendChild(s);
    const g = d.grids[site];
    const draw = () => {
      holder.innerHTML = "";
      heatmap(holder, { rows: g.restoration.map((_, i) => i), cols: labels, values: g.restoration, scale: "div", domain: [-1, 1],
        rowColor: r => typeColor(d.layer_types[r]), colMark: c => (diff.has(c) ? cssVar("--critical") : null), selected: { c: column },
        format: v => v.toFixed(1), legend: "restoration (−1 … 1, clipped)",
        tip: (r, c, v) => `${name}, layer ${r} (${d.layer_types[r] === "full_attention" ? "attention" : "gated delta"}), position ${c}\nclean ${JSON.stringify(d.clean_tokens[c])} → corrupted ${JSON.stringify(d.tokens[c])}\nrestoration ${v.toFixed(3)}\npatched LD ${g.logit_difference[r][c].toFixed(3)}`,
        onClick: (_, c) => { column = c; redrawAll(); } });
    };
    s.draw = draw;
    draw();
  }
  const curveSection = section("One position across layers", "Restoration at the selected position for each patched site.");
  curveSection.appendChild(curves);
  out.appendChild(curveSection);
  const drawCurves = () => {
    curves.innerHTML = "";
    lineChart(curves, { labels: d.layer_types.map((_, i) => String(i)), xLabel: "layer", zeroLine: true,
      yLabel: `Restoration when patching position ${column} ${plainTok(d.tokens[column])}`,
      series: d.sites.map(site => ({ name: SITE_INFO[site][0], values: d.grids[site].restoration.map(row => row[column]) })),
      tip: i => `layer ${i} (${d.layer_types[i] === "full_attention" ? "attention" : "gated delta"})\n` + d.sites.map(site => `${SITE_INFO[site][0]}: ${d.grids[site].restoration[i][column].toFixed(3)}`).join("\n") });
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
  if (lastDiffLayer >= 0) parts.push(`patching the residual stream at the changed token restores the answer through layer <b>${lastDiffLayer}</b>`);
  if (firstLastLayer >= 0) parts.push(`patching it at the final token works from layer <b>${firstLastLayer}</b> on`);
  let text = parts.length ? parts.join("; ") + "." : "No single residual-stream patch restores more than half of the clean preference.";
  if (lastDiffLayer >= 0 && firstLastLayer >= 0 && firstLastLayer <= lastDiffLayer + 2) {
    const movers = [];
    for (let l = Math.max(0, firstLastLayer - 1); l <= Math.min(lastDiffLayer + 1, d.layer_types.length - 1); l++) if (d.layer_types[l] === "full_attention") movers.push(l);
    text += ` The information moves from the changed token to the final position around layers ${firstLastLayer - 1}–${firstLastLayer}` +
      (movers.length ? `, where attention layer${movers.length > 1 ? "s" : ""} ${movers.join(", ")} sit${movers.length > 1 ? "" : "s"}. Patch <i>mixer outputs</i> to test whether ${movers.length > 1 ? "they" : "it"} move${movers.length > 1 ? "" : "s"} it.` : ".");
  }
  return text.charAt(0).toUpperCase() + text.slice(1);
}
