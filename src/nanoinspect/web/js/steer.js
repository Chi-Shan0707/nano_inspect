// Steering: add alpha * unit * v_hat to the residual stream; single comparison or alpha sweep against a control.
import { $, concept, el, esc, layerOptions, observations, persist, pct, presets, run, section, state, term, tok } from "./core.js";
import { lineChart } from "./charts.js";

export function initSteer() {
  const root = $("#steer");
  root.innerHTML = "";
  root.appendChild(el("p", "lede", "<b>Question:</b> is there a direction in the residual stream that controls a behaviour, more than a random direction of the same size?"));
  root.appendChild(concept("steer", "Concept: activation steering with controls", `
    <pre class="formula">h′ = h + α · unit · v̂        at one layer's residual stream, every position</pre>
    <ul>
      <li><b>v̂</b> is a unit direction. <i>Study C sentiment</i> is the mean difference between positive and negative review sentences, estimated in the steering study. <i>Custom</i> builds one from your own aligned prompt pairs (final-token residuals, positive minus negative). <i>Random</i> directions are the control.</li>
      <li><b>unit</b> is the typical residual-stream norm at that layer, so α is relative: α = 0.25 adds a vector a quarter as long as a typical activation.</li>
      <li><b>Policy</b>: <i>decode_time</i> keeps steering every generated token; <i>prefill_only</i> steers the prompt and then lets generation run free.</li>
    </ul>
    <p>The <b>sweep</b> measures, for each α, the <b>watch score</b> = log P(positive watch tokens) − log P(negative watch tokens) for the next token, and the ${term("KL divergence")} from the unsteered model.
    A real effect shows a steep watch curve for the direction and a flat one for the random control. KL shows the collateral damage.</p>
    <p class="muted">Controllability is not proof of a unique “sentiment feature”: the mean-difference direction can mix in style and format.</p>`));
  const form = el("form", "panel");
  form.id = "steer-form";
  form.innerHTML = `
    <label>Prompt<textarea name="prompt" rows="2"></textarea></label>
    <div class="row">
      <label>Direction <select name="direction"></select></label>
      <label>Layer <select name="layer"></select></label>
      <label>α (single run) <span class="range"><input name="alpha" type="range" min="-1" max="1" step="0.0625" value="0.25"><output>0.25</output></span></label>
      <label>Tokens <input name="steps" type="number" value="20" min="1" max="64"></label>
      <label>Policy <select name="policy"><option>decode_time</option><option>prefill_only</option></select></label>
      <label class="check"><input name="chat_template" type="checkbox"> chat template</label>
    </div>
    <div class="grid2 custom" hidden>
      <label>Positive prompts (one per line)<textarea name="positive" rows="4"></textarea></label>
      <label>Negative prompts (aligned with the positive lines)<textarea name="negative" rows="4"></textarea></label>
    </div>
    <div class="row">
      <label>Sweep α values <input name="alphas" value="-0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5" size="30"></label>
      <label>Control <select name="control"><option value="random_seed_0">random direction (seed 0)</option><option value="random_seed_1">random direction (seed 1)</option><option value="">none</option></select></label>
      <label>Watch: positive words <input name="watch_positive" value="good, great, wonderful, amazing" size="24" title="Comma-separated; each is read as a word-initial token (with a leading space)"></label>
      <label>Watch: negative words <input name="watch_negative" value="bad, terrible, awful, boring" size="24"></label>
    </div>
    <div class="row"><button type="submit" name="single">Run once at α</button><button type="submit" name="sweep" class="secondary">Sweep α</button></div>
    <div class="presets"></div>`;
  root.appendChild(form);
  const out = el("div", "out");
  root.appendChild(out);
  const f = form.elements;
  const s = state.status?.steering;
  f.direction.innerHTML = (s ? [["contrastive", "Study C sentiment direction"], ...s.random.map(r => [r, `Study C control ${r}`])] : [])
    .concat([["custom", "custom (from your prompts below)"], ["random_seed_0", "random direction (seed 0)"]])
    .map(([v, l]) => `<option value="${v}">${esc(l)}</option>`).join("");
  const fillLayers = () => {
    const saved = s && f.direction.value === "contrastive" ? new Set(s.layers) : null;
    const previous = f.layer.value;
    f.layer.innerHTML = layerOptions(i => !saved || saved.has(i));
    f.layer.value = [...f.layer.options].some(o => o.value === previous) ? previous : String(s?.selected_layer ?? 12);
  };
  const sync = () => { $(".custom", form).hidden = f.direction.value !== "custom"; fillLayers(); };
  f.direction.addEventListener("change", sync);
  f.alpha.addEventListener("input", () => { f.alpha.nextElementSibling.value = f.alpha.value; });
  f.prompt.value = "I watched the movie last night and it was";
  f.positive.value = "I love this\nWhat a wonderful day\nThis is excellent";
  f.negative.value = "I hate this\nWhat a terrible day\nThis is awful";
  sync();
  persist(form);
  sync();
  f.alpha.nextElementSibling.value = f.alpha.value;
  presets($(".presets", form), form, [
    { label: "Movie review, sentiment direction", values: { prompt: "I watched the movie last night and it was", direction: s ? "contrastive" : "custom", layer: "12", alpha: 0.25 } },
    { label: "Restaurant review", values: { prompt: "The food at this restaurant was", direction: s ? "contrastive" : "custom", layer: "12", alpha: -0.25 } },
    { label: "Custom: formal vs casual", hint: "Builds a direction from your own prompt pairs",
      values: { prompt: "Hey, about tomorrow's meeting,", direction: "custom", layer: "9", alpha: 0.5,
                positive: "Dear Sir, I am writing to inform you\nPlease find the attached document\nI would be grateful for your response",
                negative: "hey dude whats up\nlol check this out\ngonna be late sry",
                watch_positive: "Dear, Please, Sincerely, kindly", watch_negative: "lol, hey, gonna, dude" } },
  ]);
  let mode = "single";
  form.querySelector("button[name=single]").onclick = () => { mode = "single"; };
  form.querySelector("button[name=sweep]").onclick = () => { mode = "sweep"; };
  form.onsubmit = event => {
    event.preventDefault();
    if (event.submitter?.name) mode = event.submitter.name;
    const body = {
      prompt: f.prompt.value, layer: Number(f.layer.value), direction: f.direction.value, steps: Number(f.steps.value),
      policy: f.policy.value, chat_template: f.chat_template.checked, positive: f.positive.value.split("\n"), negative: f.negative.value.split("\n"),
    };
    if (mode === "sweep") {
      const split = v => v.split(",").map(x => x.trim()).filter(Boolean).map(x => " " + x);  // word-initial tokens carry a space
      Object.assign(body, { alphas: f.alphas.value.split(",").map(Number).filter(Number.isFinite), control: f.control.value,
        watch_positive: split(f.watch_positive.value), watch_negative: split(f.watch_negative.value), steps: Math.min(body.steps, 24) });
      run(form, out, "/api/sweep", body, d => renderSweep(d, out));
    } else {
      run(form, out, "/api/steer", { ...body, alpha: Number(f.alpha.value) }, d => renderSingle(d, out));
    }
  };
}

const dirName = d => ({ contrastive: "Study C sentiment", custom: "custom direction" }[d] || d.replace("random_seed_", "random seed ").replace("random_", "Study C control "));

function renderSingle(d, out) {
  const tops = r => `<table class="mini"><tr><th>next token</th><th class="num">P</th></tr>${r.top.map(x => `<tr><td class="mono">${tok(x.token)}</td><td class="num">${pct(x.prob)}</td></tr>`).join("")}</table>`;
  out.appendChild(el("div", "result-head", `Layer ${d.layer} · ${esc(dirName(d.direction))} · α = ${d.alpha} · added vector norm ${d.strength.toFixed(3)} (unit ${d.unit_value.toFixed(3)}: ${esc(d.unit_source)}) · ${esc(d.policy)} · ${d.elapsed_seconds}s`));
  const base = new Map(d.baseline.top.map(x => [x.token, x.prob]));
  const movers = d.steered.top.map(x => ({ t: x.token, delta: x.prob - (base.get(x.token) ?? 0) })).sort((a, b) => b.delta - a.delta);
  out.appendChild(observations([
    `Steering changes the next-token distribution by KL = <b>${d.kl.toFixed(3)}</b> nats (0 = unchanged).`,
    movers[0]?.delta > 0 ? `Biggest gain among the steered top-8: ${tok(movers[0].t)} (+${pct(movers[0].delta)}).` : null,
    d.baseline.text === d.steered.text ? "The greedy continuation is unchanged. Try a larger |α| or a different layer." : null,
  ], "One run proves little: use the sweep to compare against a random direction of the same size."));
  out.appendChild(el("div", "compare", `
    <div><h3>Without steering</h3><div class="gen">${esc(d.baseline.text)}</div>${tops(d.baseline)}</div>
    <div><h3>Steered (α = ${d.alpha})</h3><div class="gen">${esc(d.steered.text)}</div>${tops(d.steered)}</div>`));
}

function renderSweep(d, out) {
  const names = Object.keys(d.series);
  const alphas = d.series[names[0]].map(r => r.alpha);
  out.appendChild(el("div", "result-head", `Layer ${d.layer} · unit ${d.unit_value.toFixed(3)} (${esc(d.unit_source)}) · ${esc(d.policy)} · watch + ${d.watch.positive.map(tok).join(" ")} vs − ${d.watch.negative.map(tok).join(" ")} · ${d.elapsed_seconds}s`));
  const hasWatch = d.series[names[0]][0].watch != null;
  const items = [];
  if (hasWatch) {
    const slope = name => { const rows = d.series[name]; return rows[rows.length - 1].watch - rows[0].watch; };
    const main = slope(d.direction), ctrl = d.control ? slope(d.control) : null;
    items.push(`Across α ∈ [${alphas[0]}, ${alphas[alphas.length - 1]}], the watch score changes by <b>${main >= 0 ? "+" : ""}${main.toFixed(2)}</b> for the ${esc(dirName(d.direction))}` +
      (ctrl != null ? ` versus ${ctrl >= 0 ? "+" : ""}${ctrl.toFixed(2)} for the ${esc(dirName(d.control))} (${Math.abs(ctrl) > 1e-6 ? `${Math.abs(main / ctrl).toFixed(1)}× larger` : "control flat"}).` : "."));
  } else items.push("Give at least one positive and one negative watch token to score the sweep.");
  const rows = d.series[d.direction];
  const extreme = rows.reduce((m, r) => (r.kl > m.kl ? r : m));
  items.push(`The largest disturbance of the next-token distribution is KL ${extreme.kl.toFixed(3)} at α = ${extreme.alpha}. High KL with an off-topic continuation means the vector is breaking the model, not steering it.`);
  out.appendChild(observations(items));
  const charts = el("div", "grid2");
  out.appendChild(charts);
  const labels = alphas.map(a => String(a));
  if (hasWatch) lineChart(charts, { labels, xs: alphas, xLabel: "α", zeroLine: true, yLabel: "Watch score: log P(+) − log P(−) for the next token",
    series: names.map(n => ({ name: dirName(n), values: d.series[n].map(r => r.watch), dashed: n !== d.direction })), format: v => v.toFixed(1),
    refLine: alphas.indexOf(0) });
  lineChart(charts, { labels, xs: alphas, xLabel: "α", yLabel: "KL divergence from the unsteered model (nats)",
    series: names.map(n => ({ name: dirName(n), values: d.series[n].map(r => r.kl), dashed: n !== d.direction })), format: v => v.toFixed(2),
    domain: [0, Math.max(0.05, ...names.flatMap(n => d.series[n].map(r => r.kl))) * 1.1], refLine: alphas.indexOf(0) });
  const table = section("Greedy continuations by α", "Each cell is the model's continuation with that steering vector added.");
  table.appendChild(el("div", "scroll", `<table class="gens"><tr><th class="num">α</th>${names.map(n => `<th>${esc(dirName(n))}</th>`).join("")}</tr>${
    alphas.map((a, i) => `<tr${a === 0 ? ' class="zero"' : ""}><td class="num">${a}</td>${names.map(n => {
      const r = d.series[n][i];
      return `<td><div class="gen-cell">${esc(r.text)}</div><div class="muted small">${r.watch != null ? `watch ${r.watch.toFixed(2)} · ` : ""}KL ${r.kl.toFixed(3)} · next: ${r.top.map(x => tok(x.token)).join(" ")}</div></td>`;
    }).join("")}</tr>`).join("")}</table>`));
  out.appendChild(table);
}
