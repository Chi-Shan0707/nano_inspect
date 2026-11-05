// Steering: add alpha * unit * v_hat to the residual stream; single comparison or alpha sweep against a control.
import { $, concept, el, esc, layerOptions, observations, persist, pct, presets, run, section, state, term, tok, tr } from "./core.js";
import { lineChart } from "./charts.js";

export function initSteer() {
  const root = $("#steer");
  root.innerHTML = "";
  root.appendChild(el("p", "lede", tr("<b>Question:</b> is there a direction in the residual stream that controls a behaviour, more than a random direction of the same size?",
    "<b>问题：</b>residual stream 中是否存在一个能控制某种行为的方向，且效果强于同样大小的随机方向？")));
  root.appendChild(concept("steer", tr("Concept: activation steering with controls", "概念：带对照的 activation steering"), tr(`
    <pre class="formula">h′ = h + α · unit · v̂        at one layer's residual stream, every position</pre>
    <ul>
      <li><b>v̂</b> is a unit direction. <i>Study C sentiment</i> is the mean difference between positive and negative review sentences, estimated in the steering study. <i>Custom</i> builds one from your own aligned prompt pairs (final-token residuals, positive minus negative). <i>Random</i> directions are the control.</li>
      <li><b>unit</b> is the typical residual-stream norm at that layer, so α is relative: α = 0.25 adds a vector a quarter as long as a typical activation.</li>
      <li><b>Policy</b>: <i>decode_time</i> keeps steering every generated token; <i>prefill_only</i> steers the prompt and then lets generation run free.</li>
    </ul>
    <p>The <b>sweep</b> measures, for each α, the <b>watch score</b> = log P(positive watch tokens) − log P(negative watch tokens) for the next token, and the ${term("KL divergence")} from the unsteered model.
    A real effect shows a steep watch curve for the direction and a flat one for the random control. KL shows the collateral damage.</p>
    <p class="muted">Controllability is not proof of a unique “sentiment feature”: the mean-difference direction can mix in style and format.</p>`, `
    <pre class="formula">h′ = h + α · unit · v̂        作用于某一层的 residual stream，所有位置</pre>
    <ul>
      <li><b>v̂</b> 是一个单位方向。<i>Study C sentiment</i> 是正面与负面评论句子之间的均值差，在 steering 实验中估计得到。<i>Custom</i> 根据你自己的成对 prompt 构建方向（最后一个 token 的 residual，正面减负面）。<i>Random</i> 方向作为对照。</li>
      <li><b>unit</b> 是该层 residual stream 的典型范数，因此 α 是相对值：α = 0.25 表示加上一个长度为典型 activation 四分之一的向量。</li>
      <li><b>Policy</b>：<i>decode_time</i> 在生成每个 token 时都持续 steering；<i>prefill_only</i> 只对 prompt 做 steering，之后让生成自由进行。</li>
    </ul>
    <p><b>Sweep</b> 对每个 α 测量 next token 的 <b>watch score</b> = log P(正面 watch token) − log P(负面 watch token)，以及与未 steering 模型之间的 ${term("KL divergence")}。
    真实的效应表现为：该方向的 watch 曲线陡峭，而随机对照的曲线平坦。KL 则反映附带的破坏。</p>
    <p class="muted">可控性并不能证明存在唯一的“sentiment feature”：均值差方向可能混入了风格和格式等因素。</p>`)));
  const form = el("form", "panel");
  form.id = "steer-form";
  form.innerHTML = `
    <label>Prompt<textarea name="prompt" rows="2"></textarea></label>
    <div class="row">
      <label>${tr("Direction", "方向")} <select name="direction"></select></label>
      <label>${tr("Layer", "层")} <select name="layer"></select></label>
      <label>${tr("α (single run)", "α（单次运行）")} <span class="range"><input name="alpha" type="range" min="-1" max="1" step="0.0625" value="0.25"><output>0.25</output></span></label>
      <label>${tr("Tokens", "生成 token 数")} <input name="steps" type="number" value="20" min="1" max="64"></label>
      <label>Policy <select name="policy"><option>decode_time</option><option>prefill_only</option></select></label>
      <label class="check"><input name="chat_template" type="checkbox"> chat template</label>
    </div>
    <div class="grid2 custom" hidden>
      <label>${tr("Positive prompts (one per line)", "正面 prompt（每行一个）")}<textarea name="positive" rows="4"></textarea></label>
      <label>${tr("Negative prompts (aligned with the positive lines)", "负面 prompt（与正面逐行对应）")}<textarea name="negative" rows="4"></textarea></label>
    </div>
    <div class="row">
      <label>${tr("Sweep α values", "Sweep 的 α 值")} <input name="alphas" value="-0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5" size="30"></label>
      <label>${tr("Control", "对照")} <select name="control"><option value="random_seed_0">${tr("random direction (seed 0)", "随机方向（seed 0）")}</option><option value="random_seed_1">${tr("random direction (seed 1)", "随机方向（seed 1）")}</option><option value="">${tr("none", "无")}</option></select></label>
      <label>${tr("Watch: positive words", "Watch：正面词")} <input name="watch_positive" value="good, great, wonderful, amazing" size="24" title="${tr("Comma-separated; each is read as a word-initial token (with a leading space)", "逗号分隔；每个词按词首 token（带前导空格）读取")}"></label>
      <label>${tr("Watch: negative words", "Watch：负面词")} <input name="watch_negative" value="bad, terrible, awful, boring" size="24"></label>
    </div>
    <div class="row"><button type="submit" name="single">${tr("Run once at α", "按 α 运行一次")}</button><button type="submit" name="sweep" class="secondary">${tr("Sweep α", "扫描 α")}</button></div>
    <div class="presets"></div>`;
  root.appendChild(form);
  const out = el("div", "out");
  root.appendChild(out);
  const f = form.elements;
  const s = state.status?.steering;
  f.direction.innerHTML = (s ? [["contrastive", tr("Study C sentiment direction", "Study C sentiment 方向")], ...s.random.map(r => [r, tr(`Study C control ${r}`, `Study C 对照 ${r}`)])] : [])
    .concat([["custom", tr("custom (from your prompts below)", "custom（来自下方你的 prompt）")], ["random_seed_0", tr("random direction (seed 0)", "随机方向（seed 0）")]])
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
    { label: tr("Movie review, sentiment direction", "影评，sentiment 方向"), values: { prompt: "I watched the movie last night and it was", direction: s ? "contrastive" : "custom", layer: "12", alpha: 0.25 } },
    { label: tr("Restaurant review", "餐厅点评"), values: { prompt: "The food at this restaurant was", direction: s ? "contrastive" : "custom", layer: "12", alpha: -0.25 } },
    { label: tr("Custom: formal vs casual", "Custom：正式 vs 随意"), hint: tr("Builds a direction from your own prompt pairs", "根据你自己的成对 prompt 构建方向"),
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

const dirName = d => ({ contrastive: "Study C sentiment", custom: tr("custom direction", "custom 方向") }[d] ||
  d.replace("random_seed_", tr("random seed ", "随机 seed ")).replace("random_", tr("Study C control ", "Study C 对照 ")));

function renderSingle(d, out) {
  const tops = r => `<table class="mini"><tr><th>next token</th><th class="num">P</th></tr>${r.top.map(x => `<tr><td class="mono">${tok(x.token)}</td><td class="num">${pct(x.prob)}</td></tr>`).join("")}</table>`;
  out.appendChild(el("div", "result-head", `${tr(`Layer ${d.layer}`, `第 ${d.layer} 层`)} · ${esc(dirName(d.direction))} · α = ${d.alpha} · ${tr("added vector norm", "所加向量的范数")} ${d.strength.toFixed(3)} (unit ${d.unit_value.toFixed(3)}: ${esc(d.unit_source)}) · ${esc(d.policy)} · ${d.elapsed_seconds}s`));
  const base = new Map(d.baseline.top.map(x => [x.token, x.prob]));
  const movers = d.steered.top.map(x => ({ t: x.token, delta: x.prob - (base.get(x.token) ?? 0) })).sort((a, b) => b.delta - a.delta);
  out.appendChild(observations([
    tr(`Steering changes the next-token distribution by KL = <b>${d.kl.toFixed(3)}</b> nats (0 = unchanged).`,
       `Steering 使 next-token 分布改变了 KL = <b>${d.kl.toFixed(3)}</b> nats（0 = 无变化）。`),
    movers[0]?.delta > 0 ? tr(`Biggest gain among the steered top-8: ${tok(movers[0].t)} (+${pct(movers[0].delta)}).`,
                              `在 steering 后的 top-8 中涨幅最大的是 ${tok(movers[0].t)}（+${pct(movers[0].delta)}）。`) : null,
    d.baseline.text === d.steered.text ? tr("The greedy continuation is unchanged. Try a larger |α| or a different layer.", "greedy 续写没有变化。试试更大的 |α| 或换一层。") : null,
  ], tr("One run proves little: use the sweep to compare against a random direction of the same size.", "单次运行说明不了什么：请用 sweep 与同样大小的随机方向对比。")));
  out.appendChild(el("div", "compare", `
    <div><h3>${tr("Without steering", "不加 steering")}</h3><div class="gen">${esc(d.baseline.text)}</div>${tops(d.baseline)}</div>
    <div><h3>${tr(`Steered (α = ${d.alpha})`, `加 steering（α = ${d.alpha}）`)}</h3><div class="gen">${esc(d.steered.text)}</div>${tops(d.steered)}</div>`));
}

function renderSweep(d, out) {
  const names = Object.keys(d.series);
  const alphas = d.series[names[0]].map(r => r.alpha);
  out.appendChild(el("div", "result-head", `${tr(`Layer ${d.layer}`, `第 ${d.layer} 层`)} · unit ${d.unit_value.toFixed(3)} (${esc(d.unit_source)}) · ${esc(d.policy)} · watch + ${d.watch.positive.map(tok).join(" ")} vs − ${d.watch.negative.map(tok).join(" ")} · ${d.elapsed_seconds}s`));
  const hasWatch = d.series[names[0]][0].watch != null;
  const items = [];
  if (hasWatch) {
    const slope = name => { const rows = d.series[name]; return rows[rows.length - 1].watch - rows[0].watch; };
    const main = slope(d.direction), ctrl = d.control ? slope(d.control) : null;
    const signed = v => `${v >= 0 ? "+" : ""}${v.toFixed(2)}`, range = `[${alphas[0]}, ${alphas[alphas.length - 1]}]`;
    const ratio = ctrl != null && Math.abs(ctrl) > 1e-6 ? Math.abs(main / ctrl).toFixed(1) : null;
    items.push(tr(`Across α ∈ ${range}, the watch score changes by <b>${signed(main)}</b> for the ${esc(dirName(d.direction))}` +
        (ctrl != null ? ` versus ${signed(ctrl)} for the ${esc(dirName(d.control))} (${ratio ? `${ratio}× larger` : "control flat"}).` : "."),
      `在 α ∈ ${range} 范围内，${esc(dirName(d.direction))} 的 watch score 变化了 <b>${signed(main)}</b>` +
        (ctrl != null ? `，而 ${esc(dirName(d.control))} 为 ${signed(ctrl)}（${ratio ? `大 ${ratio} 倍` : "对照曲线平坦"}）。` : "。")));
  } else items.push(tr("Give at least one positive and one negative watch token to score the sweep.", "请至少提供一个正面和一个负面 watch token，才能为 sweep 打分。"));
  const rows = d.series[d.direction];
  const extreme = rows.reduce((m, r) => (r.kl > m.kl ? r : m));
  items.push(tr(`The largest disturbance of the next-token distribution is KL ${extreme.kl.toFixed(3)} at α = ${extreme.alpha}. High KL with an off-topic continuation means the vector is breaking the model, not steering it.`,
    `对 next-token 分布的最大扰动是 α = ${extreme.alpha} 时的 KL ${extreme.kl.toFixed(3)}。如果 KL 很高且续写跑题，说明这个向量是在破坏模型，而不是在 steering。`));
  out.appendChild(observations(items));
  const charts = el("div", "grid2");
  out.appendChild(charts);
  const labels = alphas.map(a => String(a));
  if (hasWatch) lineChart(charts, { labels, xs: alphas, xLabel: "α", zeroLine: true, yLabel: tr("Watch score: log P(+) − log P(−) for the next token", "Watch score：next token 的 log P(+) − log P(−)"),
    series: names.map(n => ({ name: dirName(n), values: d.series[n].map(r => r.watch), dashed: n !== d.direction })), format: v => v.toFixed(1),
    refLine: alphas.indexOf(0) });
  lineChart(charts, { labels, xs: alphas, xLabel: "α", yLabel: tr("KL divergence from the unsteered model (nats)", "与未 steering 模型之间的 KL divergence（nats）"),
    series: names.map(n => ({ name: dirName(n), values: d.series[n].map(r => r.kl), dashed: n !== d.direction })), format: v => v.toFixed(2),
    domain: [0, Math.max(0.05, ...names.flatMap(n => d.series[n].map(r => r.kl))) * 1.1], refLine: alphas.indexOf(0) });
  const table = section(tr("Greedy continuations by α", "不同 α 下的 greedy 续写"), tr("Each cell is the model's continuation with that steering vector added.", "每个格子是加上该 steering 向量后模型的续写。"));
  table.appendChild(el("div", "scroll", `<table class="gens"><tr><th class="num">α</th>${names.map(n => `<th>${esc(dirName(n))}</th>`).join("")}</tr>${
    alphas.map((a, i) => `<tr${a === 0 ? ' class="zero"' : ""}><td class="num">${a}</td>${names.map(n => {
      const r = d.series[n][i];
      return `<td><div class="gen-cell">${esc(r.text)}</div><div class="muted small">${r.watch != null ? `watch ${r.watch.toFixed(2)} · ` : ""}KL ${r.kl.toFixed(3)} · next: ${r.top.map(x => tok(x.token)).join(" ")}</div></td>`;
    }).join("")}</tr>`).join("")}</table>`));
  out.appendChild(table);
}
