// Shared helpers: language, DOM, API, formatting, tooltip, glossary, persisted forms, presets.

export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
export const state = { status: null, arch: null, layerTypes: [] };

export const TYPE_LABEL = { linear_attention: "gated delta", full_attention: "full attention", embedding: "embedding" };

// ---------- language ----------
// English or Chinese. Key concepts (embedding, residual stream, logit lens, ...) stay in English in both.
const LANG_KEY = "nanoinspect:lang";
export let lang = localStorage.getItem(LANG_KEY) || (navigator.language?.toLowerCase().startsWith("zh") ? "zh" : "en");
export const tr = (en, zh) => (lang === "zh" ? zh : en);
export function setLang(next) {
  lang = next;
  localStorage.setItem(LANG_KEY, next);
  applyStatic();
}
// Static markup carries its Chinese text in data-zh; the English original is kept in data-en.
export function applyStatic(root = document) {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  for (const e of root.querySelectorAll("[data-zh]")) {
    e.dataset.en ??= e.textContent;
    e.textContent = lang === "zh" ? e.dataset.zh : e.dataset.en;
  }
  for (const e of root.querySelectorAll("[data-zh-title]")) {
    e.dataset.enTitle ??= e.title;
    e.title = lang === "zh" ? e.dataset.zhTitle : e.dataset.enTitle;
  }
}

export function esc(value) {
  return String(value).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
// Token text with whitespace made visible (· space, ↵ newline, ⇥ tab).
export function tok(text) {
  if (text === "") return "∅";
  return esc(String(text).replace(/ /g, "·").replace(/\n/g, "↵").replace(/\t/g, "⇥"));
}
export const plainTok = text => String(text).replace(/\n/g, "↵").replace(/\t/g, "⇥");
export const fmtInt = n => n == null ? "–" : Number(n).toLocaleString("en-US");
export function fmtBytes(n) {
  const units = ["B", "KiB", "MiB", "GiB"]; let i = 0;
  while (Math.abs(n) >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? (n < 10 ? 2 : 1) : 0)} ${units[i]}`;
}
export const pct = p => p === 0 ? "0%" : p >= 0.001 ? `${(p * 100).toFixed(p >= 0.1 ? 1 : 2)}%` : `${(p * 100).toExponential(1)}%`;
export const num = (v, d = 3) => v == null ? "–" : typeof v === "string" ? ({ "-inf": "−∞", inf: "∞", nan: "NaN" }[v] ?? v) : Number.isInteger(v) && Math.abs(v) < 1e7 ? String(v) : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : v.toFixed(d);
export const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const isDark = () => matchMedia("(prefers-color-scheme: dark)").matches;
export const typeColor = type => type === "full_attention" ? cssVar("--attn") : type === "embedding" ? cssVar("--muted") : cssVar("--delta");
export const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };

export async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} :
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

// ---------- tooltip ----------
const tooltip = () => $("#tooltip");
export function showTip(event, content, html = false) {
  const t = tooltip();
  if (html) t.innerHTML = content; else t.textContent = content;
  t.style.display = "block";
  const x = Math.min(event.clientX + 14, innerWidth - t.offsetWidth - 8);
  const y = event.clientY + 14 + t.offsetHeight > innerHeight ? event.clientY - t.offsetHeight - 10 : event.clientY + 14;
  t.style.left = `${Math.max(8, x)}px`; t.style.top = `${Math.max(8, y)}px`;
}
export const hideTip = () => { tooltip().style.display = "none"; };

// ---------- glossary ----------
// Each entry: [English, Chinese]. The term itself (the key) stays in English.
export const GLOSSARY = {
  "residual stream": ["The 1,024-number vector each token carries from block to block. Blocks never overwrite it; each adds an update: x ← x + f(x). Everything the model 'knows' about a position at a given depth lives here.",
    "每个 token 在 block 之间传递的 1,024 维向量。block 从不覆盖它，只往上加一个更新：x ← x + f(x)。模型在某个深度对某个位置“知道”的一切都在这里。"],
  "RMSNorm": ["x / rms(x) · (1 + w): rescales a vector to unit root-mean-square, then applies a learned per-feature gain (Qwen3.5 stores w centred on zero). Applied before every mixer and MLP and once before the output.",
    "x / rms(x) · (1 + w)：先把向量缩放到均方根为 1，再乘上可学习的逐特征增益（Qwen3.5 存储的 w 以 0 为中心）。每个 mixer 和 MLP 之前各用一次，输出前再用一次。"],
  "token mixer": ["The half of a block that moves information between positions: full attention or gated delta. The MLP half works on each position independently.",
    "block 中负责在不同位置之间搬运信息的那一半：full attention 或 gated delta。另一半 MLP 则对每个位置独立计算。"],
  "full attention": ["Each position scores every earlier position with q·k/√d, softmaxes the scores and averages their values. Exact random access to the past; the KV cache grows with every token.",
    "每个位置用 q·k/√d 给所有更早的位置打分，对分数做 softmax，再对它们的 value 加权平均。可以精确地随机访问过去；KV cache 随每个 token 增长。"],
  "gated delta": ["Gated DeltaNet, a linear-attention mixer. Each head keeps a fixed 128×128 matrix memory S. Per token it decays S by exp(g), writes the delta-rule correction and reads with q. Constant cost and memory per token.",
    "Gated DeltaNet，一种 linear attention mixer。每个 head 维护一个固定大小的 128×128 矩阵记忆 S。每个 token 先用 exp(g) 衰减 S，再写入 delta rule 修正，最后用 q 读取。每个 token 的计算量和内存都是常数。"],
  "delta rule": ["S ← S + β·k·(v − kᵀS)ᵀ. Instead of adding v blindly, it writes only the error between v and what memory already returns for key k, so repeated facts do not pile up.",
    "S ← S + β·k·(v − kᵀS)ᵀ。它不是盲目地加上 v，而是只写入 v 与记忆对 key k 已经返回的内容之间的误差，因此重复的事实不会不断累积。"],
  "GQA": ["Grouped-query attention: 8 query heads share 2 key/value heads (4 queries per KV head). Queries stay diverse while the KV cache shrinks 4×.",
    "Grouped-query attention：8 个 query head 共享 2 个 key/value head（每个 KV head 对应 4 个 query）。query 保持多样，而 KV cache 缩小为 1/4。"],
  "RoPE": ["Rotary position embedding: rotates pairs of q/k features by angles proportional to position, so q·k depends on relative distance. Here only the first 64 of each head's 256 features rotate (partial_rotary_factor = 0.25).",
    "Rotary position embedding：按与位置成正比的角度旋转 q/k 的特征对，使 q·k 只取决于相对距离。这里每个 head 的 256 个特征中只有前 64 个旋转（partial_rotary_factor = 0.25）。"],
  "output gate": ["Qwen3.5 multiplies each mixer's output elementwise by a learned, input-dependent gate before the output projection: sigmoid(g) in attention, SiLU(z) in gated delta. A gate can switch a head off token by token.",
    "Qwen3.5 在 output projection 之前，把每个 mixer 的输出逐元素乘以一个可学习、依赖输入的门：attention 中是 sigmoid(g)，gated delta 中是 SiLU(z)。门可以逐 token 地关闭某个 head。"],
  "SwiGLU": ["The MLP: down(SiLU(gate(x)) ⊙ up(x)), 1,024 → 3,584 → 1,024. One branch decides how much of the other passes.",
    "即 MLP：down(SiLU(gate(x)) ⊙ up(x))，1,024 → 3,584 → 1,024。一个分支决定另一个分支能通过多少。"],
  "logit": ["Unnormalized score for one vocabulary token. Softmax over the 248,320 logits gives next-token probabilities; only differences between logits matter.",
    "某个词表 token 的未归一化分数。对 248,320 个 logit 做 softmax 得到 next-token 概率；只有 logit 之间的差值才有意义。"],
  "unembedding": ["Final RMSNorm, then a product with the (tied) embedding matrix: turns a 1,024-vector into 248,320 logits.",
    "最后一个 RMSNorm，再乘以（tied）embedding 矩阵：把 1,024 维向量变成 248,320 个 logit。"],
  "logit lens": ["Apply the unembedding to an intermediate residual stream to see what it would predict if the network stopped there. A probe, not ground truth: early layers were never trained to be read this way.",
    "对中间层的 residual stream 应用 unembedding，看网络若在此处停下会预测什么。它只是探针而非真相：浅层从未被训练成可以这样读取。"],
  "logit difference": ["logit(target) − logit(foil). One number for how strongly the model prefers the right answer over one specific wrong answer; it is unaffected by the rest of the vocabulary.",
    "logit(target) − logit(foil)。用一个数衡量模型相对某个特定错误答案有多偏好正确答案；不受词表其余部分影响。"],
  "activation patching": ["Run a corrupted input, but replace one internal activation with its value from the clean input. If the answer recovers, that activation carries the information that differs between the inputs.",
    "运行 corrupted 输入，但把某一个内部 activation 替换成它在 clean 输入上的值。如果答案恢复了，说明这个 activation 携带着两个输入之间不同的那部分信息。"],
  "steering": ["Adding a fixed vector to the residual stream during the forward pass. If behaviour shifts along the vector's meaning, and random vectors of the same size do not shift it, the direction is causally usable.",
    "在前向传播时向 residual stream 加上一个固定向量。如果行为沿着该向量的含义发生变化，而同样大小的随机向量不会，那么这个方向在因果上是可用的。"],
  "attention sink": ["Many heads put most of their probability on the first token when nothing else is relevant. The first token becomes a parking spot whose value contributes little.",
    "当没有相关内容时，许多 head 会把大部分概率放在第一个 token 上。第一个 token 成了一个“停车位”，它的 value 贡献很小。"],
  "KV cache": ["Keys and values saved for every earlier token so generation need not recompute them. Grows linearly with length; only the 6 full-attention layers have one here.",
    "为每个之前的 token 保存的 key 和 value，这样生成时无需重新计算。随长度线性增长；这里只有 6 个 full attention 层有 KV cache。"],
  "tied embeddings": ["The input embedding matrix (248,320 × 1,024, 254M parameters) is reused as the output unembedding, so it is stored once.",
    "输入 embedding 矩阵（248,320 × 1,024，2.54 亿参数）同时被复用为输出 unembedding，因此只存储一次。"],
  "head": ["An independent slice of a mixer with its own queries, keys and values (or its own memory matrix). Heads run in parallel and their outputs are concatenated.",
    "mixer 中一个独立的切片，拥有自己的 query、key 和 value（或自己的记忆矩阵）。各 head 并行运行，输出拼接在一起。"],
  "direct logit attribution": ["Push one component's output straight through the unembedding (with the final norm's scale frozen) to see how much it directly raises one token's logit. Ignores effects through later layers.",
    "把某个组件的输出直接送入 unembedding（冻结最后一个 norm 的缩放），看它直接把某个 token 的 logit 抬高了多少。忽略经由后续层的间接效应。"],
  "half-life": ["Tokens until memory decays to half under a per-token retention r: ln 0.5 / ln r. r = 0.99 gives about 69 tokens; r = 0.5 gives 1.",
    "在每个 token 的保留率为 r 时，记忆衰减到一半所需的 token 数：ln 0.5 / ln r。r = 0.99 约为 69 个 token；r = 0.5 为 1 个。"],
  "KL divergence": ["How different two probability distributions are (≥ 0, 0 = identical). Used here to measure how much steering disturbs the next-token distribution.",
    "衡量两个概率分布差别有多大（≥ 0，0 表示完全相同）。这里用来衡量 steering 对 next-token 分布的扰动程度。"],
  "restoration": ["(LD_patched − LD_corrupted) / (LD_clean − LD_corrupted). 1 means the patch fully restores the clean preference, 0 means no effect, negative means it pushes further toward the foil.",
    "(LD_patched − LD_corrupted) / (LD_clean − LD_corrupted)。1 表示 patch 完全恢复了 clean 时的偏好，0 表示没有效果，负数表示把模型进一步推向 foil。"],
};
export const gloss = key => tr(...GLOSSARY[key]);

// <span class="term">…</span> elements show their glossary entry on hover.
export function term(word, label = word) {
  return `<span class="term" data-term="${esc(word)}" tabindex="0">${esc(label)}</span>`;
}
document.addEventListener("mouseover", event => {
  const t = event.target.closest(".term");
  if (t && GLOSSARY[t.dataset.term]) showTip(event, `<b>${esc(t.dataset.term)}</b><br>${esc(gloss(t.dataset.term))}`, true);
});
document.addEventListener("mouseout", event => { if (event.target.closest(".term")) hideTip(); });

// ---------- forms ----------
export function formData(form) {
  const out = {};
  for (const e of form.elements) {
    if (!e.name || e.type === "button" || e.type === "submit") continue;
    if (e.type === "checkbox" && e.dataset.group) { (out[e.dataset.group] ||= []); if (e.checked) out[e.dataset.group].push(e.name); continue; }
    out[e.name] = e.type === "checkbox" ? e.checked : (e.type === "number" || e.type === "range") ? Number(e.value) : e.value;
  }
  return out;
}

export function setFields(form, values) {
  for (const [key, value] of Object.entries(values)) {
    const field = form.elements.namedItem(key);
    if (!field) continue;
    if (field.type === "checkbox") field.checked = Boolean(value); else field.value = value;
    field.dispatchEvent(new Event("input", { bubbles: true }));
    field.dispatchEvent(new Event("change", { bubbles: true }));
  }
}

// Keep each form's last inputs across reloads, so your own prompts are not lost.
export function persist(form) {
  const key = `nanoinspect:${form.id}`;
  try {
    const saved = JSON.parse(localStorage.getItem(key) || "null");
    if (saved) for (const e of form.elements) {
      if (!e.name || !(e.name in saved)) continue;
      if (e.type === "checkbox") e.checked = saved[e.name]; else if (e.tagName !== "SELECT" || [...e.options].some(o => o.value === saved[e.name])) e.value = saved[e.name];
    }
  } catch { /* ignore corrupt storage */ }
  const save = () => {
    const values = {};
    for (const e of form.elements) if (e.name) values[e.name] = e.type === "checkbox" ? e.checked : e.value;
    localStorage.setItem(key, JSON.stringify(values));
  };
  form.addEventListener("input", save);
  form.addEventListener("change", save);
  // Ctrl/Cmd+Enter runs from any field.
  form.addEventListener("keydown", event => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) { event.preventDefault(); form.requestSubmit(); }
  });
}

// Example chips: fill the form (every field stays editable) and run.
export function presets(container, form, examples) {
  container.innerHTML = `<span class="label">${tr("Examples", "示例")}</span>`;
  for (const example of examples) {
    const b = el("button", "chip", esc(example.label));
    b.type = "button"; b.title = example.hint || tr("Fill the form with this example and run it", "用这个示例填写表单并运行");
    b.onclick = () => { setFields(form, example.values); form.requestSubmit(); };
    container.appendChild(b);
  }
  container.insertAdjacentHTML("beforeend", `<span class="label muted">${tr("or edit any field and press Run (Ctrl+Enter)", "或修改任意字段后点击运行（Ctrl+Enter）")}</span>`);
}

// Last result per form, so a view rebuilt after a language switch can show it again without rerunning the model.
const lastRuns = new Map();
let replaying = false;
document.addEventListener("submit", event => { event.target.dataset.submitter = event.submitter?.name || ""; }, true);

// Resubmit a freshly built form with its cached result (no-op if it has none).
export function replay(form) {
  const last = lastRuns.get(form.id);
  if (!last) return;
  replaying = true;
  try { form.requestSubmit(last.submitter ? form.querySelector(`button[name="${last.submitter}"]`) : null); } finally { replaying = false; }
}

// Run an analysis with button feedback and error display.
export async function run(form, out, path, body, render) {
  const cached = replaying ? lastRuns.get(form.id) : null;
  const buttons = $$("button[type=submit], button:not([type])", form).filter(b => !b.classList.contains("chip"));
  const button = buttons[0];
  const original = button ? button.textContent : "";
  buttons.forEach(b => { b.disabled = true; });
  if (button) button.textContent = tr("Running…", "运行中…");
  out.classList.add("busy");
  try {
    const data = cached?.path === path ? cached.data : await api(path, body);
    lastRuns.set(form.id, { path, data, submitter: form.dataset.submitter });
    out.innerHTML = "";
    render(data);
  } catch (err) {
    lastRuns.delete(form.id);
    out.innerHTML = `<div class="error"><b>${tr("Could not run.", "运行失败。")}</b> ${esc(err.message)}</div>`;
  } finally {
    buttons.forEach(b => { b.disabled = false; });
    if (button) button.textContent = original;
    out.classList.remove("busy");
  }
}

// Collapsible concept card that remembers whether you closed it.
export function concept(id, title, html) {
  const key = `nanoinspect:concept:${id}`;
  const open = localStorage.getItem(key) !== "closed";
  const d = el("details", "concept");
  d.open = open;
  d.innerHTML = `<summary>${esc(title)}</summary><div class="concept-body">${html}</div>`;
  d.addEventListener("toggle", () => localStorage.setItem(key, d.open ? "open" : "closed"));
  return d;
}

// "What the data says" box: generated observations, descriptive unless stated.
export function observations(items, note) {
  const box = el("div", "observations");
  box.innerHTML = `<div class="obs-title">${tr("What the data says", "数据告诉我们什么")}</div><ul>${items.filter(Boolean).map(i => `<li>${i}</li>`).join("")}</ul>` +
    (note ? `<div class="obs-note">${note}</div>` : "");
  return box;
}

export function section(title, caption) {
  const s = el("div", "result-section");
  s.innerHTML = `<h3>${esc(title)}</h3>` + (caption ? `<p class="caption">${caption}</p>` : "");
  return s;
}

export function segmented(options, active, onChange) {
  const wrap = el("div", "seg");
  wrap.setAttribute("role", "tablist");
  for (const [value, label] of options) {
    const b = el("button", value === active ? "active" : "", esc(label));
    b.type = "button";
    b.onclick = () => { $$("button", wrap).forEach(x => x.classList.toggle("active", x === b)); onChange(value); };
    wrap.appendChild(b);
  }
  return wrap;
}

// Clickable token chips; `marks` maps position -> extra class.
export function tokenStrip(tokens, { selected, onPick, marks = {}, title = i => tr(`position ${i}`, `位置 ${i}`) } = {}) {
  const strip = el("div", "tokens");
  tokens.forEach((t, i) => {
    const span = el("span", `tok${i === selected ? " sel" : ""}${marks[i] ? " " + marks[i] : ""}`);
    span.innerHTML = `<sub>${i}</sub>${tok(t)}`;
    span.title = title(i);
    if (onPick) span.onclick = () => onPick(i);
    strip.appendChild(span);
  });
  return strip;
}

export function layerOptions(filter) {
  return state.layerTypes.map((t, i) => [i, t]).filter(([i, t]) => !filter || filter(i, t))
    .map(([i, t]) => `<option value="${i}">${i} · ${TYPE_LABEL[t] || t}</option>`).join("");
}

export function goTo(tab) {
  const button = $(`#tabs button[data-tab="${tab}"]`);
  if (button) { button.click(); scrollTo({ top: 0 }); }
}
