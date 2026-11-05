// Shared helpers: DOM, API, formatting, tooltip, glossary, persisted forms, presets.

export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
export const state = { status: null, arch: null, layerTypes: [] };

export const TYPE_LABEL = { linear_attention: "gated delta", full_attention: "full attention", embedding: "embedding" };

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
export const GLOSSARY = {
  "residual stream": "The 1,024-number vector each token carries from block to block. Blocks never overwrite it; each adds an update: x ← x + f(x). Everything the model 'knows' about a position at a given depth lives here.",
  "RMSNorm": "x / rms(x) · (1 + w): rescales a vector to unit root-mean-square, then applies a learned per-feature gain (Qwen3.5 stores w centred on zero). Applied before every mixer and MLP and once before the output.",
  "token mixer": "The half of a block that moves information between positions: full attention or gated delta. The MLP half works on each position independently.",
  "full attention": "Each position scores every earlier position with q·k/√d, softmaxes the scores and averages their values. Exact random access to the past; the KV cache grows with every token.",
  "gated delta": "Gated DeltaNet, a linear-attention mixer. Each head keeps a fixed 128×128 matrix memory S. Per token it decays S by exp(g), writes the delta-rule correction and reads with q. Constant cost and memory per token.",
  "delta rule": "S ← S + β·k·(v − kᵀS)ᵀ. Instead of adding v blindly, it writes only the error between v and what memory already returns for key k, so repeated facts do not pile up.",
  "GQA": "Grouped-query attention: 8 query heads share 2 key/value heads (4 queries per KV head). Queries stay diverse while the KV cache shrinks 4×.",
  "RoPE": "Rotary position embedding: rotates pairs of q/k features by angles proportional to position, so q·k depends on relative distance. Here only the first 64 of each head's 256 features rotate (partial_rotary_factor = 0.25).",
  "output gate": "Qwen3.5 multiplies each mixer's output elementwise by a learned, input-dependent gate before the output projection: sigmoid(g) in attention, SiLU(z) in gated delta. A gate can switch a head off token by token.",
  "SwiGLU": "The MLP: down(SiLU(gate(x)) ⊙ up(x)), 1,024 → 3,584 → 1,024. One branch decides how much of the other passes.",
  "logit": "Unnormalized score for one vocabulary token. Softmax over the 248,320 logits gives next-token probabilities; only differences between logits matter.",
  "unembedding": "Final RMSNorm, then a product with the (tied) embedding matrix: turns a 1,024-vector into 248,320 logits.",
  "logit lens": "Apply the unembedding to an intermediate residual stream to see what it would predict if the network stopped there. A probe, not ground truth: early layers were never trained to be read this way.",
  "logit difference": "logit(target) − logit(foil). One number for how strongly the model prefers the right answer over one specific wrong answer; it is unaffected by the rest of the vocabulary.",
  "activation patching": "Run a corrupted input, but replace one internal activation with its value from the clean input. If the answer recovers, that activation carries the information that differs between the inputs.",
  "steering": "Adding a fixed vector to the residual stream during the forward pass. If behaviour shifts along the vector's meaning, and random vectors of the same size do not shift it, the direction is causally usable.",
  "attention sink": "Many heads put most of their probability on the first token when nothing else is relevant. The first token becomes a parking spot whose value contributes little.",
  "KV cache": "Keys and values saved for every earlier token so generation need not recompute them. Grows linearly with length; only the 6 full-attention layers have one here.",
  "tied embeddings": "The input embedding matrix (248,320 × 1,024, 254M parameters) is reused as the output unembedding, so it is stored once.",
  "head": "An independent slice of a mixer with its own queries, keys and values (or its own memory matrix). Heads run in parallel and their outputs are concatenated.",
  "direct logit attribution": "Push one component's output straight through the unembedding (with the final norm's scale frozen) to see how much it directly raises one token's logit. Ignores effects through later layers.",
  "half-life": "Tokens until memory decays to half under a per-token retention r: ln 0.5 / ln r. r = 0.99 gives about 69 tokens; r = 0.5 gives 1.",
  "KL divergence": "How different two probability distributions are (≥ 0, 0 = identical). Used here to measure how much steering disturbs the next-token distribution.",
  "restoration": "(LD_patched − LD_corrupted) / (LD_clean − LD_corrupted). 1 means the patch fully restores the clean preference, 0 means no effect, negative means it pushes further toward the foil.",
};

// <span class="term">…</span> elements show their glossary entry on hover.
export function term(word, label = word) {
  return `<span class="term" data-term="${esc(word)}" tabindex="0">${esc(label)}</span>`;
}
document.addEventListener("mouseover", event => {
  const t = event.target.closest(".term");
  if (t && GLOSSARY[t.dataset.term]) showTip(event, `<b>${esc(t.dataset.term)}</b><br>${esc(GLOSSARY[t.dataset.term])}`, true);
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
  container.innerHTML = `<span class="label">Examples</span>`;
  for (const example of examples) {
    const b = el("button", "chip", esc(example.label));
    b.type = "button"; b.title = example.hint || "Fill the form with this example and run it";
    b.onclick = () => { setFields(form, example.values); form.requestSubmit(); };
    container.appendChild(b);
  }
  container.insertAdjacentHTML("beforeend", `<span class="label muted">or edit any field and press Run (Ctrl+Enter)</span>`);
}

// Run an analysis with button feedback and error display.
export async function run(form, out, path, body, render, label = "Run") {
  const buttons = $$("button[type=submit], button:not([type])", form).filter(b => !b.classList.contains("chip"));
  const button = buttons[0];
  const original = button ? button.textContent : label;
  buttons.forEach(b => { b.disabled = true; });
  if (button) button.textContent = "Running…";
  out.classList.add("busy");
  try {
    const data = await api(path, body);
    out.innerHTML = "";
    render(data);
  } catch (err) {
    out.innerHTML = `<div class="error"><b>Could not run.</b> ${esc(err.message)}</div>`;
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
  box.innerHTML = `<div class="obs-title">What the data says</div><ul>${items.filter(Boolean).map(i => `<li>${i}</li>`).join("")}</ul>` +
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
export function tokenStrip(tokens, { selected, onPick, marks = {}, title = i => `position ${i}` } = {}) {
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
