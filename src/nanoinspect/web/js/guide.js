// "Start here": the model in one picture, a learning path, how to use the workbench, glossary.
import { $, esc, term, GLOSSARY, goTo, typeColor, state } from "./core.js";

const PATH = [
  ["architecture", "1 · Architecture", "What is inside: 24 blocks of two kinds, where the 0.85B parameters sit, and why only 6 layers keep a growing cache.",
   "Click layer 3 and layer 0 to compare a full-attention block with a gated-delta block."],
  ["walkthrough", "2 · Layer walkthrough", "Follow one token through one mixer, operation by operation, with real numbers recomputed by hand.",
   "Run layer 15 on the retrieval prompt and find the step where the answer token gets its attention."],
  ["mixer", "3 · Mixer maps", "See every head at once: attention patterns for full attention, memory retention and write strength for gated delta.",
   "Compare head 5 of layer 15 with the others. Then open layer 14 and look for heads with a long half-life."],
  ["lens", "4 · Logit lens", "Read the residual stream at every depth and position to see when the prediction forms.",
   "On the retrieval prompt, find the first layer whose top-1 at the last position is ' coin'."],
  ["patch", "5 · Activation patching", "Move from correlation to causation: which (layer, position) carries the information that changes the answer?",
   "Patch the coin/bell pair and locate the layer where information jumps from the fact to the final token."],
  ["steer", "6 · Steering", "Add a direction to the residual stream and test whether behaviour moves more than with a random direction of equal size.",
   "Sweep α for the sentiment direction and compare its curve with the random control."],
];

export function renderGuide() {
  const root = $("#guide");
  const types = state.layerTypes.length ? state.layerTypes : Array.from({ length: 24 }, (_, i) => (i % 4 === 3 ? "full_attention" : "linear_attention"));
  const blocks = types.map((t, i) => `<span class="mini-block" style="background:${typeColor(t)}" title="layer ${i}: ${t === "full_attention" ? "full attention" : "gated delta"}">${i}</span>`).join("");
  root.innerHTML = `
  <div class="hero">
    <h2>Learn how a hybrid language model computes, by experimenting on it</h2>
    <p>This workbench runs <b>Qwen3.5-0.8B</b> locally and lets you open it up. Each view asks one question, explains the idea, runs the model on <i>your</i> input and summarizes what the result shows.</p>
  </div>

  <h2>The model in one picture</h2>
  <div class="pipeline-figure">
    <div class="pf-step"><b>Tokens</b><span>text → token ids</span></div><div class="pf-arrow">→</div>
    <div class="pf-step"><b>Embedding</b><span>248,320 × 1,024 lookup</span></div><div class="pf-arrow">→</div>
    <div class="pf-step wide"><b>24 residual blocks</b><div class="mini-schedule">${blocks}</div>
      <span><i class="sw" style="background:${typeColor("linear_attention")}"></i>gated delta ×18 &nbsp; <i class="sw" style="background:${typeColor("full_attention")}"></i>full attention ×6</span></div><div class="pf-arrow">→</div>
    <div class="pf-step"><b>Final RMSNorm</b><span>rescale</span></div><div class="pf-arrow">→</div>
    <div class="pf-step"><b>Unembedding</b><span>tied: same matrix as the embedding</span></div><div class="pf-arrow">→</div>
    <div class="pf-step"><b>Next-token probabilities</b><span>softmax over 248,320</span></div>
  </div>
  <div class="grid2">
    <div class="card prose">
      <h3>One block</h3>
      <pre class="formula">h = x + Mixer(RMSNorm(x))
y = h + MLP(RMSNorm(h))</pre>
      <p>Each token carries a 1,024-number vector, the ${term("residual stream")}. A block never replaces it; it <i>adds</i> two updates. The ${term("token mixer")} moves information <b>between positions</b>. The ${term("SwiGLU")} MLP transforms each position <b>on its own</b>. Because everything is additive, you can read the stream at any depth (logit lens) or replace a piece of it (patching, steering).</p>
    </div>
    <div class="card prose">
      <h3>Why two kinds of mixer?</h3>
      <p>${term("full attention")} looks back at every earlier token exactly, but its ${term("KV cache")} grows with every token. ${term("gated delta")} compresses the past into a fixed 128×128 matrix per head: constant memory, but lossy.</p>
      <p>Qwen3.5 repeats <b>three gated-delta blocks, then one attention block</b>. Cheap recurrent layers do most of the work; a few attention layers provide precise lookup. Expect retrieval to depend on the attention layers (3, 7, 11, 15, 19, 23).</p>
    </div>
  </div>

  <h2>Suggested learning path</h2>
  <div class="path">${PATH.map(([tab, title, what, task]) => `
    <div class="path-card">
      <h3>${esc(title)}</h3><p>${esc(what)}</p><p class="task"><b>Try:</b> ${esc(task)}</p>
      <button type="button" class="ghost" data-go="${tab}">Open →</button>
    </div>`).join("")}</div>

  <h2>How to use this workbench</h2>
  <div class="card prose">
    <ul>
      <li><b>Your own input everywhere.</b> Every prompt, layer, head, token and direction field is free to edit. <i>Examples</i> only pre-fill the form. Press <kbd>Run</kbd> or <kbd>Ctrl</kbd>+<kbd>Enter</kbd>. Your last inputs are remembered in this browser.</li>
      <li><b>Hover</b> any chart cell or point for exact values. Hover <span class="term" data-term="residual stream" tabindex="0">dotted terms</span> for definitions. Many charts are <b>clickable</b>: tokens, cells and columns select what the detail panels show.</li>
      <li><b>“What the data says”</b> boxes are generated from the current result. They describe what was measured; only patching and steering (with controls) support causal statements.</li>
      <li>Prompts are completions, not chat: the model continues your text. Tick <i>chat template</i> to wrap it as a user turn instead.</li>
      <li>Leading spaces matter in token fields: <code>" coin"</code> (with a space) and <code>"coin"</code> are different tokens. Spaces are shown as <code>·</code>.</li>
    </ul>
  </div>

  <h2>Glossary</h2>
  <dl class="glossary">${Object.entries(GLOSSARY).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>`;
  root.querySelectorAll("[data-go]").forEach(b => { b.onclick = () => goTo(b.dataset.go); });
}
