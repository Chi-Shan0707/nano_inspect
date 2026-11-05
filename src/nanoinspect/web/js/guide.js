// "Start here": the model in one picture, a learning path, how to use the workbench, glossary.
import { $, esc, term, GLOSSARY, gloss, goTo, typeColor, state, tr } from "./core.js";

// [tab, title, what, task], each text as [English, Chinese].
const PATH = [
  ["architecture", ["1 · Architecture", "1 · 架构"],
   ["What is inside: 24 blocks of two kinds, where the 0.85B parameters sit, and why only 6 layers keep a growing cache.",
    "模型里有什么：两种共 24 个 block，0.85B 参数分布在哪里，以及为什么只有 6 层需要不断增长的 cache。"],
   ["Click layer 3 and layer 0 to compare a full-attention block with a gated-delta block.",
    "点击第 3 层和第 0 层，对比 full attention block 与 gated delta block。"]],
  ["walkthrough", ["2 · Layer walkthrough", "2 · 逐层拆解"],
   ["Follow one token through one mixer, operation by operation, with real numbers recomputed by hand.",
    "跟随一个 token 穿过一个 mixer，逐个操作查看真实数值，并手动重算验证。"],
   ["Run layer 15 on the retrieval prompt and find the step where the answer token gets its attention.",
    "在 retrieval prompt 上运行第 15 层，找到答案 token 获得 attention 的那一步。"]],
  ["mixer", ["3 · Mixer maps", "3 · Mixer 图谱"],
   ["See every head at once: attention patterns for full attention, memory retention and write strength for gated delta.",
    "一次看到所有 head：full attention 的 attention pattern，gated delta 的记忆保留率（retention）与写入强度（write strength）。"],
   ["Compare head 5 of layer 15 with the others. Then open layer 14 and look for heads with a long half-life.",
    "把第 15 层的 head 5 与其他 head 对比。然后打开第 14 层，寻找 half-life 较长的 head。"]],
  ["lens", ["4 · Logit lens", "4 · Logit lens"],
   ["Read the residual stream at every depth and position to see when the prediction forms.",
    "在每个深度和位置读取 residual stream，观察预测在何时成形。"],
   ["On the retrieval prompt, find the first layer whose top-1 at the last position is ' coin'.",
    "在 retrieval prompt 上，找到最后一个位置的 top-1 首次变为 ' coin' 的那一层。"]],
  ["patch", ["5 · Activation patching", "5 · Activation patching"],
   ["Move from correlation to causation: which (layer, position) carries the information that changes the answer?",
    "从相关走向因果：哪个（层，位置）携带了改变答案的信息？"],
   ["Patch the coin/bell pair and locate the layer where information jumps from the fact to the final token.",
    "对 coin/bell 这对 prompt 做 patching，定位信息从事实所在位置跳到最后一个 token 的那一层。"]],
  ["steer", ["6 · Steering", "6 · Steering"],
   ["Add a direction to the residual stream and test whether behaviour moves more than with a random direction of equal size.",
    "向 residual stream 加入一个方向，检验行为的变化是否大于同等大小的随机方向。"],
   ["Sweep α for the sentiment direction and compare its curve with the random control.",
    "对 sentiment 方向扫描 α，并把它的曲线与随机对照比较。"]],
];

export function renderGuide() {
  const root = $("#guide");
  const types = state.layerTypes.length ? state.layerTypes : Array.from({ length: 24 }, (_, i) => (i % 4 === 3 ? "full_attention" : "linear_attention"));
  const blocks = types.map((t, i) => `<span class="mini-block" style="background:${typeColor(t)}" title="${tr(`layer ${i}`, `第 ${i} 层`)}: ${t === "full_attention" ? "full attention" : "gated delta"}">${i}</span>`).join("");
  root.innerHTML = tr(`
  <div class="hero">
    <h2>Learn how a hybrid language model computes, by experimenting on it</h2>
    <p>This workbench runs <b>Qwen3.5-0.8B</b> locally and lets you open it up. Each view asks one question, explains the idea, runs the model on <i>your</i> input and summarizes what the result shows.</p>
  </div>`, `
  <div class="hero">
    <h2>通过动手实验，理解一个混合架构语言模型如何计算</h2>
    <p>这个工作台在本地运行 <b>Qwen3.5-0.8B</b>，让你把它拆开来看。每个视图提出一个问题，解释其背后的概念，在<i>你自己的</i>输入上运行模型，并总结结果说明了什么。</p>
  </div>`) + `

  <h2>${tr("The model in one picture", "一图看懂模型")}</h2>
  <div class="pipeline-figure">
    <div class="pf-step"><b>Tokens</b><span>text → token ids</span></div><div class="pf-arrow">→</div>
    <div class="pf-step"><b>Embedding</b><span>${tr("248,320 × 1,024 lookup", "248,320 × 1,024 查表")}</span></div><div class="pf-arrow">→</div>
    <div class="pf-step wide"><b>${tr("24 residual blocks", "24 个 residual block")}</b><div class="mini-schedule">${blocks}</div>
      <span><i class="sw" style="background:${typeColor("linear_attention")}"></i>gated delta ×18 &nbsp; <i class="sw" style="background:${typeColor("full_attention")}"></i>full attention ×6</span></div><div class="pf-arrow">→</div>
    <div class="pf-step"><b>Final RMSNorm</b><span>${tr("rescale", "重新缩放")}</span></div><div class="pf-arrow">→</div>
    <div class="pf-step"><b>Unembedding</b><span>${tr("tied: same matrix as the embedding", "tied：与 embedding 是同一个矩阵")}</span></div><div class="pf-arrow">→</div>
    <div class="pf-step"><b>${tr("Next-token probabilities", "Next-token 概率")}</b><span>${tr("softmax over 248,320", "对 248,320 个 token 做 softmax")}</span></div>
  </div>
  <div class="grid2">
    <div class="card prose">
      <h3>${tr("One block", "一个 block")}</h3>
      <pre class="formula">h = x + Mixer(RMSNorm(x))
y = h + MLP(RMSNorm(h))</pre>
      ${tr(`<p>Each token carries a 1,024-number vector, the ${term("residual stream")}. A block never replaces it; it <i>adds</i> two updates. The ${term("token mixer")} moves information <b>between positions</b>. The ${term("SwiGLU")} MLP transforms each position <b>on its own</b>. Because everything is additive, you can read the stream at any depth (logit lens) or replace a piece of it (patching, steering).</p>`,
      `<p>每个 token 携带一个 1,024 维向量，即 ${term("residual stream")}。block 从不替换它，而是在上面<i>加上</i>两个更新。${term("token mixer")} 负责<b>在位置之间</b>搬运信息；${term("SwiGLU")} MLP 则<b>对每个位置单独</b>做变换。由于一切都是相加的，你可以在任意深度读取这条 stream（logit lens），或替换其中一部分（patching、steering）。</p>`)}
    </div>
    <div class="card prose">
      <h3>${tr("Why two kinds of mixer?", "为什么有两种 mixer？")}</h3>
      ${tr(`<p>${term("full attention")} looks back at every earlier token exactly, but its ${term("KV cache")} grows with every token. ${term("gated delta")} compresses the past into a fixed 128×128 matrix per head: constant memory, but lossy.</p>
      <p>Qwen3.5 repeats <b>three gated-delta blocks, then one attention block</b>. Cheap recurrent layers do most of the work; a few attention layers provide precise lookup. Expect retrieval to depend on the attention layers (3, 7, 11, 15, 19, 23).</p>`,
      `<p>${term("full attention")} 能精确回看之前的每个 token，但它的 ${term("KV cache")} 随每个 token 增长。${term("gated delta")} 把过去压缩进每个 head 一个固定的 128×128 矩阵：内存恒定，但有损。</p>
      <p>Qwen3.5 的结构是<b>三个 gated delta block 之后接一个 attention block</b>，循环往复。廉价的循环层承担大部分工作，少数 attention 层提供精确查找。可以预期，检索（retrieval）依赖这些 attention 层（3、7、11、15、19、23）。</p>`)}
    </div>
  </div>

  <h2>${tr("Suggested learning path", "建议学习路径")}</h2>
  <div class="path">${PATH.map(([tab, title, what, task]) => `
    <div class="path-card">
      <h3>${esc(tr(...title))}</h3><p>${esc(tr(...what))}</p><p class="task"><b>${tr("Try:", "试试：")}</b> ${esc(tr(...task))}</p>
      <button type="button" class="ghost" data-go="${tab}">${tr("Open →", "打开 →")}</button>
    </div>`).join("")}</div>

  <h2>${tr("How to use this workbench", "如何使用这个工作台")}</h2>
  <div class="card prose">
    <ul>${tr(`
      <li><b>Your own input everywhere.</b> Every prompt, layer, head, token and direction field is free to edit. <i>Examples</i> only pre-fill the form. Press <kbd>Run</kbd> or <kbd>Ctrl</kbd>+<kbd>Enter</kbd>. Your last inputs are remembered in this browser.</li>
      <li><b>Hover</b> any chart cell or point for exact values. Hover <span class="term" data-term="residual stream" tabindex="0">dotted terms</span> for definitions. Many charts are <b>clickable</b>: tokens, cells and columns select what the detail panels show.</li>
      <li><b>“What the data says”</b> boxes are generated from the current result. They describe what was measured; only patching and steering (with controls) support causal statements.</li>
      <li>Prompts are completions, not chat: the model continues your text. Tick <i>chat template</i> to wrap it as a user turn instead.</li>
      <li>Leading spaces matter in token fields: <code>" coin"</code> (with a space) and <code>"coin"</code> are different tokens. Spaces are shown as <code>·</code>.</li>`, `
      <li><b>处处都可以用你自己的输入。</b>每个 prompt、层、head、token 和方向字段都可以自由编辑。<i>示例</i>只是预先填好表单。点击<kbd>运行</kbd>或按 <kbd>Ctrl</kbd>+<kbd>Enter</kbd>。你最近的输入会保存在这个浏览器里。</li>
      <li>把鼠标<b>悬停</b>在任意图表格子或数据点上可以看到精确数值；悬停在<span class="term" data-term="residual stream" tabindex="0">带虚线的术语</span>上可以看到定义。很多图表都<b>可以点击</b>：token、格子和列会决定详情面板显示什么。</li>
      <li><b>“数据告诉我们什么”</b>框是根据当前结果自动生成的。它们描述测量到了什么；只有 patching 和（带对照的）steering 才支持因果结论。</li>
      <li>Prompt 是续写而非对话：模型会接着你的文本往下写。勾选 <i>chat template</i> 则会把它包装成一轮用户消息。</li>
      <li>token 字段里的前导空格很重要：<code>" coin"</code>（带空格）和 <code>"coin"</code> 是不同的 token。空格显示为 <code>·</code>。</li>`)}
    </ul>
  </div>

  <h2>${tr("Glossary", "术语表")}</h2>
  <dl class="glossary">${Object.keys(GLOSSARY).map(k => `<dt>${esc(k)}</dt><dd>${esc(gloss(k))}</dd>`).join("")}</dl>`;
  root.querySelectorAll("[data-go]").forEach(b => { b.onclick = () => goTo(b.dataset.go); });
}
