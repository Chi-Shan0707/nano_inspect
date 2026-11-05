// Architecture: config, layer schedule, per-layer pipeline with shapes, parameters, cache scaling, notes.
import { $, api, concept, el, esc, fmtBytes, fmtInt, observations, section, state, term, tr, TYPE_LABEL, typeColor } from "./core.js";
import { lineChart } from "./charts.js";
import { markdown } from "./markdown.js";

// Plain-language meaning of each observed module, keyed by its path inside the block: [title, note, Chinese note].
// Titles are module names and stay in English.
const MODULE_NOTES = {
  input_layernorm: ["Input RMSNorm", "Rescales the residual stream to unit RMS, then applies a learned per-feature gain (1 + w). The mixer sees this normalized copy; the residual itself is untouched.",
    "把 residual stream 缩放到 RMS 为 1，再乘以可学习的逐特征增益 (1 + w)。mixer 看到的是这份归一化后的副本；residual 本身不受影响。"],
  "self_attn.q_proj": ["Query + gate projection", "One matrix produces, per head, a 256-d query and a 256-d output-gate logit (8 heads × 2 × 256 = 4,096).",
    "一个矩阵为每个 head 同时产生 256 维 query 和 256 维 output gate logit（8 个 head × 2 × 256 = 4,096）。"],
  "self_attn.q_norm": ["Query RMSNorm (QK-norm)", "Normalizes each head's query separately; keeps attention logits in a stable range.",
    "对每个 head 的 query 分别归一化，使 attention logit 保持在稳定范围内。"],
  "self_attn.k_proj": ["Key projection", "Only 2 key heads (GQA): 2 × 256 = 512 features.", "只有 2 个 key head（GQA）：2 × 256 = 512 个特征。"],
  "self_attn.k_norm": ["Key RMSNorm", "Per-head normalization of keys, matching the queries.", "对 key 逐 head 归一化，与 query 对应。"],
  "self_attn.v_proj": ["Value projection", "2 value heads × 256 = 512 features: what each position offers when attended to.",
    "2 个 value head × 256 = 512 个特征：每个位置被关注时所提供的内容。"],
  "self_attn.o_proj": ["Output projection", "Concatenated gated head outputs (8 × 256 = 2,048) back to the 1,024-d residual width.",
    "把拼接后的、经过门控的 head 输出（8 × 256 = 2,048）映射回 1,024 维的 residual 宽度。"],
  self_attn: ["Full-attention mixer (output)", "Its output is added to the residual stream.", "它的输出被加到 residual stream 上。"],
  "linear_attn.in_proj_qkv": ["q/k/v projection", "One matrix gives queries, keys and values for 16 heads × 128 each (3 × 2,048 = 6,144 features).",
    "一个矩阵同时给出 16 个 head × 128 维的 query、key 和 value（3 × 2,048 = 6,144 个特征）。"],
  "linear_attn.in_proj_z": ["Output-gate projection", "z (16 heads × 128): the SiLU(z) gate applied after the memory readout.",
    "z（16 个 head × 128）：在记忆读出之后施加的 SiLU(z) 门。"],
  "linear_attn.in_proj_b": ["Write-strength logits", "One number per head and token; β = sigmoid(b) decides how strongly this token writes.",
    "每个 head、每个 token 一个数；β = sigmoid(b) 决定这个 token 写入的强度。"],
  "linear_attn.in_proj_a": ["Decay logits", "One number per head and token; sets the retention exp(g) of the memory.",
    "每个 head、每个 token 一个数；决定记忆的 retention exp(g)。"],
  "linear_attn.norm": ["Gated RMSNorm", "Normalizes each head's 128-d readout and multiplies by SiLU(z). Runs on (tokens × heads) rows, hence the flattened shape.",
    "对每个 head 的 128 维读出做归一化，再乘以 SiLU(z)。它按（token × head）行运行，所以形状是展平的。"],
  "linear_attn.out_proj": ["Output projection", "16 heads × 128 = 2,048 → 1,024 residual width.", "16 个 head × 128 = 2,048 → 1,024 维 residual 宽度。"],
  linear_attn: ["Gated-delta mixer (output)", "Its output is added to the residual stream.", "它的输出被加到 residual stream 上。"],
  post_attention_layernorm: ["Post-mixer RMSNorm", "Normalizes h = x + mixer output before the MLP.", "在 MLP 之前对 h = x + mixer 输出做归一化。"],
  "mlp.gate_proj": ["MLP gate branch", "1,024 → 3,584; passed through SiLU.", "1,024 → 3,584；经过 SiLU。"],
  "mlp.act_fn": ["SiLU", "x · sigmoid(x): a smooth gate.", "x · sigmoid(x)：一个平滑的门。"],
  "mlp.up_proj": ["MLP up branch", "1,024 → 3,584; multiplied elementwise by SiLU(gate).", "1,024 → 3,584；与 SiLU(gate) 逐元素相乘。"],
  "mlp.down_proj": ["MLP down projection", "3,584 → 1,024, added to the residual stream.", "3,584 → 1,024，加到 residual stream 上。"],
  mlp: ["SwiGLU MLP (output)", "Per-position transformation; no mixing between tokens.", "逐位置的变换；token 之间不混合。"],
};

// Operations that run inside the mixer without their own module (so the inventory hook cannot observe them).
const HIDDEN_STEPS = {
  full_attention: {
    after: "self_attn.v_proj",
    steps: c => {
      const rot = Math.round(c.head_dim * (c.rope_parameters?.partial_rotary_factor ?? 1)), theta = fmtInt(c.rope_parameters?.rope_theta);
      const group = c.num_attention_heads / c.num_key_value_heads;
      return [
        ["Partial RoPE", tr(`Rotates the first ${rot} of ${c.head_dim} features of each q and k by position-dependent angles (θ = ${theta}).`,
                            `把每个 q 和 k 的 ${c.head_dim} 个特征中的前 ${rot} 个按与位置相关的角度旋转（θ = ${theta}）。`), "[token, head, 256] → same"],
        ["Grouped scores + causal softmax", tr(`Each of ${c.num_attention_heads} query heads uses KV head ⌊h / ${group}⌋. score = q·k/√${c.head_dim}; future positions masked; softmax over keys.`,
                                               `${c.num_attention_heads} 个 query head 中的每一个使用 KV head ⌊h / ${group}⌋。score = q·k/√${c.head_dim}；屏蔽未来位置；对 key 做 softmax。`), "→ [head 8, query, key]"],
        ["Weighted sum of values", tr("o = Σⱼ pⱼ vⱼ for every head.", "对每个 head 计算 o = Σⱼ pⱼ vⱼ。"), "→ [token, 8 × 256]"],
        ["Output gate", tr("o ⊙ sigmoid(gate): per feature, input-dependent.", "o ⊙ sigmoid(gate)：逐特征、依赖输入。"), "→ [token, 2,048]"],
      ];
    },
  },
  linear_attention: {
    after: "linear_attn.in_proj_a",
    steps: c => [
      ["Causal depthwise conv + SiLU", tr(`Each of the 6,144 q/k/v channels mixes the last ${c.linear_conv_kernel_dim} tokens (functional call on conv1d.weight).`,
                                          `6,144 个 q/k/v 通道中的每一个混合最近 ${c.linear_conv_kernel_dim} 个 token（对 conv1d.weight 的函数式调用）。`), "[token, 6,144] → same"],
      ["Normalize q, k", tr(`L2-normalize per head; scale q by 1/√${c.linear_key_head_dim}.`, `逐 head 做 L2 归一化；q 再乘以 1/√${c.linear_key_head_dim}。`), "→ 16 heads × 128"],
      ["Gates", tr("β = sigmoid(b) ∈ (0,1); g = −exp(A_log)·softplus(a + dt_bias) ≤ 0; retention exp(g).",
                   "β = sigmoid(b) ∈ (0,1)；g = −exp(A_log)·softplus(a + dt_bias) ≤ 0；retention 为 exp(g)。"), "→ [token, 16]"],
      ["Delta-rule memory", tr("Per head: S ← exp(g)·S; S ← S + β·k·(v − kᵀS)ᵀ; read o = qᵀS. S is a 128×128 matrix carried across tokens.",
                               "每个 head：S ← exp(g)·S；S ← S + β·k·(v − kᵀS)ᵀ；读出 o = qᵀS。S 是一个在 token 之间传递的 128×128 矩阵。"), "→ [token, 16, 128]"],
    ],
  },
};

export async function renderArchitecture() {
  const root = $("#architecture");
  const a = await api("/api/architecture");
  state.arch = a;
  state.layerTypes = (a.layers || []).map(l => l.type);
  const c = a.config || {};
  root.innerHTML = "";
  root.appendChild(el("p", "lede", tr(`<b>Question:</b> what is inside the model, how big is each part, and how does the data change shape on its way through one block?`,
    `<b>问题：</b>模型里有什么，每个部分有多大，数据穿过一个 block 时形状如何变化？`)));
  root.appendChild(concept("arch", tr("Concept: a hybrid stack of residual blocks", "概念：由 residual block 组成的混合堆叠"), tr(`
    <p>The text model is <code>embedding → 24 blocks → final ${term("RMSNorm")} → ${term("unembedding")}</code>. Every block adds a ${term("token mixer")} update and an MLP update to the ${term("residual stream")}. Blocks differ only in their mixer:</p>
    <ul>
      <li><b style="color:${typeColor("linear_attention")}">Gated delta</b> (layers ≠ 3 mod 4): a recurrent ${term("delta rule")} memory per head. Fixed size, so long contexts cost the same per token.</li>
      <li><b style="color:${typeColor("full_attention")}">Full attention</b> (layers 3, 7, …, 23): softmax attention with ${term("GQA")}, ${term("RoPE")} and an ${term("output gate")}. Exact lookup, with a ${term("KV cache")} that grows.</li>
    </ul>
    <p>Weight shapes are stored as <code>[out, in]</code>: a <code>[4096, 1024]</code> matrix maps 1,024 inputs to 4,096 outputs. Shapes below in the pipeline are <i>observed</i> on a 5-token input, so the token axis reads 5.</p>`, `
    <p>文本模型的结构是 <code>embedding → 24 个 block → final ${term("RMSNorm")} → ${term("unembedding")}</code>。每个 block 向 ${term("residual stream")} 加上一个 ${term("token mixer")} 更新和一个 MLP 更新。各 block 之间只有 mixer 不同：</p>
    <ul>
      <li><b style="color:${typeColor("linear_attention")}">Gated delta</b>（层号 ≠ 3 mod 4）：每个 head 一个循环的 ${term("delta rule")} 记忆。大小固定，因此长上下文中每个 token 的代价不变。</li>
      <li><b style="color:${typeColor("full_attention")}">Full attention</b>（第 3、7、…、23 层）：带 ${term("GQA")}、${term("RoPE")} 和 ${term("output gate")} 的 softmax attention。精确查找，但 ${term("KV cache")} 会不断增长。</li>
    </ul>
    <p>权重形状按 <code>[out, in]</code> 存储：<code>[4096, 1024]</code> 的矩阵把 1,024 个输入映射为 4,096 个输出。下面 pipeline 中的形状是在 5 个 token 的输入上<i>实际观测</i>到的，所以 token 轴为 5。</p>`)));

  const cards = [
    [tr("Text layers", "文本层数"), c.num_hidden_layers, "residual blocks"], [tr("Residual width", "Residual 宽度"), fmtInt(c.hidden_size), tr("numbers per token", "每个 token 的维数")],
    [tr("MLP width", "MLP 宽度"), fmtInt(c.intermediate_size), tr("hidden units per MLP", "每个 MLP 的隐藏单元")], [tr("Vocabulary", "词表"), fmtInt(c.vocab_size), "tokens"],
    ["Attention heads", `${c.num_attention_heads} q / ${c.num_key_value_heads} kv`, tr(`${c.head_dim}-d each (GQA ${c.num_attention_heads / c.num_key_value_heads}:1)`, `每个 ${c.head_dim} 维（GQA ${c.num_attention_heads / c.num_key_value_heads}:1）`)],
    ["Delta heads", `${c.linear_num_value_heads}`, tr(`${c.linear_key_head_dim}×${c.linear_value_head_dim} memory each`, `每个 ${c.linear_key_head_dim}×${c.linear_value_head_dim} 记忆`)],
    ["Conv kernel", c.linear_conv_kernel_dim, tr("tokens of local mixing", "个 token 的局部混合")],
    ["RoPE", tr(`${(c.rope_parameters?.partial_rotary_factor ?? 1) * 100}% dims`, `${(c.rope_parameters?.partial_rotary_factor ?? 1) * 100}% 维度`), `θ = ${fmtInt(c.rope_parameters?.rope_theta)}`],
    ["Embeddings", c.tie_word_embeddings ? "tied" : "untied", tr("input = output matrix", "输入矩阵 = 输出矩阵")],
  ];
  root.appendChild(el("div", "cards", cards.map(([k, v, s]) => `<div class="card stat"><div class="k">${esc(k)}</div><div class="v">${esc(v ?? "–")}</div><div class="s">${esc(s)}</div></div>`).join("")));
  if (!a.inventory) root.appendChild(el("p", "hint", tr("No inventory found; showing the config only. Run <code>nanoinspect inventory</code> for weights, dataflow and accounting.",
    "未找到 inventory，只显示配置。运行 <code>nanoinspect inventory</code> 可获得权重、数据流与参数统计。")));

  // Schedule
  const sched = section(tr("Layer schedule", "层排布"), tr("Click a layer to see its pipeline. Colour marks the mixer type; the pattern repeats every 4 layers.",
    "点击某一层查看它的 pipeline。颜色表示 mixer 类型；该模式每 4 层重复一次。"));
  const strip = el("div", "schedule");
  (a.layers || []).forEach(layer => {
    const b = el("button", "", String(layer.index));
    b.type = "button"; b.style.background = typeColor(layer.type);
    b.title = `${tr(`layer ${layer.index}`, `第 ${layer.index} 层`)}: ${TYPE_LABEL[layer.type] || layer.type}`;
    b.onclick = () => selectLayer(layer.index);
    strip.appendChild(b);
  });
  sched.appendChild(strip);
  sched.appendChild(el("div", "legend", `<span><i style="background:${typeColor("linear_attention")}"></i>gated delta: ${tr("fixed-size recurrent memory", "固定大小的循环记忆")}</span>` +
    `<span><i style="background:${typeColor("full_attention")}"></i>full attention: ${tr("KV cache grows with length", "KV cache 随长度增长")}</span>`));
  sched.appendChild(el("div", "", "")).id = "arch-layer";
  root.appendChild(sched);

  if (a.accounting) root.appendChild(renderParams(a));
  if (a.caches) renderCache(a, root);
  if (a.svgs?.length) {
    const diagrams = section(tr("Block diagrams", "Block 结构图"), tr("Generated by <code>nanoinspect inventory</code> from the loaded weights. Click to open full size.",
      "由 <code>nanoinspect inventory</code> 根据已加载的权重生成。点击查看原图。"));
    diagrams.appendChild(el("div", "svgs", a.svgs.map(s => `<a href="${s}" target="_blank"><img src="${s}" alt="${esc(s)}"></a>`).join("")));
    root.appendChild(diagrams);
  }
  const notesText = tr(a.notes, a.notes_zh || a.notes);
  if (notesText) {
    const notes = section(tr("Reading notes", "阅读笔记"), tr("From <code>docs/architecture.md</code>.", a.notes_zh ? "来自 <code>docs/architecture.zh.md</code>。" : "来自 <code>docs/architecture.md</code>。"));
    notes.appendChild(el("article", "notes prose", markdown(notesText)));
    root.appendChild(notes);
  }
  if (a.layers?.[0]?.weight_shapes) selectLayer(3);
}

function selectLayer(index) {
  const layer = state.arch.layers[index], c = state.arch.config;
  document.querySelectorAll("#architecture .schedule button").forEach((b, i) => b.classList.toggle("selected", i === index));
  const target = $("#arch-layer");
  if (!layer.weight_shapes) { target.innerHTML = ""; return; }
  const params = name => Object.entries(layer.weight_shapes).filter(([w]) => w.startsWith(name + ".")).reduce((s, [, sh]) => s + sh.reduce((p, x) => p * x, 1), 0);
  const shape = s => s ? `[${s.join(", ")}]` : "–";
  const hidden = HIDDEN_STEPS[layer.type];
  const rows = [];
  for (const f of layer.dataflow) {
    if (f.module.startsWith("model.")) continue;  // the block itself
    const [title, en, zh] = MODULE_NOTES[f.module] || [f.module, "", ""];
    const note = tr(en, zh);
    const p = params(f.module);
    const isSummary = !f.module.includes(".") && ["self_attn", "linear_attn", "mlp"].includes(f.module);
    rows.push(`<div class="pstep${isSummary ? " summary" : ""}"><div class="pname">${esc(title)} <code>${esc(f.module)}</code></div>
      <div class="pshape">${f.input.map(shape).join(" + ") || "–"} → ${shape(f.output)}</div>
      <div class="pnote">${esc(note)}${p ? ` <span class="muted">· ${fmtInt(p)} ${tr("params", "参数")}</span>` : ""}</div></div>`);
    if (hidden && f.module === hidden.after) {
      for (const [t, n, s] of hidden.steps(c)) rows.push(`<div class="pstep hidden-op"><div class="pname">${esc(t)} <span class="tag">${tr("inside the mixer", "mixer 内部")}</span></div><div class="pshape">${esc(s)}</div><div class="pnote">${esc(n)}</div></div>`);
    }
  }
  const mixer = layer.type === "full_attention" ? "self_attn" : "linear_attn";
  const weights = Object.entries(layer.weight_shapes).map(([name, s]) =>
    `<tr><td class="mono">${esc(name)}</td><td class="mono">[${s.join(", ")}]</td><td class="num">${fmtInt(s.reduce((p, x) => p * x, 1))}</td></tr>`).join("");
  target.innerHTML = `
    <div class="layer-head"><span class="dot" style="background:${typeColor(layer.type)}"></span>
      <b>${tr(`Layer ${index}`, `第 ${index} 层`)}</b> · ${esc(TYPE_LABEL[layer.type])} · <code>${esc(layer.mixer_class)}</code> ·
      ${fmtInt(layer.parameters)} ${tr("params", "参数")} (mixer ${fmtInt(params(mixer))}, MLP ${fmtInt(params("mlp"))}, norms ${fmtInt(params("input_layernorm") + params("post_attention_layernorm"))})</div>
    <p class="caption">${tr("Execution order, top to bottom. Shapes are [batch, tokens, features] from a 5-token probe. Grey steps happen inside the mixer without a separate module.",
      "按执行顺序从上到下排列。形状为 [batch, tokens, features]，来自一个 5 token 的探测输入。灰色步骤发生在 mixer 内部，没有单独的模块。")}</p>
    <div class="pipeline">${rows.join("")}</div>
    <details class="raw"><summary>${tr("All weight tensors of this layer", "本层的所有权重张量")}</summary>
      <div class="scroll"><table><tr><th>${tr("Tensor", "张量")}</th><th>${tr("Shape [out, in]", "形状 [out, in]")}</th><th class="num">${tr("Elements", "元素数")}</th></tr>${weights}</table></div></details>`;
}

function renderParams(a) {
  const acc = a.accounting, c = a.config;
  const comps = Object.entries(acc.by_component || {}).sort((x, y) => y[1].parameters - x[1].parameters);
  const total = acc.loaded_unique_parameters;
  const label = { embedding: tr("Embedding (tied with output)", "Embedding（与输出 tied）"), text_mixer: "Token mixers (24)", text_mlp: "MLPs (24)",
                  vision: tr("Vision tower (not used for text)", "Vision tower（文本任务不使用）"), other: tr("Norms & other", "Norm 及其他") };
  const s = section(tr("Where the parameters are", "参数分布在哪里"), tr(`Unique loaded parameters: ${fmtInt(total)}. The tied embedding is counted once.`,
    `去重后已加载的参数：${fmtInt(total)}。tied embedding 只计一次。`));
  const max = Math.max(...comps.map(([, v]) => v.parameters));
  s.appendChild(el("div", "barlist", comps.map(([k, v]) =>
    `<div class="barrow"><span class="bl">${esc(label[k] || k)}</span><span class="bt"><span class="bf" style="left:0;width:${(v.parameters / max * 100).toFixed(1)}%"></span></span>` +
    `<span class="bv">${fmtInt(v.parameters)} <span class="muted">${(v.parameters / total * 100).toFixed(1)}%</span></span></div>`).join("")));
  const layers = a.layers || [];
  const attn = layers.find(l => l.type === "full_attention"), delta = layers.find(l => l.type === "linear_attention");
  const textActive = acc.active_text_unique_parameters;
  const emb = acc.by_component.embedding.parameters;
  const share = (emb / textActive * 100).toFixed(0);
  s.appendChild(observations([
    tr(`The embedding matrix (${fmtInt(c.vocab_size)} × ${fmtInt(c.hidden_size)}) is <b>${share}%</b> of the text path. The vocabulary is huge relative to the width; in large models this share is small.`,
       `Embedding 矩阵（${fmtInt(c.vocab_size)} × ${fmtInt(c.hidden_size)}）占文本路径的 <b>${share}%</b>。相对于宽度，词表非常大；在大模型中这个比例很小。`),
    attn && delta ? tr(`A gated-delta block has <b>${fmtInt(delta.parameters)}</b> parameters, a full-attention block <b>${fmtInt(attn.parameters)}</b>. The difference is in the mixer; every MLP is identical (${fmtInt(3 * c.hidden_size * c.intermediate_size)}).`,
                       `一个 gated delta block 有 <b>${fmtInt(delta.parameters)}</b> 个参数，一个 full attention block 有 <b>${fmtInt(attn.parameters)}</b> 个。差别在 mixer；所有 MLP 都相同（${fmtInt(3 * c.hidden_size * c.intermediate_size)}）。`) : null,
    tr(`The checkpoint holds ${fmtInt(acc.checkpoint_tensor_parameters)} parameters. ${fmtInt(acc.reconciliation?.checkpoint_only_parameters)} belong to a multi-token-prediction head that the loader skips, so they never run.`,
       `checkpoint 中有 ${fmtInt(acc.checkpoint_tensor_parameters)} 个参数。其中 ${fmtInt(acc.reconciliation?.checkpoint_only_parameters)} 个属于 multi-token-prediction head，加载时被跳过，因此从不运行。`),
    tr(`Text-only inference executes ${fmtInt(textActive)} parameters; the ${fmtInt(acc.inactive_unique_parameters)}-parameter vision tower stays idle.`,
       `纯文本推理执行 ${fmtInt(textActive)} 个参数；拥有 ${fmtInt(acc.inactive_unique_parameters)} 个参数的 vision tower 处于闲置状态。`),
  ]));
  return s;
}

function renderCache(a, root) {
  const caches = a.caches;
  const s = section(tr("Cache: what generation must remember", "Cache：生成时必须记住什么"), tr(`Measured tensor storage after a prefill of each length (BF16). Not peak GPU memory.`,
    `在各长度 prefill 之后实测的张量存储（BF16）。不是 GPU 显存峰值。`));
  root.appendChild(s);
  s.appendChild(el("div", "scroll", `<table><tr><th class="num">Tokens</th><th class="num">KV ${tr("(6 attention layers)", "（6 个 attention 层）")}</th><th class="num">Recurrent state ${tr("(18 layers)", "（18 层）")}</th><th class="num">Conv state</th><th class="num">${tr("Total", "合计")}</th></tr>${caches.map(c =>
    `<tr><td class="num">${c.sequence_length}</td><td class="num">${fmtBytes(c.bytes_by_kind.kv || 0)}</td><td class="num">${fmtBytes(c.bytes_by_kind.recurrent || 0)}</td><td class="num">${fmtBytes(c.bytes_by_kind.conv || 0)}</td><td class="num">${fmtBytes(c.measured_storage_bytes)}</td></tr>`).join("")}</table>`));
  if (caches.length >= 2) {
    const first = caches[0], last = caches[caches.length - 1];
    const perToken = ((last.bytes_by_kind.kv || 0) - (first.bytes_by_kind.kv || 0)) / (last.sequence_length - first.sequence_length);
    const fixed = (first.bytes_by_kind.recurrent || 0) + (first.bytes_by_kind.conv || 0);
    const attnLayers = state.layerTypes.filter(t => t === "full_attention").length || 6;
    const allAttention = perToken / attnLayers * state.layerTypes.length;
    const lengths = Array.from({ length: 17 }, (_, i) => i * 256);  // evenly spaced: the x axis is linear
    const mib = v => v / 2 ** 20;
    lineChart(s, {
      labels: lengths.map(String), xLabel: tr("context length (tokens)", "上下文长度（tokens）"),
      yLabel: tr("Cache size (MiB), extrapolated from the measured per-token slope", "Cache 大小（MiB），按实测的每 token 斜率外推"),
      series: [{ name: tr("this hybrid (6 attention + 18 gated delta)", "本混合模型（6 attention + 18 gated delta）"), values: lengths.map(n => mib(fixed + perToken * n)) },
               { name: tr("hypothetical: all 24 layers full attention", "假设：24 层全部为 full attention"), values: lengths.map(n => mib(allAttention * n)), dashed: true }],
      format: v => v.toFixed(0), domain: [0, mib(allAttention * lengths[lengths.length - 1]) * 1.05],
      tip: i => `${fmtInt(lengths[i])} tokens\n${tr("hybrid", "混合")}: ${fmtBytes(fixed + perToken * lengths[i])}\n${tr("all attention", "全 attention")}: ${fmtBytes(allAttention * lengths[i])}`,
    });
    const crossover = Math.round(fixed / (allAttention - perToken));
    const kvh = state.arch.config.num_key_value_heads, hd = state.arch.config.head_dim;
    const at32k = fmtBytes(fixed + perToken * 32768), all32k = fmtBytes(allAttention * 32768);
    s.appendChild(observations([
      tr(`KV cache grows by <b>${fmtInt(perToken)} bytes per token</b> = 2 (K and V) × ${kvh} KV heads × ${hd} dims × 2 bytes × ${attnLayers} layers.`,
         `KV cache <b>每个 token 增长 ${fmtInt(perToken)} 字节</b> = 2（K 和 V）× ${kvh} 个 KV head × ${hd} 维 × 2 字节 × ${attnLayers} 层。`),
      tr(`The gated-delta state is a constant <b>${fmtBytes(fixed)}</b> however long the input is.`,
         `无论输入多长，gated delta 的状态始终是 <b>${fmtBytes(fixed)}</b>。`),
      tr(`The hybrid needs more cache than an all-attention model below about <b>${fmtInt(crossover)} tokens</b>. Beyond that it needs less. At 32k tokens it is ${at32k} versus ${all32k}.`,
         `在约 <b>${fmtInt(crossover)} 个 token</b> 以下，混合模型需要的 cache 比全 attention 模型多；超过之后则更少。在 32k token 时是 ${at32k}，对比 ${all32k}。`),
    ], tr("The straight-line extrapolation uses the measured slope; the hypothetical model is a formula, not a measurement.",
          "直线外推使用实测斜率；假设模型是按公式计算的，并非实测。")));
  }
}
