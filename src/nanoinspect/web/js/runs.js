// Saved reports from runs/.
import { $, api, el, esc, tr } from "./core.js";

const KIND = {
  inventory: ["Architecture atlas: every module, parameter and cache measurement (milestone M1).",
              "架构图谱：每个模块、参数与 cache 的测量（里程碑 M1）。"],
  trace: ["Computation microscope: one prompt through a gated-delta and a full-attention layer, with parity checks (M2).",
          "计算显微镜：一个 prompt 穿过一个 gated delta 层和一个 full attention 层，附带一致性校验（M2）。"],
  experiment: ["Controlled studies: retrieval behaviour, activation patching, steering with controls (M3).",
               "对照实验：retrieval 行为、activation patching、带对照的 steering（M3）。"],
  baseline: ["Deterministic baseline generation and its offline reproduction (M0).",
             "确定性的基线生成及其离线复现（M0）。"],
};

export async function renderRuns() {
  const root = $("#runs");
  root.innerHTML = `<p class="lede">${tr("<b>Saved reports.</b> These are the fixed, reproducible experiments behind <code>docs/findings.md</code>. Use them to check what you see interactively against the controlled results. Reports open in a new tab and link to their JSON and tensor artifacts.",
    "<b>已保存的报告。</b>这些是 <code>docs/findings.md</code> 背后固定且可复现的实验。可以用它们来核对你在交互中看到的现象与对照实验的结果是否一致。报告会在新标签页打开，并链接到对应的 JSON 和张量文件。（报告本身为英文。）")}</p>`;
  try {
    const runs = await api("/api/runs");
    root.appendChild(el("div", "runs", runs.map(r => `<div class="card run">
      <a href="${r.report}" target="_blank" rel="noopener">${esc(r.name)} ↗</a>
      <div class="muted small">${esc(r.kind || "run")} · ${new Date(r.modified * 1000).toLocaleString(tr(undefined, "zh-CN"))}</div>
      <p>${esc(KIND[r.kind] ? tr(...KIND[r.kind]) : "")}</p>
      <div class="run-thumbs">${r.figures.map(f => `<a href="${f}" target="_blank"><img src="${f}" alt="" loading="lazy"></a>`).join("")}</div></div>`).join("")));
  } catch (err) { root.appendChild(el("div", "error", esc(err.message))); }
}
