// Saved reports from runs/.
import { $, api, el, esc } from "./core.js";

const KIND = {
  inventory: "Architecture atlas: every module, parameter and cache measurement (milestone M1).",
  trace: "Computation microscope: one prompt through a gated-delta and a full-attention layer, with parity checks (M2).",
  experiment: "Controlled studies: retrieval behaviour, activation patching, steering with controls (M3).",
  baseline: "Deterministic baseline generation and its offline reproduction (M0).",
};

export async function renderRuns() {
  const root = $("#runs");
  root.innerHTML = `<p class="lede"><b>Saved reports.</b> These are the fixed, reproducible experiments behind <code>docs/findings.md</code>. Use them to check what you see interactively against the controlled results. Reports open in a new tab and link to their JSON and tensor artifacts.</p>`;
  try {
    const runs = await api("/api/runs");
    root.appendChild(el("div", "runs", runs.map(r => `<div class="card run">
      <a href="${r.report}" target="_blank" rel="noopener">${esc(r.name)} ↗</a>
      <div class="muted small">${esc(r.kind || "run")} · ${new Date(r.modified * 1000).toLocaleString()}</div>
      <p>${esc(KIND[r.kind] || "")}</p>
      <div class="run-thumbs">${r.figures.map(f => `<a href="${f}" target="_blank"><img src="${f}" alt="" loading="lazy"></a>`).join("")}</div></div>`).join("")));
  } catch (err) { root.appendChild(el("div", "error", esc(err.message))); }
}
