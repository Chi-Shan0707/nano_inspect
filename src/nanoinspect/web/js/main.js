// Entry point: tabs, model status, lazy initialization of each view.
import { $, $$, api, fmtBytes, state } from "./core.js";
import { renderGuide } from "./guide.js";
import { renderArchitecture } from "./architecture.js";
import { initWalkthrough } from "./walkthrough.js";
import { initMixer } from "./mixer.js";
import { initLens } from "./lens.js";
import { initPatch } from "./patch.js";
import { initSteer } from "./steer.js";
import { renderRuns } from "./runs.js";

const VIEWS = { guide: renderGuide, architecture: renderArchitecture, walkthrough: initWalkthrough, mixer: initMixer,
                lens: initLens, patch: initPatch, steer: initSteer, runs: renderRuns };
const ready = new Set();

async function show(tab) {
  if (!VIEWS[tab]) tab = "guide";
  $$("#tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === tab));
  $$(".tab").forEach(t => t.classList.toggle("active", t.id === tab));
  history.replaceState(null, "", `#${tab}`);
  if (!ready.has(tab) || tab === "runs") {
    ready.add(tab);
    try { await VIEWS[tab](); } catch (err) { $(`#${tab}`).innerHTML = `<div class="error">${err.message}</div>`; ready.delete(tab); }
  }
}

async function pollStatus() {
  let delay = 2000;
  try {
    const s = await api("/api/status");
    state.status = s;
    const pill = $("#status");
    pill.className = `pill ${s.status}`;
    if (s.status === "ready") {
      pill.textContent = `${s.model_id} · ${s.device} · ${s.dtype}`;
      pill.title = `revision ${s.revision}\nloaded in ${s.load_seconds}s${s.gpu ? `\n${s.gpu}` : ""}${s.allocated_bytes ? `\nGPU memory allocated ${fmtBytes(s.allocated_bytes)}` : ""}`;
      delay = 15000;
    } else pill.textContent = s.status === "error" ? `model error: ${s.error}` : `model ${s.status}…`;
  } catch {
    $("#status").className = "pill error"; $("#status").textContent = "server unreachable";
  }
  setTimeout(pollStatus, delay);
}

async function start() {
  $$("#tabs button").forEach(b => { b.onclick = () => show(b.dataset.tab); });
  // Layer types and steering assets are needed by several forms; fetch them before the first view.
  try {
    const [status, arch] = await Promise.all([api("/api/status"), api("/api/architecture")]);
    state.status = status; state.arch = arch;
    state.layerTypes = (arch.layers || []).map(l => l.type);
  } catch { /* the status pill reports connection problems */ }
  pollStatus();
  show(location.hash.slice(1) || "guide");
}

start();
