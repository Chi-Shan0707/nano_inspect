// Entry point: tabs, language, model status, lazy initialization of each view.
import { $, $$, api, applyStatic, fmtBytes, lang, replay, setLang, state, tr } from "./core.js";
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
let current = "guide";

async function show(tab) {
  if (!VIEWS[tab]) tab = "guide";
  current = tab;
  $$("#tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === tab));
  $$(".tab").forEach(t => t.classList.toggle("active", t.id === tab));
  history.replaceState(null, "", `#${tab}`);
  if (!ready.has(tab) || tab === "runs") {
    ready.add(tab);
    try { await VIEWS[tab](); $$("form", $(`#${tab}`)).forEach(replay); } catch (err) { $(`#${tab}`).innerHTML = `<div class="error">${err.message}</div>`; ready.delete(tab); }
  }
}

const STATUS_ZH = { idle: "空闲", loading: "加载中" };

function renderStatus() {
  const s = state.status, pill = $("#status");
  if (!s) {
    pill.textContent = state.unreachable ? tr("server unreachable", "无法连接服务器") : tr("connecting…", "连接中…");
    pill.title = tr("model status", "模型状态");
    return;
  }
  pill.className = `pill ${s.status}`;
  if (s.status === "ready") {
    pill.textContent = `${s.model_id} · ${s.device} · ${s.dtype}`;
    pill.title = tr(`revision ${s.revision}\nloaded in ${s.load_seconds}s`, `版本 ${s.revision}\n加载耗时 ${s.load_seconds}s`) + (s.gpu ? `\n${s.gpu}` : "") +
      (s.allocated_bytes ? tr(`\nGPU memory allocated ${fmtBytes(s.allocated_bytes)}`, `\n已分配 GPU 显存 ${fmtBytes(s.allocated_bytes)}`) : "");
  } else {
    pill.textContent = s.status === "error" ? tr(`model error: ${s.error}`, `模型错误：${s.error}`) : tr(`model ${s.status}…`, `模型${STATUS_ZH[s.status] || s.status}…`);
    pill.title = tr("model status", "模型状态");
  }
}

async function pollStatus() {
  let delay = 2000;
  try {
    state.status = await api("/api/status");
    state.unreachable = false;
    if (state.status.status === "ready") delay = 15000;
  } catch {
    state.status = null; state.unreachable = true;
    $("#status").className = "pill error";
  }
  renderStatus();
  setTimeout(pollStatus, delay);
}

function renderLang() {
  applyStatic();
  $$("#lang [data-lang]").forEach(s => s.classList.toggle("active", s.dataset.lang === lang));
}

// Rebuild the open view in the new language; other views rebuild when next opened. Cached results are replayed.
function switchLang() {
  setLang(lang === "zh" ? "en" : "zh");
  renderLang();
  renderStatus();
  ready.clear();
  show(current);
}

async function start() {
  renderLang();
  renderStatus();
  $("#lang").onclick = switchLang;
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
