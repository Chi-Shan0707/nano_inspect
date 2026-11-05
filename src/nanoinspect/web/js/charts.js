// Canvas heatmaps, SVG line charts and small inline bars. One y axis per chart, always.
import { $, cssVar, esc, isDark, showTip, hideTip, el } from "./core.js";

// Usable width inside a chart frame; falls back to the page width if the container is not attached yet.
const widthOf = wrap => (wrap.isConnected && wrap.clientWidth ? wrap.clientWidth : ($("main")?.clientWidth || 1200) - 60) - 18;

// Sequential: one hue, near-zero recedes toward the surface. Diverging: red <- gray -> blue.
const RAMPS = {
  light: { seq: ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
           div: ["#a32a2a", "#e34948", "#f0efec", "#3987e5", "#184f95"] },
  dark: { seq: ["#1a1a19", "#104281", "#1c5cab", "#2a78d6", "#5598e7", "#9ec5f4"],
          div: ["#f09090", "#c24a4a", "#383835", "#2a78d6", "#86b6ef"] },
};
export const SERIES = () => isDark() ? ["#3987e5", "#d95926", "#199e70", "#c98500"] : ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"];
const hexRgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
export function ramp(stops, t) {
  t = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
  const x = t * (stops.length - 1), i = Math.min(Math.floor(x), stops.length - 2), f = x - i;
  const a = hexRgb(stops[i]), b = hexRgb(stops[i + 1]);
  return a.map((v, k) => Math.round(v + (b[k] - v) * f));
}
const rgb = c => `rgb(${c.join(",")})`;
const luminance = c => (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255;

/**
 * Heatmap on canvas.
 * opts: rows, cols (labels); values[r][c]; scale "seq"|"div"; domain [lo, hi]; format(v);
 *       tip(r, c, v) -> string; skip(r, c); rowColor(r); colMark(c) -> css color for the label;
 *       text(r, c) -> string drawn in the cell; cell / cellHeight / minCell / maxCell; onClick(r, c);
 *       selected {r?, c?} outline; legend; compact (no labels, for small multiples).
 */
export function heatmap(container, opts) {
  const wrap = el("div", opts.compact ? "chart compact" : "chart");
  const canvas = document.createElement("canvas");
  wrap.appendChild(canvas);
  container.appendChild(wrap);
  const { rows, cols, values } = opts;
  const compact = Boolean(opts.compact);
  const rowLabelWidth = compact ? 0 : opts.rowLabelWidth || 64;
  const colLabelHeight = compact || opts.colLabels === false ? 4 : opts.colLabelHeight || 64;
  const legendHeight = compact || opts.legend === false ? 0 : 36;
  const available = Math.max(compact ? 60 : 300, (opts.width || widthOf(wrap)) - rowLabelWidth);
  const cell = opts.cell || Math.max(opts.minCell || 4, Math.min(opts.maxCell || 28, Math.floor(available / cols.length)));
  const cellH = opts.cellHeight || (compact ? cell : Math.max(cell, 13));
  const width = rowLabelWidth + cell * cols.length + 2, height = colLabelHeight + cellH * rows.length + legendHeight;
  const dpr = devicePixelRatio || 1;
  canvas.width = width * dpr; canvas.height = height * dpr;
  canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const stops = RAMPS[isDark() ? "dark" : "light"][opts.scale === "div" ? "div" : "seq"];
  const [lo, hi] = opts.domain;
  const ink = cssVar("--ink-2"), muted = cssVar("--muted"), surface = cssVar("--surface"), strong = cssVar("--ink");
  const format = opts.format || (v => v.toFixed(3));
  const fontSize = Math.min(11, Math.max(8, cellH - 4));
  ctx.textBaseline = "middle";
  rows.forEach((label, r) => {
    const y = colLabelHeight + r * cellH;
    if (opts.rowColor && !compact) { ctx.fillStyle = opts.rowColor(r); ctx.fillRect(0, y + 2, 4, cellH - 4); }
    if (!compact && (cellH >= 11 || r % Math.ceil(11 / cellH) === 0)) {
      ctx.font = "11px ui-monospace, monospace"; ctx.fillStyle = ink; ctx.textAlign = "right";
      ctx.fillText(String(label).slice(0, 9), rowLabelWidth - 6, y + cellH / 2);
    }
    values[r].forEach((v, c) => {
      const x = rowLabelWidth + c * cell;
      if (opts.skip && opts.skip(r, c)) { ctx.fillStyle = surface; ctx.fillRect(x, y, cell, cellH); return; }
      const color = ramp(stops, (v - lo) / (hi - lo || 1));
      ctx.fillStyle = rgb(color);
      ctx.fillRect(x, y, cell - (cell > 8 ? 1 : 0), cellH - (cellH > 8 ? 1 : 0));
      if (opts.text) {
        const text = opts.text(r, c);
        if (text) {
          ctx.font = `${fontSize}px ui-monospace, monospace`; ctx.textAlign = "center";
          ctx.fillStyle = luminance(color) > 0.55 ? "#0b0b0b" : "#ffffff";
          const maxChars = Math.max(1, Math.floor((cell - 4) / (fontSize * 0.62)));
          ctx.fillText(text.length > maxChars ? text.slice(0, maxChars - 1) + "…" : text, x + cell / 2, y + cellH / 2 + 0.5);
        }
      }
    });
  });
  if (opts.selected) {
    const { r, c } = opts.selected;
    ctx.strokeStyle = strong; ctx.lineWidth = 2;
    if (r != null && c != null) ctx.strokeRect(rowLabelWidth + c * cell + 1, colLabelHeight + r * cellH + 1, cell - 2, cellH - 2);
    else if (c != null) ctx.strokeRect(rowLabelWidth + c * cell + 1, colLabelHeight + 1, cell - 2, rows.length * cellH - 2);
    else if (r != null) ctx.strokeRect(rowLabelWidth + 1, colLabelHeight + r * cellH + 1, cols.length * cell - 2, cellH - 2);
  }
  if (!compact && opts.colLabels !== false) {
    const every = Math.ceil(12 / cell);
    ctx.font = "11px ui-monospace, monospace"; ctx.textAlign = "left";
    cols.forEach((label, c) => {
      if (c % every) return;
      ctx.save();
      ctx.translate(rowLabelWidth + c * cell + cell / 2 + 3, colLabelHeight - 5);
      ctx.rotate(-Math.PI / 3);
      ctx.fillStyle = opts.colMark && opts.colMark(c) ? opts.colMark(c) : ink;
      ctx.fillText(String(label).slice(0, 12), 0, 0);
      ctx.restore();
    });
  }
  if (legendHeight) {
    const ly = colLabelHeight + rows.length * cellH + 12, lw = Math.min(200, Math.max(120, width - rowLabelWidth - 200));
    for (let i = 0; i < lw; i++) { ctx.fillStyle = rgb(ramp(stops, i / (lw - 1))); ctx.fillRect(rowLabelWidth + i, ly, 1, 8); }
    ctx.font = "11px system-ui, sans-serif"; ctx.fillStyle = muted; ctx.textAlign = "left";
    ctx.fillText(format(lo), rowLabelWidth, ly + 19);
    ctx.textAlign = "right"; ctx.fillText(format(hi), rowLabelWidth + lw, ly + 19);
    if (opts.legend) { ctx.textAlign = "left"; ctx.fillStyle = ink; ctx.fillText(opts.legend, rowLabelWidth + lw + 12, ly + 4); }
  }
  const locate = event => {
    const rect = canvas.getBoundingClientRect();
    const c = Math.floor((event.clientX - rect.left - rowLabelWidth) / cell);
    const r = Math.floor((event.clientY - rect.top - colLabelHeight) / cellH);
    return r >= 0 && c >= 0 && r < rows.length && c < cols.length ? [r, c] : null;
  };
  canvas.onmousemove = event => {
    const hit = locate(event);
    if (!hit || (opts.skip && opts.skip(...hit))) { hideTip(); canvas.style.cursor = ""; return; }
    canvas.style.cursor = opts.onClick ? "pointer" : "";
    const [r, c] = hit;
    showTip(event, opts.tip ? opts.tip(r, c, values[r][c]) : `${rows[r]} × ${cols[c]}\n${format(values[r][c])}`);
  };
  canvas.onmouseleave = hideTip;
  if (opts.onClick) canvas.onclick = event => { const hit = locate(event); if (hit) opts.onClick(...hit); };
  return wrap;
}

/**
 * Line chart, one shared y axis.
 * series: [{name, values, color?, dashed?}]; labels: x labels; yLabel; xLabel;
 * domain [lo, hi] (auto if absent); log (log10 y); format(v); tip(i) -> string; marker(i) highlight;
 * zeroLine; height.
 */
// Round tick values covering [lo, hi]: steps of 1, 2 or 5 times a power of ten.
function niceTicks(lo, hi, count = 5) {
  const raw = (hi - lo || Math.abs(hi) || 1) / (count - 1);
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(m => m * power).find(v => v >= raw);
  const start = Math.floor(lo / step + 1e-9) * step, end = Math.ceil(hi / step - 1e-9) * step;
  const ticks = [];
  for (let v = start; v <= end + step / 2; v += step) ticks.push(Math.abs(v) < step / 1e6 ? 0 : v);
  return ticks;
}

export function lineChart(container, { series, labels, xs, yLabel, xLabel = "", domain, log = false, format = v => v.toFixed(2),
                                        tip, height = 230, zeroLine = false, refLine }) {
  const wrap = el("div", "chart");
  container.appendChild(wrap);
  const W = Math.max(320, widthOf(wrap)), H = height, m = { l: 56, r: 16, t: 14, b: 38 };
  const tf = v => log ? Math.log10(Math.max(v, 1e-12)) : v;
  const all = series.flatMap(s => s.values).filter(v => v != null && Number.isFinite(tf(v)));
  let [lo, hi] = domain || [Math.min(...all), Math.max(...all)];
  let linearTicks = null;
  if (!log) {
    linearTicks = niceTicks(lo, hi);
    if (!domain) { lo = linearTicks[0]; hi = linearTicks[linearTicks.length - 1]; }
    else linearTicks = linearTicks.filter(v => v >= lo - 1e-9 && v <= hi + 1e-9);
  }
  const [tlo, thi] = [tf(lo), tf(hi)];
  // xs: numeric x positions (uneven spacing drawn to scale); otherwise points are evenly spaced.
  const [xlo, xhi] = xs ? [Math.min(...xs), Math.max(...xs)] : [0, labels.length - 1];
  const x = i => m.l + (xhi === xlo ? 0.5 : ((xs ? xs[i] : i) - xlo) / (xhi - xlo)) * (W - m.l - m.r);
  const y = v => m.t + (1 - (tf(v) - tlo) / (thi - tlo || 1)) * (H - m.t - m.b);
  const grid = cssVar("--grid"), axis = cssVar("--axis"), muted = cssVar("--muted"), surface = cssVar("--surface");
  const colors = SERIES();
  let svg = `<svg width="${W}" height="${H}" role="img" aria-label="${esc(yLabel)}">`;
  const ticks = log ? Array.from({ length: Math.floor(thi) - Math.ceil(tlo) + 1 }, (_, k) => 10 ** (Math.ceil(tlo) + k))
                    : linearTicks;
  for (const v of ticks) {
    svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="${grid}"/>`;
    svg += `<text x="${m.l - 6}" y="${y(v) + 4}" font-size="11" fill="${muted}" text-anchor="end">${esc(format(v))}</text>`;
  }
  if (zeroLine && lo < 0 && hi > 0) svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(0)}" y2="${y(0)}" stroke="${axis}" stroke-width="1.5"/>`;
  if (refLine != null) svg += `<line x1="${x(refLine)}" x2="${x(refLine)}" y1="${m.t}" y2="${H - m.b}" stroke="${axis}" stroke-dasharray="4 3"/>`;
  svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}" stroke="${axis}"/>`;
  const every = Math.ceil(labels.length / Math.max(2, Math.floor((W - m.l - m.r) / 36)));
  let lastLabelX = -Infinity;
  labels.forEach((label, i) => {
    if (!xs && i % every && i !== labels.length - 1) return;
    if (xs && x(i) - lastLabelX < 34) return;  // uneven spacing: skip labels that would collide
    lastLabelX = x(i);
    svg += `<text x="${x(i)}" y="${H - m.b + 16}" font-size="11" fill="${muted}" text-anchor="middle">${esc(label)}</text>`;
  });
  if (xLabel) svg += `<text x="${W - m.r}" y="${H - 4}" font-size="11" fill="${muted}" text-anchor="end">${esc(xLabel)}</text>`;
  series.forEach((s, k) => {
    const color = s.color || colors[k % colors.length];
    const points = s.values.map((v, i) => v == null ? null : `${x(i)},${y(Math.min(Math.max(v, log ? lo : -Infinity), Infinity))}`).filter(Boolean);
    svg += `<polyline fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round" ${s.dashed ? 'stroke-dasharray="6 4"' : ""} points="${points.join(" ")}"/>`;
    s.values.forEach((v, i) => { if (v != null) svg += `<circle cx="${x(i)}" cy="${y(v)}" r="3.5" fill="${color}" stroke="${surface}" stroke-width="1.5"/>`; });
  });
  labels.forEach((_, i) => {
    const left = i === 0 ? m.l : (x(i - 1) + x(i)) / 2, right = i === labels.length - 1 ? W - m.r : (x(i) + x(i + 1)) / 2;
    svg += `<rect data-i="${i}" x="${left}" y="${m.t}" width="${Math.max(1, right - left)}" height="${H - m.t - m.b}" fill="transparent"/>`;
  });
  svg += `<line class="cross" y1="${m.t}" y2="${H - m.b}" stroke="${axis}" visibility="hidden"/></svg>`;
  const legend = series.length > 1 ? `<div class="legend">${series.map((s, k) =>
    `<span><i style="background:${s.color || colors[k % colors.length]}"></i>${esc(s.name)}</span>`).join("")}</div>` : "";
  wrap.innerHTML = `<div class="chart-title">${esc(yLabel)}</div>${legend}${svg}`;
  const cross = $(".cross", wrap);
  wrap.querySelectorAll("rect[data-i]").forEach(rect => {
    rect.onmousemove = event => {
      const i = Number(rect.dataset.i);
      cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
      showTip(event, tip ? tip(i) : `${labels[i]}\n${series.map(s => `${s.name}: ${s.values[i] == null ? "–" : format(s.values[i])}`).join("\n")}`);
    };
    rect.onmouseleave = () => { cross.setAttribute("visibility", "hidden"); hideTip(); };
  });
  return wrap;
}

// Inline signed bars for a short vector (e.g. the first 8 values of an activation slice).
export function sparkbars(values, width = 120, height = 26) {
  const max = Math.max(...values.map(Math.abs), 1e-12);
  const w = width / values.length, mid = height / 2;
  const pos = cssVar("--delta"), neg = cssVar("--critical");
  return `<svg class="spark" width="${width}" height="${height}">${values.map((v, i) => {
    const h = Math.max(1, Math.abs(v) / max * (mid - 1));
    return `<rect x="${i * w + 1}" y="${v >= 0 ? mid - h : mid}" width="${Math.max(1, w - 2)}" height="${h}" fill="${v >= 0 ? pos : neg}" rx="1"/>`;
  }).join("")}<line x1="0" x2="${width}" y1="${mid}" y2="${mid}" stroke="${cssVar("--axis")}"/></svg>`;
}

// Horizontal bars for a ranked list: [{label, value, note}], signed when any value is negative.
export function barList(items, { format = v => v.toFixed(3), max } = {}) {
  const signed = items.some(i => i.value < 0);
  const top = max || Math.max(...items.map(i => Math.abs(i.value)), 1e-12);
  return `<div class="barlist${signed ? " signed" : ""}">${items.map(i => {
    const w = Math.abs(i.value) / top * (signed ? 50 : 100);
    const style = signed ? (i.value >= 0 ? `left:50%;width:${w}%` : `left:${50 - w}%;width:${w}%`) : `left:0;width:${w}%`;
    return `<div class="barrow"><span class="bl">${i.label}</span><span class="bt"><span class="bf${i.value < 0 ? " neg" : ""}" style="${style}"></span></span><span class="bv">${format(i.value)}${i.note ? ` <span class="muted">${i.note}</span>` : ""}</span></div>`;
  }).join("")}</div>`;
}
