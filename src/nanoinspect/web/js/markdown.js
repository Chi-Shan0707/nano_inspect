// Minimal markdown for the repository docs: headings, paragraphs, lists, code fences, inline code, bold, links.
import { esc } from "./core.js";

export function markdown(text) {
  const inline = s => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, t, u) => `<a href="${/^https?:/.test(u) ? u : "/docs/" + u}" target="_blank" rel="noopener">${t}</a>`);
  let html = "", para = [], list = [];
  const flush = () => {
    if (para.length) html += `<p>${inline(para.join(" "))}</p>`;
    if (list.length) html += `<ul>${list.map(i => `<li>${inline(i)}</li>`).join("")}</ul>`;
    para = []; list = [];
  };
  text.split(/^```.*$/m).forEach((block, i) => {
    if (i % 2) { flush(); html += `<pre>${esc(block.replace(/^\n|\n$/g, ""))}</pre>`; return; }
    for (const line of block.split("\n")) {
      const heading = line.match(/^(#{1,4})\s+(.*)/), item = line.match(/^\s*[-*]\s+(.*)/);
      if (heading) { flush(); const n = Math.min(heading[1].length + 2, 5); html += `<h${n}>${inline(heading[2])}</h${n}>`; }
      else if (item) { if (para.length) flush(); list.push(item[1]); }
      else if (!line.trim()) flush();
      else if (list.length) list[list.length - 1] += " " + line.trim();
      else para.push(line.trim());
    }
    flush();
  });
  return html;
}
