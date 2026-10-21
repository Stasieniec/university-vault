#!/usr/bin/env node
/*
 * Render an exam cheat-sheet note to a dense, print-ready two-sided A4 PDF.
 *
 *   node docs/superpowers/tools/cheat_sheet_pdf.cjs <note.md> <out.pdf> [out.html]
 *
 * Needs the npm packages marked@12 and katex@0.16 (any folder, put on NODE_PATH) and
 * playwright with its Chromium. Example:
 *
 *   npm install --prefix /tmp/cs marked@12.0.2 katex@0.16.11
 *   NODE_PATH=/tmp/cs/node_modules:$(npm root -g) node docs/superpowers/tools/cheat_sheet_pdf.cjs \
 *     "Courses/MNLP/Exam Prep/MNLP - Cheat Sheet.md" Courses/MNLP/Assets/MNLP-Cheat-Sheet.pdf
 *
 * What it does to the note:
 * - drops the frontmatter, the H1 and the intro between the H1 and the first "## " heading;
 * - uses the text of an HTML comment "<!-- print-header: ... -->" as the title in the index strip;
 * - "<!-- side 2 -->" starts the second side, "<!-- colbreak -->" forces a column break;
 * - "## L05 ..." headings become coloured lecture bars, "### ..." topic headers;
 * - a paragraph starting "!! " becomes a highlighted traps line;
 * - turns [[Target|alias]] and [[Target]] into plain text and renders $...$ / $$...$$ with KaTeX;
 * - fills an index strip at the top of each side with every topic and the column it landed in.
 * Layout knobs are environment variables: CS_FONT_PT (5.5), CS_COLS (4), CS_MARGIN_MM (3.5),
 * CS_FONT (Inter). It reports overflow and how full each side's last column is; it does not
 * shrink content to fit.
 */
const fs = require("fs");
const path = require("path");
const { marked } = require("marked");
const katex = require("katex");
const { chromium } = require("playwright");

const [mdPath, pdfPath, htmlArg] = process.argv.slice(2);
if (!mdPath || !pdfPath) {
  console.error("usage: cheat_sheet_pdf.cjs <note.md> <out.pdf> [out.html]");
  process.exit(2);
}
const FONT_PT = parseFloat(process.env.CS_FONT_PT || "5.5");
const COLS = parseInt(process.env.CS_COLS || "4", 10);
const MARGIN_MM = parseFloat(process.env.CS_MARGIN_MM || "3");
const FONT = process.env.CS_FONT || "Inter";

let src = fs.readFileSync(mdPath, "utf8");
src = src.replace(/^---\n[\s\S]*?\n---\n/, "");
const headerMatch = src.match(/<!--\s*print-header:\s*([\s\S]*?)-->/);
const header = headerMatch ? headerMatch[1].trim() : "";
const firstH2 = src.search(/^## /m);
if (firstH2 > 0) src = src.slice(firstH2);
src = src.replace(/<!--\s*colbreak\s*-->/g, '\n\n<div class="colbreak"></div>\n\n');
const sides = src.split(/<!--\s*side 2\s*-->/);
const LECTURE_COLOURS = ["#4a4a4a", "#1f4e79", "#6b3fa0", "#a34d00", "#1e6b45", "#9c2242", "#1f2a44", "#5a5a00"];

function render(md) {
  md = md.replace(/<!--(?!\s*colbreak)[\s\S]*?-->/g, "");
  md = md.replace(/\[\[([^\]|\\]+?)(?:\\?\|([^\]]+))?\]\]/g, (_, target, alias) => (alias || target).trim());
  // Protect math from the markdown parser, render it with KaTeX afterwards.
  const math = [];
  const stash = (tex, display) => {
    math.push(katex.renderToString(tex, { displayMode: display, throwOnError: true, output: "html" }));
    return `MATHPH${math.length - 1}X`;
  };
  md = md.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => stash(tex.trim(), true));
  md = md.replace(/\$([^$\n]+?)\$/g, (_, tex) => stash(tex, false));
  // A single "~" (as in "~25:1") must not pair up into GFM strikethrough.
  md = md.replace(/~/g, "\\~");
  let html = marked.parse(md, { gfm: true });
  html = html.replace(/MATHPH(\d+)X/g, (_, i) => math[+i]);
  html = html.replace(/<p>!!\s*/g, '<p class="trap">');
  html = html.replace(/<h2>L0?(\d+)/g, (_, n) => `<h2 data-l="${n}" style="--lc:${LECTURE_COLOURS[n] || "#333"}">L${n.padStart(2, "0")}`);
  // Topic headers inherit the colour of the lecture they sit under.
  let current = "#333";
  html = html.replace(/<h2 data-l="(\d+)"|<h3>/g, (m, n) => {
    if (n) { current = LECTURE_COLOURS[n] || "#333"; return m; }
    return `<h3 style="--lc:${current}">`;
  });
  return html;
}

const katexDir = path.dirname(require.resolve("katex/dist/katex.min.css"));
const katexCss = fs
  .readFileSync(path.join(katexDir, "katex.min.css"), "utf8")
  .replace(/url\(fonts\//g, `url(file://${katexDir}/fonts/`);

const css = `
@page { size: 210mm 297mm; margin: 0; }
:root { color-scheme: light; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: #fff; color: #000; }
body {
  font-family: "${FONT}", "Liberation Sans", "WenQuanYi Zen Hei", sans-serif;
  font-size: ${FONT_PT}pt; line-height: 1.1; font-feature-settings: "cv11", "ss03";
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
.page { width: 210mm; height: 297mm; padding: ${MARGIN_MM}mm; display: flex; flex-direction: column;
        overflow: hidden; break-after: page; }
.page:last-child { break-after: auto; }
.idx { flex: none; font-size: ${FONT_PT}pt; line-height: 1.2; border-bottom: 0.6pt solid #000;
       padding-bottom: 0.6pt; margin-bottom: 1pt; }
.idx b.t { font-weight: 800; }
.idx .lec { font-weight: 700; color: #fff; padding: 0 1.4pt; border-radius: 1pt; }
.idx .c { font-weight: 700; }
.cols { flex: 1; min-height: 0; column-count: ${COLS}; column-gap: 1.8mm; column-rule: 0.3pt solid #999;
        column-fill: auto; }
.colbreak { break-after: column; height: 0; }
h2 { font-size: ${FONT_PT + 1.2}pt; margin: 1.2pt 0 0.5pt; padding: 0.4pt 2pt; background: var(--lc); color: #fff;
     break-after: avoid; break-inside: avoid; }
.cols > h2:first-child { margin-top: 0; }
h3 { font-size: ${FONT_PT + 0.2}pt; margin: 1.2pt 0 0.3pt; padding: 0.1pt 1.6pt; font-weight: 800;
     border-left: 2pt solid var(--lc); background: #ececec; break-after: avoid; break-inside: avoid; }
h2 + h3 { margin-top: 0.6pt; }
p { margin: 0 0 0.5pt; }
p.trap { border-left: 1.4pt solid #a00; background: #fbecec; padding: 0.2pt 1.2pt; margin: 0.4pt 0 0.6pt; }
ul { margin: 0 0 0.4pt; padding-left: 4.4pt; list-style: none; }
li { margin: 0 0 0.3pt; position: relative; }
li::before { content: "\\2022"; position: absolute; left: -3.8pt; font-size: 0.9em; }
li > ul { margin-top: 0.3pt; }
strong { font-weight: 700; }
code { font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace; font-size: 0.9em; background: #eee; padding: 0 0.5pt; }
table { border-collapse: collapse; width: 100%; margin: 0.5pt 0 0.8pt; line-height: 1.08; break-inside: avoid; }
th, td { border: 0.3pt solid #999; padding: 0 1pt; text-align: left; vertical-align: top; }
th { background: #e2e5ec; font-weight: 700; }
tr:nth-child(even) td { background: #f5f5f5; }
.katex { font-size: 1.0em; }
.katex-display { margin: 0.4pt 0; }
`;

const pagesHtml = sides
  .map((md, i) => `<section class="page"><div class="idx" data-side="${i + 1}"></div><div class="cols">${render(md)}</div></section>`)
  .join("\n");
const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Cheat Sheet</title>
<style>${katexCss}</style><style>${css}</style></head>
<body>${pagesHtml}</body></html>`;

const htmlPath = htmlArg || pdfPath.replace(/\.pdf$/, ".html");

// Runs in the page: fill each side's index strip from where its headings landed, and measure fit.
function layoutPass({ header, cols }) {
  const out = [];
  let prev = null;
  document.querySelectorAll(".page").forEach((page, si) => {
    const box = page.querySelector(".cols");
    const r = box.getBoundingClientRect();
    const gap = parseFloat(getComputedStyle(box).columnGap);
    const colW = (r.width - gap * (cols - 1)) / cols;
    const colOf = (el) => Math.floor((el.getBoundingClientRect().left - r.left + 1) / (colW + gap)) + 1;
    const parts = [];
    let lecture = null;
    box.querySelectorAll("h2, h3").forEach((h) => {
      if (h.tagName === "H2") {
        const label = h.textContent.match(/^L\d+/)[0];
        lecture = { label, colour: getComputedStyle(h).backgroundColor, items: [] };
        parts.push(lecture);
        prev = { label, colour: lecture.colour };
        if (!h.nextElementSibling || h.nextElementSibling.tagName !== "H3") lecture.items.push([h.textContent.replace(/^L\d+\s*/, ""), colOf(h)]);
      } else {
        if (!lecture) { lecture = { label: prev ? prev.label + " cont." : "", colour: prev ? prev.colour : "#333", items: [] }; parts.push(lecture); }
        lecture.items.push([h.textContent, colOf(h)]);
      }
    });
    const idx = page.querySelector(".idx");
    const title = si === 0 && header ? `<b class="t">${header}</b> · ` : `<b class="t">Side ${si + 1}</b> · `;
    idx.innerHTML = title + parts.map((p) =>
      `<span class="lec" style="background:${p.colour}">${p.label}</span> ` +
      p.items.map(([t, c]) => `${t} <span class="c">${c}</span>`).join(" · ")
    ).join(" ");
    // Overflow: anything that landed right of the last column.
    let maxCol = 1, lastBottom = 0;
    // Use every line box (a paragraph or table split across columns has several fragments).
    const at = (x) => Math.floor((x - r.left + 1) / (colW + gap)) + 1;
    for (const el of box.querySelectorAll("p, li, tr, h2, h3, .katex-display")) {
      for (const b of el.getClientRects()) {
        if (b.width === 0) continue;
        const c = at(b.left);
        if (c > maxCol) { maxCol = c; lastBottom = 0; }
        if (c === maxCol) lastBottom = Math.max(lastBottom, b.bottom);
      }
    }
    const used = maxCol - 1 + (lastBottom - r.top) / r.height;
    out.push({ side: si + 1, used, overflowCols: Math.max(0, maxCol - cols), lastColFill: used - (maxCol - 1) });
  });
  return out;
}

(async () => {
  fs.writeFileSync(htmlPath, html);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 794, height: 1123 } });
  await page.emulateMedia({ media: "print" });
  await page.goto("file://" + path.resolve(htmlPath), { waitUntil: "load" });
  await page.evaluate(() => document.fonts.ready);
  let fit;
  for (let i = 0; i < 3; i++) fit = await page.evaluate(layoutPass, { header, cols: COLS });
  fs.writeFileSync(htmlPath, await page.content());
  await page.pdf({ path: pdfPath, width: "210mm", height: "297mm", printBackground: true, preferCSSPageSize: true });
  await browser.close();
  const pages = (fs.readFileSync(pdfPath, "latin1").match(/\/Type\s*\/Page[^s]/g) || []).length;
  console.log(`wrote ${pdfPath} (${pages} pages), html ${htmlPath}`);
  for (const f of fit)
    console.log(`side ${f.side}: ${f.overflowCols ? "OVERFLOW" : "fits"}, content fills ${f.used.toFixed(2)} of ${COLS} columns`);
})();
