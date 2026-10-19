#!/usr/bin/env node
/*
 * Render an exam cheat-sheet note to a dense, print-ready A4 PDF.
 *
 *   node docs/superpowers/tools/cheat_sheet_pdf.cjs <note.md> <out.pdf> [out.html]
 *
 * Needs the npm packages marked@12 and katex@0.16 (any folder, put on NODE_PATH) and
 * playwright with its Chromium. Example:
 *
 *   npm install --prefix /tmp/cs marked@12.0.2 katex@0.16.11
 *   NODE_PATH=/tmp/cs/node_modules:$NODE_PATH node docs/superpowers/tools/cheat_sheet_pdf.cjs \
 *     "Courses/MNLP/Exam Prep/MNLP - Cheat Sheet.md" Courses/MNLP/Assets/MNLP-Cheat-Sheet.pdf
 *
 * What it does to the note:
 * - drops the frontmatter, the H1 and the intro between the H1 and the first "## " heading;
 * - uses the text of an HTML comment "<!-- print-header: ... -->" as the one-line page header;
 * - turns [[Target|alias]] and [[Target]] into plain text;
 * - renders $...$ and $$...$$ with KaTeX.
 * Layout knobs (font size, columns) are environment variables: CS_FONT_PT, CS_COLS, CS_MARGIN_MM.
 * The page count is printed at the end; the script does not shrink content to fit.
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
const FONT_PT = parseFloat(process.env.CS_FONT_PT || "6.3");
const COLS = parseInt(process.env.CS_COLS || "3", 10);
const MARGIN_MM = parseFloat(process.env.CS_MARGIN_MM || "4");

let src = fs.readFileSync(mdPath, "utf8");
src = src.replace(/^---\n[\s\S]*?\n---\n/, "");
const headerMatch = src.match(/<!--\s*print-header:\s*([\s\S]*?)-->/);
const header = headerMatch ? headerMatch[1].trim() : "";
const firstH2 = src.search(/^## /m);
if (firstH2 > 0) src = src.slice(firstH2);
src = src.replace(/<!--[\s\S]*?-->/g, "");
src = src.replace(/\[\[([^\]|\\]+?)(?:\\?\|([^\]]+))?\]\]/g, (_, target, alias) => (alias || target).trim());

// Protect math from the markdown parser, render it with KaTeX afterwards.
const math = [];
const stash = (tex, display) => {
  math.push(katex.renderToString(tex, { displayMode: display, throwOnError: true, output: "html" }));
  return `MATHPH${math.length - 1}X`;
};
src = src.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => stash(tex.trim(), true));
src = src.replace(/\$([^$\n]+?)\$/g, (_, tex) => stash(tex, false));
// A single "~" (as in "~25:1") must not pair up into GFM strikethrough.
src = src.replace(/~/g, "\\~");

let body = marked.parse(src, { gfm: true });
body = body.replace(/MATHPH(\d+)X/g, (_, i) => math[+i]);

const katexDir = path.dirname(require.resolve("katex/dist/katex.min.css"));
const katexCss = fs
  .readFileSync(path.join(katexDir, "katex.min.css"), "utf8")
  .replace(/url\(fonts\//g, `url(file://${katexDir}/fonts/`);

const css = `
@page { size: A4; margin: ${MARGIN_MM}mm; }
:root { color-scheme: light; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: #fff; color: #000; }
body {
  font-family: Carlito, "Liberation Sans", Arial, "WenQuanYi Zen Hei", sans-serif;
  font-size: ${FONT_PT}pt; line-height: 1.13;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
.hdr { font-size: ${FONT_PT}pt; border-bottom: 0.5pt solid #000; margin-bottom: 1pt; padding-bottom: 0.5pt; }
.cols { column-count: ${COLS}; column-gap: 2.6mm; column-rule: 0.3pt solid #aaa; column-fill: auto; }
h2 { font-size: ${FONT_PT + 1.2}pt; margin: 2pt 0 1pt; padding: 0.3pt 2pt; background: #1f2a44; color: #fff;
     break-after: avoid; break-inside: avoid; }
h2:first-child { margin-top: 0; }
p { margin: 0 0 1pt; }
ul { margin: 0 0 0.8pt; padding-left: 5.5pt; list-style: none; }
li { margin: 0 0 0.7pt; position: relative; }
li::before { content: "\\2022"; position: absolute; left: -4.6pt; font-size: 0.9em; }
strong { font-weight: 700; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 0.86em; background: #eee; padding: 0 0.6pt; }
table { border-collapse: collapse; width: 100%; margin: 1pt 0 1.5pt; font-size: ${FONT_PT - 0.3}pt;
        line-height: 1.12; break-inside: avoid; }
th, td { border: 0.3pt solid #999; padding: 0.1pt 1.4pt; text-align: left; vertical-align: top; }
th { background: #e4e7ee; font-weight: 700; }
tr:nth-child(even) td { background: #f6f6f6; }
.katex { font-size: 1.02em; }
.katex-display { margin: 0.6pt 0; }
.katex-display > .katex { font-size: 1.0em; }
`;

const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>MNLP Cheat Sheet</title>
<style>${katexCss}</style><style>${css}</style></head>
<body><div class="hdr">${header}</div><div class="cols">${body}</div></body></html>`;

const htmlPath = htmlArg || pdfPath.replace(/\.pdf$/, ".html");
fs.writeFileSync(htmlPath, html);

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto("file://" + path.resolve(htmlPath), { waitUntil: "load" });
  await page.evaluate(() => document.fonts.ready);
  await page.pdf({ path: pdfPath, format: "A4", printBackground: true, preferCSSPageSize: true });
  await browser.close();
  const pages = (fs.readFileSync(pdfPath, "latin1").match(/\/Type\s*\/Page[^s]/g) || []).length;
  console.log(`wrote ${pdfPath} (${pages} pages), html ${htmlPath}`);
})();
