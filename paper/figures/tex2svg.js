// usage: node tex2svg.js in.json out.json   (in: {key: tex}, out: {key: {svg, w_ex, h_ex, va_ex}})
const fs = require('fs');
const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
const {AllPackages} = require('mathjax-full/js/input/tex/AllPackages.js');
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const doc = mathjax.document('', {InputJax: new TeX({packages: AllPackages}), OutputJax: new SVG({fontCache: 'none'})});
const inp = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = {};
for (const [k, tex] of Object.entries(inp)) {
  const node = doc.convert(tex, {display: false});
  const svg = adaptor.firstChild(node);
  const w = parseFloat(adaptor.getAttribute(svg, 'width')), h = parseFloat(adaptor.getAttribute(svg, 'height'));
  const va = parseFloat((adaptor.getAttribute(svg, 'style') || '').replace(/.*vertical-align:\s*([-\d.]+)ex.*/, '$1')) || 0;
  out[k] = {svg: adaptor.outerHTML(svg), w_ex: w, h_ex: h, va_ex: va};
}
fs.writeFileSync(process.argv[3], JSON.stringify(out));
