"""Token-level attention table (toy: 4 image cells, 3 text tokens, 3 candidates) for ours (region-restricted) and the
free-attention ablation -> attention_table.svg / .pdf.  usage: python paper/figures/make_attention_table.py"""
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOK = [("A", "img"), ("B", "img"), ("C", "img"), ("D", "img"), ("handle", "txt"), ("mug", "txt"), ("<SEL>", "sel"),
       ("M_handle", "msk"), ("M_mug", "msk"), ("M_table", "msk")]
REGION = {7: {1}, 8: {0, 1, 2}, 9: {3}}
NOTE = {7: "only cell B", 8: "cells A, B, C", 9: "only cell D"}
FILL = {"img": "#f0a27a", "txt": "#8fd18b", "sel": "#e58ac0", "msk": "#4aa8dc"}
BLOCK, INK, MUTED = "#eef0f4", "#1d2433", "#5b6475"


def allowed(r, c, free):
    kr, kc = TOK[r][1], TOK[c][1]
    if kr != "msk":
        return kc != "msk" and c <= r                      # causal reading, never a candidate
    if kc == "img":
        return free or c in REGION[r]
    return True


def panel(x0, y0, free, title):
    s, cell = [], 34
    s.append(f'<text x="{x0}" y="{y0-58}" font-size="17" font-weight="700" fill="{INK}">{title}</text>')
    for c, (name, k) in enumerate(TOK):
        cx = x0 + c * cell + cell / 2
        s.append(f'<text transform="translate({cx+4},{y0-8}) rotate(-50)" font-size="12" fill="{INK}" '
                 f'font-family="IBM Plex Mono, monospace">{name.replace("<", "&lt;").replace(">", "&gt;")}</text>')
    for r, (name, k) in enumerate(TOK):
        y = y0 + r * cell
        s.append(f'<text x="{x0-10}" y="{y+cell/2+4}" font-size="12" text-anchor="end" fill="{INK}" '
                 f'font-family="IBM Plex Mono, monospace">{r}  {name.replace("<", "&lt;").replace(">", "&gt;")}</text>')
        for c, (_, kc) in enumerate(TOK):
            x = x0 + c * cell
            ok = allowed(r, c, free)
            bias = ok and k == "msk" and kc == "msk" and r != c
            s.append(f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" rx="4" '
                     f'fill="{FILL[kc] if ok else BLOCK}" opacity="{1 if ok else 1}"/>')
            if bias:
                s.append(f'<text x="{x+cell/2-1}" y="{y+cell/2+5}" font-size="14" font-weight="700" fill="#fff" '
                         f'text-anchor="middle" font-family="Georgia, serif" font-style="italic">b</text>')
            if free and k == "msk" and kc == "img" and c not in REGION[r]:
                s.append(f'<rect x="{x+1}" y="{y+1}" width="{cell-4}" height="{cell-4}" rx="4" fill="none" '
                         f'stroke="#c0392b" stroke-width="2.5"/>')
        if not free and r in NOTE:
            s.append(f'<text x="{x0+10*cell+12}" y="{y+cell/2+4}" font-size="12.5" fill="{MUTED}">← {NOTE[r]}</text>')
    for k in (4, 7):
        s.append(f'<line x1="{x0+k*cell-1}" y1="{y0-4}" x2="{x0+k*cell-1}" y2="{y0+10*cell+2}" stroke="{INK}" stroke-width="1.6"/>')
        s.append(f'<line x1="{x0-4}" y1="{y0+k*cell-1}" x2="{x0+10*cell+2}" y2="{y0+k*cell-1}" stroke="{INK}" stroke-width="1.6"/>')
    return "\n".join(s)


W, H = 1180, 560
parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="IBM Plex Sans, Inter, Helvetica, sans-serif"><rect width="{W}" height="{H}" fill="#fff"/>',
         panel(120, 130, False, "Ours: region-restricted attention"),
         panel(720, 130, True, "Free attention (ablation)")]
ly = 500
legend = [("img", "image cell"), ("txt", "question word"), ("sel", "answer slot <SEL>"), ("msk", "candidate (mask token)")]
x = 40
for k, lab in legend:
    parts.append(f'<rect x="{x}" y="{ly-12}" width="16" height="16" rx="3" fill="{FILL[k]}"/>'
                 f'<text x="{x+24}" y="{ly+1}" font-size="13" fill="{INK}">{lab.replace("<", "&lt;").replace(">", "&gt;")}</text>')
    x += 190
parts.append(f'<rect x="{x}" y="{ly-12}" width="16" height="16" rx="3" fill="{BLOCK}"/>'
             f'<text x="{x+24}" y="{ly+1}" font-size="13" fill="{INK}">blocked (score −∞)</text>')
parts.append(f'<text x="40" y="{ly+30}" font-size="13" fill="{MUTED}">Row = token that attends, column = token it '
             f'attends to; coloured = allowed. <tspan font-style="italic" font-family="Georgia, serif">b</tspan>: learned '
             f'bias by tree relation (parent, child, sibling, …).</text>')
parts.append(f'<text x="40" y="{ly+50}" font-size="13" fill="{MUTED}">Red outline: the only cells that differ in the '
             f'free-attention ablation (a candidate also sees image cells outside its mask).</text>')
parts.append(f'<text x="940" y="{130+3*34+18}" font-size="12.5" fill="{MUTED}" text-anchor="middle"></text></svg>')
svg = "\n".join(parts)
(HERE / "attention_table.svg").write_text(svg)
js = (f"const {{chromium}}=require('playwright');const fs=require('fs');(async()=>{{const svg=fs.readFileSync("
      f"'{HERE/'attention_table.svg'}','utf8');const b=await chromium.launch();const p=await b.newPage({{viewport:"
      f"{{width:{W},height:{H}}},deviceScaleFactor:2}});await p.setContent(`<!doctype html><html><head><style>@page{{"
      f"size:{W}px {H}px;margin:0}}html,body{{margin:0}}</style></head><body>${{svg}}</body></html>`);await "
      f"p.screenshot({{path:'{HERE/'attention_table.png'}'}});await p.pdf({{path:'{HERE/'attention_table.pdf'}',"
      f"width:'{W}px',height:'{H}px',printBackground:true,pageRanges:'1'}});await b.close();}})();")
subprocess.run(["node", "-e", js], check=True)
print(HERE / "attention_table.pdf")
