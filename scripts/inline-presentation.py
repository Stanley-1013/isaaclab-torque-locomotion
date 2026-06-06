#!/usr/bin/env python3
"""
Build a self-contained, offline-openable copy of the final-report deck.

Reads docs/final-presentation.html, base64-inlines every local image/GIF it
references (each unique asset stored ONCE, referenced by a tiny JS map so a GIF
used on two slides isn't embedded twice), and writes
docs/final-presentation.standalone.html — a single file you can double-click,
email, or upload. No server, no network, no sibling files needed.

Usage:  python scripts/inline-presentation.py
"""
import base64, mimetypes, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "final-presentation.html"
OUT = ROOT / "docs" / "final-presentation.standalone.html"

html = SRC.read_text(encoding="utf-8")

# Collect unique local assets referenced by src="...".
local = []
for s in re.findall(r'src="([^"]+)"', html):
    if not s.startswith(("http://", "https://", "data:")) and s not in local:
        local.append(s)

# Build a JS asset map: key -> data URI. Each file encoded once.
entries, key_of = [], {}
for i, rel in enumerate(local):
    p = (SRC.parent / rel).resolve()
    if not p.is_file():
        raise SystemExit(f"missing asset: {rel} -> {p}")
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    key = f"a{i}"
    key_of[rel] = key
    entries.append(f'  "{key}":"data:{mime};base64,{b64}"')

# Rewrite each <img ... src="REL" ...> to carry data-k="aN" and a blank src;
# a tiny boot script swaps in the data URI on load. Keeps the file minimal
# even when an asset appears on multiple slides.
def swap(m):
    tag = m.group(0)
    rel = m.group(1)
    key = key_of.get(rel)
    if key is None:
        return tag
    tag = tag.replace(f'src="{rel}"', f'data-k="{key}"')
    return tag

html = re.sub(r'<img\b[^>]*\bsrc="([^"]+)"[^>]*>', swap, html)

boot = (
    "\n<script>\n"
    "const ASSETS={\n" + ",\n".join(entries) + "\n};\n"
    "document.querySelectorAll('img[data-k]').forEach(el=>{"
    "const u=ASSETS[el.dataset.k];if(u)el.src=u;});\n"
    "</script>\n"
)
html = html.replace("</body>", boot + "</body>")

OUT.write_text(html, encoding="utf-8")
mb = OUT.stat().st_size / 1e6
print(f"wrote {OUT.relative_to(ROOT)}  ({mb:.1f} MB, {len(local)} assets inlined)")
