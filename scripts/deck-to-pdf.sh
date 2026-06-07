#!/usr/bin/env bash
# Render the slide deck to a static PDF (for written submission; GIFs become a
# single frame each — for the animated version present the .standalone.html).
#
# Why not a one-liner: decktape's `generic` plugin can't detect this custom deck's
# last slide (the .rise animations re-fire on every key press, so it never sees
# "no change" and loops forever). So we drive a headless browser deterministically:
# load the offline standalone, press ArrowRight exactly N-1 times, screenshot each
# slide, then stitch the PNGs into a PDF.
#
# One-time setup (downloads a headless Chromium into ~/.cache/puppeteer):
#   npx -y decktape@latest --version    # any npx puppeteer install works
# Then:
#   bash scripts/deck-to-pdf.sh [num_slides]   # default 15
set -euo pipefail
cd "$(dirname "$0")/.."
N="${1:-15}"
SRC="docs/final-presentation.standalone.html"
[ -f "$SRC" ] || python scripts/inline-presentation.py   # ensure the standalone exists

NODEMODS=$(dirname "$(find "$HOME/.npm/_npx" -maxdepth 3 -type d -name puppeteer 2>/dev/null | head -1)")
[ -d "$NODEMODS/puppeteer" ] || { echo "puppeteer not found — run: npx -y decktape@latest --version"; exit 1; }

cat > /tmp/_deck2png.js <<'JS'
const puppeteer = require('puppeteer');
(async () => {
  const url = 'file://' + process.argv[2], n = parseInt(process.argv[3] || '15', 10);
  const b = await puppeteer.launch({headless: true, args: ['--no-sandbox','--disable-gpu']});
  const p = await b.newPage();
  await p.setViewport({width: 1920, height: 1080, deviceScaleFactor: 1.5});
  await p.goto(url, {waitUntil: 'networkidle0', timeout: 60000});
  await new Promise(r => setTimeout(r, 1600));
  for (let i = 0; i < n; i++) {
    if (i > 0) await p.keyboard.press('ArrowRight');
    await new Promise(r => setTimeout(r, 1100));
    await p.screenshot({path: `/tmp/_slide_${String(i).padStart(2,'0')}.png`});
  }
  await b.close();
})().catch(e => { console.error(e); process.exit(1); });
JS

rm -f /tmp/_slide_*.png
NODE_PATH="$NODEMODS" node /tmp/_deck2png.js "$PWD/$SRC" "$N"

python - "$N" <<'PY'
from PIL import Image; import glob
fs = sorted(glob.glob("/tmp/_slide_*.png"))
ims = [Image.open(f).convert("RGB") for f in fs]
ims[0].save("docs/final-presentation.pdf", save_all=True, append_images=ims[1:], resolution=150.0)
print(f"wrote docs/final-presentation.pdf ({len(ims)} pages)")
PY
