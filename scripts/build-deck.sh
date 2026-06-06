#!/usr/bin/env bash
# Regenerate the progress-report deck (pptx) and export a PDF.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[deck] generating pptx…"
node scripts/deck/generate.js

PPTX="$(pwd)/docs/20260525_progress_report_v1.pptx"
echo "[deck] exporting pdf via LibreOffice…"
if command -v soffice >/dev/null 2>&1; then
  HOME="${HOME:-/root}" soffice --headless --norestore \
    -env:UserInstallation=file:///tmp/lo-deck \
    --convert-to pdf --outdir docs "$PPTX" >/dev/null
  echo "[deck] wrote docs/20260525_progress_report_v1.pdf"
else
  echo "[deck] WARN: soffice not found — run scripts/setup.sh first."
fi
