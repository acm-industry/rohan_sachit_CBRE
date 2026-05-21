#!/usr/bin/env bash
# build_docs.sh — render docs/design_doc.md → docs/design_doc.pdf
#
# Toolchain (macOS, install once):
#   brew install pandoc weasyprint
#   npm install -g @mermaid-js/mermaid-cli   # for `mmdc`
#
# Toolchain (Debian/Ubuntu):
#   sudo apt-get install -y pandoc
#   pipx install weasyprint   # or: pip install --user weasyprint
#   npm install -g @mermaid-js/mermaid-cli
#
# Why mmdc and not rsvg-convert? The committed diagrams/*.svg files use
# <foreignObject>+HTML for node labels (mermaid output), which rsvg-convert
# strips silently — the result is text-less boxes. mmdc renders the mermaid
# source directly via headless chromium, so the labels survive.
#
# Usage (from repo root):
#   bash scripts/build_docs.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC_DIR="$REPO_ROOT/docs"
DIAG_DIR="$DOC_DIR/diagrams"
SRC="$DOC_DIR/design_doc.md"
OUT="$DOC_DIR/design_doc.pdf"
CSS="$DOC_DIR/print.css"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

for cmd in pandoc weasyprint python3 mmdc; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "missing: $cmd" >&2; exit 1; }
done

# 1) Re-render mermaid source → PNG (high DPI for print). Uses headless
#    chromium under the hood, so HTML text labels (`<foreignObject>`) render.
for name in agent_graph hitl_resume scale_10k; do
  mmdc -i "$DIAG_DIR/$name.mmd" -o "$DIAG_DIR/$name.png" \
       -t default -b white --width 1800 --scale 2 >/dev/null
done

# 2) Preprocess the markdown: replace each ```mermaid ... ``` fenced block
#    with an inline image reference to its rasterized PNG. Order in the doc:
#    agent_graph (§2), hitl_resume (§4.1), scale_10k (§10).
python3 - "$SRC" "$TMP/design_doc.md" <<'PY'
import re, sys, pathlib
src, dst = sys.argv[1], sys.argv[2]
text = pathlib.Path(src).read_text()
diagrams = iter(["agent_graph", "hitl_resume", "scale_10k"])
def sub(_m):
    name = next(diagrams)
    # pandoc markdown image with a .diagram CSS class
    return "\n![](diagrams/" + name + ".png){.diagram}\n"
out = re.sub(r"```mermaid\n.*?\n```", sub, text, flags=re.DOTALL)
pathlib.Path(dst).write_text(out)
PY

# 3) Render via pandoc → HTML → weasyprint → PDF. Use --embed-resources so the
#    intermediate HTML is self-contained (images get base64-embedded).
pandoc "$TMP/design_doc.md" \
  --resource-path="$DOC_DIR" \
  --from=markdown+pipe_tables+yaml_metadata_block+gfm_auto_identifiers \
  --to=html5 \
  --standalone \
  --embed-resources \
  --metadata pagetitle="Design Document — CBRE HITL Call-Intake Agent" \
  --css="$CSS" \
  --output="$TMP/design_doc.html"

weasyprint "$TMP/design_doc.html" "$OUT"

echo "wrote: $OUT ($(du -h "$OUT" | cut -f1), $(python3 -c '
import sys
data = open(sys.argv[1], "rb").read()
print(data.count(b"/Type /Page\n") + data.count(b"/Type/Page\n") + data.count(b"/Type /Page "), "pages")
' "$OUT" 2>/dev/null || echo "?"))"
