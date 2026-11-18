#!/usr/bin/env bash
# Compile audited, frozen tables and figures without rerunning experiments.
set -euo pipefail
paper_root="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "$paper_root/.." && pwd)"
python_bin="${PIVOT_PYTHON:-python3}"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1787227200}"
cd "$paper_root"
mkdir -p build
TEXINPUTS="$paper_root/style:${TEXINPUTS:-}" \
BSTINPUTS="$paper_root/style:${BSTINPUTS:-}" \
BIBINPUTS="$paper_root:${BIBINPUTS:-}" \
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
cp build/main.pdf pivot_iclr2027_submission.pdf
"$python_bin" "$project_root/scripts/verify_paper.py" \
  --pdf build/main.pdf --source main.tex --output verification.json \
  --preview preview.png --max-main-pages 10
