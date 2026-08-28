#!/usr/bin/env bash
set -euo pipefail

paper_root="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "$paper_root/../.." && pwd)"
cd "$paper_root"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"

# Keep the submission PDF byte-stable across clean rebuilds.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1787227200}"
python_bin="$project_root/.venv/bin/python"
if [ ! -x "$python_bin" ]; then
  python_bin="python3"
fi

"$python_bin" "$project_root/scripts/build_paper_tables.py" \
  --snapshot ../snapshot \
  --output ../tables
"$python_bin" "$project_root/scripts/build_v10_tables.py" \
  --root "$project_root" \
  --output paper/tables
"$python_bin" "$project_root/scripts/build_paper_snippets.py" --root "$project_root"
"$python_bin" "$project_root/scripts/build_opentikz_architecture.py" \
  --source "$paper_root/figures/fig3_pivot_architecture.tex" \
  --output-pdf "$paper_root/figures/fig3_pivot_architecture.pdf" \
  --output-svg "$paper_root/figures/fig3_pivot_architecture.svg" \
  --opentikz-root "$project_root/.tools/opentikz"
"$python_bin" -m experiments.v10.figures --root "$project_root"
"$python_bin" "$project_root/scripts/build_release_assets.py" --root "$project_root"
mkdir -p build
cp style/iclr2027_conference.bst iclr2027_conference.bst
TEXINPUTS="$paper_root/style:${TEXINPUTS:-}" \
BSTINPUTS="$paper_root/style:${BSTINPUTS:-}" \
BIBINPUTS="$paper_root:${BIBINPUTS:-}" \
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
cp build/main.pdf pivot_iclr2027_submission.pdf
"$python_bin" "$project_root/scripts/build_iclr_supplement.py" \
  --project-root "$project_root" \
  --output-root "$paper_root/supplementary" \
  --archive "$paper_root/pivot_iclr2027_supplementary.zip"
"$python_bin" "$project_root/scripts/verify_paper.py" \
  --pdf build/main.pdf \
  --source main.tex \
  --output verification.json \
  --preview preview.png \
  --max-main-pages 9
printf 'pdf=%s\n' "$paper_root/pivot_iclr2027_submission.pdf"
