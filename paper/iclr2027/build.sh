#!/usr/bin/env bash
set -euo pipefail

paper_root="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "$paper_root/../.." && pwd)"
cd "$paper_root"

# Keep the submission PDF byte-stable across clean rebuilds. Callers may
# override the epoch when producing a deliberately versioned release.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1787227200}"

python3 "$project_root/scripts/build_paper_tables.py" \
  --snapshot ../snapshot \
  --output ../tables
python3 "$project_root/scripts/build_v9_paper_snippets.py" --root "$project_root"
python3 "$project_root/scripts/sync_v9_paper_assets.py" --root "$project_root"
mkdir -p build
cp style/iclr2027_conference.bst iclr2027_conference.bst
TEXINPUTS="$paper_root/style:${TEXINPUTS:-}" \
BSTINPUTS="$paper_root/style:${BSTINPUTS:-}" \
BIBINPUTS="$paper_root:${BIBINPUTS:-}" \
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
cp build/main.pdf pivot_iclr2027_submission.pdf
python3 "$project_root/scripts/build_iclr_supplement.py" \
  --project-root "$project_root" \
  --output-root "$paper_root/supplementary" \
  --archive "$paper_root/pivot_iclr2027_supplementary.zip"
python3 "$project_root/scripts/verify_paper.py" \
  --pdf build/main.pdf \
  --source main.tex \
  --output verification.json \
  --preview preview.png \
  --max-main-pages 9
printf 'pdf=%s\n' "$paper_root/pivot_iclr2027_submission.pdf"
