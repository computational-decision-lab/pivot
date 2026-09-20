#!/usr/bin/env bash
set -euo pipefail

paper_root="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "$paper_root/.." && pwd)"
cd "$paper_root"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"

# Keep the submission PDF byte-stable across clean rebuilds.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1787227200}"
python_bin="$project_root/.venv/bin/python"
if [ ! -x "$python_bin" ]; then
  python_bin="python3"
fi

"$python_bin" "$project_root/scripts/build_v15_paper_assets.py" --root "$project_root"
"$python_bin" "$project_root/scripts/build_paper_snippets.py" --root "$project_root"
"$python_bin" "$project_root/scripts/build_revision_evidence.py" --root "$project_root"
mkdir -p build
cp style/iclr2027_conference.bst iclr2027_conference.bst
TEXINPUTS="$paper_root/style:${TEXINPUTS:-}" \
BSTINPUTS="$paper_root/style:${BSTINPUTS:-}" \
BIBINPUTS="$paper_root:${BIBINPUTS:-}" \
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
cp build/main.pdf pivot_iclr2027_submission.pdf
"$python_bin" "$project_root/scripts/verify_paper.py" \
  --pdf build/main.pdf \
  --source main.tex \
  --output verification.json \
  --preview preview.png \
  --max-main-pages 9
supplement_staging="$project_root/artifacts/revision/supplement_staging"
rm -rf "$supplement_staging"
"$python_bin" "$project_root/scripts/build_iclr_supplement.py" \
  --project-root "$project_root" \
  --output-root "$supplement_staging" \
  --archive "$paper_root/pivot_iclr2027_supplementary.zip"
"$python_bin" "$project_root/scripts/verify_iclr_submission.py" \
  --pdf "$paper_root/pivot_iclr2027_submission.pdf" \
  --source "$paper_root/main.tex" \
  --supplement "$paper_root/pivot_iclr2027_supplementary.zip" \
  --style-dir "$paper_root/style" \
  --aux "$paper_root/build/main.aux" \
  --output "$paper_root/submission_verification.json"
printf 'pdf=%s\n' "$paper_root/pivot_iclr2027_submission.pdf"
