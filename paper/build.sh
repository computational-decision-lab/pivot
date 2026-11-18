#!/usr/bin/env bash
set -euo pipefail

paper_root="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "$paper_root/.." && pwd)"
cd "$paper_root"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"

# Keep the submission PDF byte-stable across clean rebuilds.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1787227200}"
python_bin="${PIVOT_PYTHON:-python3}"

# The submission release uses frozen audited assets. Regeneration of evidence
# is an explicit analysis step, not a side effect of packaging the manuscript.
PIVOT_PYTHON="$python_bin" bash "$paper_root/build_frozen.sh"
supplement_staging="$project_root/artifacts/revision/supplement_staging"
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
