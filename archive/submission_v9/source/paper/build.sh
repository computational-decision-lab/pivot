#!/usr/bin/env bash
set -euo pipefail

paper_root="$(cd "$(dirname "$0")" && pwd)"
cd "$paper_root"
python3 ../scripts/build_paper_tables.py --snapshot snapshot --output tables
mkdir -p build
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
cp build/main.pdf pivot_working_paper.pdf
python3 ../scripts/verify_paper.py --pdf build/main.pdf --source main.tex --output verification.json --preview preview.png
printf 'paper=%s\n' "$paper_root/pivot_working_paper.pdf"
