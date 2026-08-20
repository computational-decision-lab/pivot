# PIVOT ICLR 2027 Submission Package

This directory is the anonymous ICLR 2027 submission candidate for
*When Better Gets Worse: Improvement Fidelity of Self-Improvement Operators in
Adaptive Worlds*.

## Build and audit

```bash
./build.sh
cd ../..
.venv/bin/python scripts/build_iclr_supplement.py \
  --output-root paper/iclr2027/supplementary \
  --archive paper/iclr2027/pivot_iclr2027_supplementary.zip
.venv/bin/python scripts/verify_iclr_submission.py \
  --pdf paper/iclr2027/build/main.pdf \
  --source paper/iclr2027/main.tex \
  --supplement paper/iclr2027/pivot_iclr2027_supplementary.zip \
  --style-dir paper/iclr2027/style \
  --output paper/iclr2027/submission_verification.json
```

The current decision is `CONDITIONAL GO`: local formatting and artifact gates
pass, but author-side OpenReview gates and external causal/strategic validity
remain open. No submission was uploaded.

The controlled E4 spotlight diagnostic is reproducible with:

```bash
.venv/bin/python experiments/e4_value_vs_improvement.py \
  --config configs/sweeps/e4_value_vs_improvement.yaml \
  --output /tmp/pivot-e4-value-vs-improvement
```

It compares an evaluator with slightly better isolated policy-value accuracy
against one with exact paired deltas. The result is intentionally labelled a
controlled estimand diagnostic, not a universal method claim.
