# IMPROVE-X / PIVOT ICLR 2027 Supplementary Artifact

This archive contains the anonymous source, public configurations, tests, the
controlled ImprovementBench releases, the current manuscript sources, and the
hash-indexed paper snapshot used for the submission PDF. The three sealed
response-world cohorts (Leduc, Kuhn, and Melting Pot) are copied byte-for-byte
under `evidence/latest`; the public audit and decision-relevance bridge are
included alongside them. Sealed task instructions and lock history remain
local; only the redacted task-membership summary is included.
The HighwayEnv evidence and frozen simulator source are under
`evidence/highway` and `reproduction/highway`. The latter README documents the
seed cohorts, exact dependency lock, budgets, smoke exposure, and simulator
rerun command. From the extracted artifact root, install Python 3.12 and the
locked dependencies, then recompute the Highway table and figure:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r reproduction/highway/requirements-lock.txt
.venv/bin/python scripts/build_highway_evidence.py --root .
.venv/bin/python reproduction/highway/run.py --check-only
```

The public finance audit uses virtual fills and observational depth
proxies. No credentials, private data, raw vendor archives, live
orders, or author identity metadata are included.

The controlled value-versus-improvement diagnostic can be regenerated with:

```bash
.venv/bin/python experiments/e4_value_vs_improvement.py \
  --config configs/sweeps/e4_value_vs_improvement.yaml \
  --output artifacts/v9/reproduction/e4-value-vs-improvement
```

This is a controlled estimand diagnostic, not a universal method or market
performance claim.

Earlier controlled checks and frozen external classifications remain under
`results/theory` and `results/v7` for provenance. Decompress a row stream with
`gzip -dk transition_rows.jsonl.gz` when a row-level audit is needed. The
analytic checks can be regenerated with:

```bash
.venv/bin/python experiments/e10_theory_empirical.py \
  --config configs/theory/v6_empirical.yaml \
  --output artifacts/v9/reproduction/e10-theory-empirical
```

They test the constructive Global Fidelity Blindness and Response-Footprint
Sensitivity claims; they are not causal market evidence.

The evidence archive is a frozen copy, not a rerun. Rebuild the current paper
from this archive using TeX Live (including latexmk) and Poppler (pdfinfo,
pdftotext, pdffonts, and pdftoppm):

```bash
PIVOT_PYTHON="$PWD/.venv/bin/python" bash paper/build_frozen.sh
```

The manuscript reports the scientific names of the evidence layers. Internal
run identifiers are retained only in manifests and source metadata.

The editable architecture source is adapted from the pinned OpenTikZ
`system-block-diagram` template. Rebuild it after installing the lock-bound
checkout with `scripts/bootstrap_opentikz.py` and
`scripts/build_opentikz_architecture.py`.
