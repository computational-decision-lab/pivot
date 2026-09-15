# IMPROVE-X / PIVOT ICLR 2027 Supplementary Artifact

This archive contains the anonymous source, public configurations, tests, the
controlled ImprovementBench v1/v2 releases, and the hash-indexed paper
snapshot used for the submission PDF. The sealed V15 task manifest and lock
history remain local; only the redacted task-membership summary is included.
From the
repository root, install the project in editable mode and run:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/python scripts/build_paper_tables.py --snapshot paper/snapshot --output paper/tables
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

The frozen confirmatory package is included under `results/v9`; publication
transforms are under the historical figure/source directories. They do not rerun
science: they read hash-indexed source rows and emit PDF/SVG/PNG figures plus
CSV provenance tables. Rebuild and audit the complete package with:

```bash
.venv/bin/python -m experiments.v15 reports --root .
```

The manuscript reports the scientific names of the evidence layers. Internal
run identifiers are retained only in manifests and source metadata.

The editable architecture source is adapted from the pinned OpenTikZ
`system-block-diagram` template. Rebuild it after installing the lock-bound
checkout with `scripts/bootstrap_opentikz.py` and
`scripts/build_opentikz_architecture.py`.
