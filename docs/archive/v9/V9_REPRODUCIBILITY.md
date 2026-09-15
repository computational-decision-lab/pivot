# PIVOT V9 Reproducibility

Run from the repository root with the pinned `.venv`:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check experiments/v9 src/pivot/v9 scripts
.venv/bin/mypy experiments/v9 src/pivot/v9
.venv/bin/python -m experiments.v9.validate --root .
.venv/bin/python -m experiments.v9.analyze --root .
.venv/bin/python scripts/build_v9_figures.py --root .
.venv/bin/python scripts/build_v9_tables.py --root .
.venv/bin/python scripts/audit_v9_statistics.py --root .
.venv/bin/python scripts/audit_v9_figures.py --root .
.venv/bin/python scripts/audit_v9_claims.py --root .
```

Confirmatory runs are isolated and resumable:

```bash
.venv/bin/python -m experiments.v9.run --experiment e2c --profile confirmatory --output results/v9/e2c-confirmatory --root .
```

Every output directory contains compressed transition rows, configuration and
provenance, a scientific decision, and a SHA-256 manifest. The V7 snapshot is
the immutable comparison boundary.
