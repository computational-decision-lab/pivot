# PIVOT V9 Repository Audit

This audit freezes the V7 publication boundary before any V9 result is used.

## Baseline

- Baseline manifest: `snapshot/v9_preupgrade/manifest.json`.
- Baseline source commit: recorded by `scripts/freeze_v9_baseline.py`.
- V7 PDFs, supplement, figures, results, claims, and provenance are hash-indexed
  under `snapshot/v9_preupgrade`; tracked baseline blobs are verified against
  commit `c5c9496`, while ignored LaTeX/supplement derivatives are explicitly
  reported as derived in `artifacts/v9/baseline_verification.json`.
  The parent checkout status inventory is intentionally omitted from the
  public manifest.
- V9 artifacts are isolated under `results/v9`, `figures/v9`, `tables/v9`, and
  `artifacts/v9`.

## Required checks

```bash
.venv/bin/python scripts/freeze_v9_baseline.py
.venv/bin/python scripts/freeze_v9_baseline.py --verify
.venv/bin/python -m experiments.v9.run --experiment e2c --profile smoke --output /tmp/pivot-e2c-smoke --root .
.venv/bin/python -m experiments.v9.validate --root .
```

The only permitted scientific terminal states are `IMPLEMENTATION_FAILURE`,
`DESIGN_INVALID`, `UNDERPOWERED`, `HYPOTHESIS_SUPPORTED`, and
`HYPOTHESIS_NOT_SUPPORTED`. A profile with fewer than 30 independent seeds is
diagnostic and cannot support a confirmatory claim.

## Boundary

No credentials, live orders, capital authorization, raw vendor L2, M3 outputs,
LLM-generated candidate logs, or private execution data are part of the V9
release tree. The finance extension remains a testbed boundary, not market
ground truth.
