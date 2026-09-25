# HighwayEnv B=4 redesign reproduction

This snapshot contains the eight-candidate HighwayEnv redesign reported in
the paper. It is separate from the historical five-candidate cohort so that
the older source hashes remain auditable. The redesign uses 40 calibration
seeds, 120 disjoint test seeds, B=4 as the primary budget, and B=2 as a
secondary budget. Two test roots were exposed by a technical smoke run before
the formal run; they were retained without exclusions or retuning.

From the artifact root, install the locked dependencies and verify the source
snapshot:

```bash
python3.12 -m venv .venv-highway-redesign
.venv-highway-redesign/bin/python -m pip install -r reproduction/highway_redesign/requirements-lock.txt
.venv-highway-redesign/bin/python reproduction/highway_redesign/run.py --check-only
```

The evidence audit is rerun by the main paper build:

```bash
.venv-highway-redesign/bin/python scripts/build_highway_evidence.py --root .
```

The complete cloud run is preserved under `evidence/highway/redesign`. A new
run must use a new output directory and the registered seeds; changing the
candidate panel, budget assignment, or seed cohort would define a different
experiment.
