# Registered Run Protocol

The registered layer separates implementation smoke checks from evidence used
for a scientific gate.

## Files

- `configs/registered/p2.yaml`: three disjoint five-seed P2 jobs.
- `configs/registered/e4.yaml`: three disjoint five-seed E4 jobs.
- `configs/registered/e5.yaml`: three disjoint five-seed E5 jobs.
- `configs/registered/e6.yaml` through `configs/registered/e9.yaml`: isolated
  finance, strategic, and closed-loop jobs.
- `scripts/run_registered.py`: materializes immutable per-run configs and
  refuses to overwrite a non-empty run directory.
- `scripts/aggregate_registered.py`: computes independent-run summaries and
  paired intervals, retaining failed runs in the input record.

## Commands

```bash
python scripts/run_registered.py \
  --registry configs/registered/p2.yaml \
  --output results/registered/p2

python scripts/aggregate_registered.py \
  --experiment p2 \
  --inputs results/registered/p2/p2-r01 results/registered/p2/p2-r02 results/registered/p2/p2-r03 \
  --output results/registered/p2-summary.json
```

Use the analogous `e4` and `e5` registries. Outputs contain `config.yaml`,
`run_manifest.json`, `stdout.log`, `stderr.log`, and the experiment's normal
artifacts. A second invocation against the same non-empty run directory fails
instead of silently replacing evidence.

For the later stages:

```bash
python scripts/run_registered.py --registry configs/registered/e6.yaml --output results/registered/e6
python scripts/aggregate_registered.py --experiment e6 --target-participation 0.05 \
  --inputs results/registered/e6/e6-r01 results/registered/e6/e6-r02 results/registered/e6/e6-r03 \
  --output results/registered/e6-summary.json

python scripts/run_registered.py --registry configs/registered/e7.yaml --output results/registered/e7
python scripts/run_registered.py --registry configs/registered/e8.yaml --output results/registered/e8
python scripts/aggregate_registered.py --experiment f --mode adaptive \
  --inputs results/registered/e7/e7-r01 results/registered/e7/e7-r02 results/registered/e7/e7-r03 \
  --e8-inputs results/registered/e8/e8-r01 results/registered/e8/e8-r02 results/registered/e8/e8-r03 \
  --output results/registered/f-summary.json

python scripts/run_registered.py --registry configs/registered/e9.yaml --output results/registered/e9
python scripts/aggregate_registered.py --experiment e9 \
  --inputs results/registered/e9/e9-r01 results/registered/e9/e9-r02 results/registered/e9/e9-r03 \
  --output results/registered/e9-summary.json
```

## Statistical contract

- Seed sets are non-empty, non-negative, and disjoint across registered runs.
- P2 response contrasts are computed within each run before the independent-run
  bootstrap interval.
- E4 model differences are paired by registered run.
- E5 CTI/ISR differences are paired by run and candidate group at a fixed HF
  budget; `group_metrics.jsonl` is the source table.
- Failed runs remain listed and cannot contribute to a gate estimate.
- A gate requires at least three valid registered runs and a positive lower
  confidence bound for its predeclared criterion.

The resulting JSON is evidence for the gate ledger, not an automatic paper
claim. Fixture-level passes still require external review of environment
plausibility and the registered configuration before being promoted.
