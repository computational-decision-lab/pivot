# HighwayEnv Calibrated Replication Preregistration

Status: frozen before execution. This is one external DEV replication of the
calibrated HighwayEnv result. It is not a new benchmark family and it does not
unlock a universal promotion claim.

## Question and estimands

The primary question is whether calibrated PIVOT-KG has lower
improvement-selection regret than Uniform HF at a fixed two-query budget on
fresh HighwayEnv roots. For each held-out test seed, the paired contrast is

```text
ISR(Uniform HF, B=2) - ISR(Calibrated PIVOT-KG, B=2)
```

Positive values favor calibrated PIVOT-KG. The primary estimate is the mean of
these paired seed contrasts over all 60 held-out test seeds. `B=4` uses the same
contrast and is a secondary budget only. The all-HF oracle is an audit/reference
arm at `B=5`; it is not a competing adaptive method.

## Frozen protocol

- Protocol ID: `highway-physical-reactive-calibrated-replication-v1`.
- Calibration cohort: 20 seeds `2300003 + 37*i`, `i=0..19`.
- Held-out test cohort: 60 seeds `2400003 + 37*i`, `i=0..59`.
- Calibration and test namespaces are distinct and disjoint from the original
  calibrated v2 run.
- HighwayEnv configuration: four lanes, initial lane 1, twenty traffic
  vehicles, density 1.0, horizon 20, simulation frequency 5, action hold 1.
- Candidate panel: incumbent plus exactly five fixed actions: `SLOWER`,
  `LANE_LEFT`, `IDLE`, `LANE_RIGHT`, `FASTER`.
- Calibration is sealed before any test root is opened. The fitted correction
  model is immutable throughout the test cohort.
- Every method receives the same proxy rows and paired actor evaluation
  semantics. Post-decision truth audits are persisted but excluded from the
  logical HF query budget.
- Registered adaptive budgets are exactly `B=2` (primary) and `B=4`
  (secondary). No `B=1` or `B>4` result is part of this replication.

## Analysis and stopping rules

- Compute the paired seed contrast above for every test seed; do not drop seeds,
  trajectories, or outcomes after unsealing.
- Report the arithmetic mean and a two-sided percentile bootstrap 95% interval
  with 4,000 draws. Bootstrap RNG seeds are fixed by the runner and are recorded
  in the output identity.
- The primary conclusion is `supported` only if the B=2 interval is strictly
  above zero. If it crosses zero, report `unresolved`; if it is strictly below
  zero, report `not supported`.
- B=4 is reported descriptively and cannot replace an unresolved B=2 result.
- There is no retuning of candidates, shrinkage, horizon, seeds, budgets, or
  analysis after the test outcomes are visible.
- A failed calibration, missing checkpoint, action-panel mismatch, or any
  execution failure is a protocol failure, not a reason to rerun with altered
  settings.

## Evidence package

The run must retain `identity.json`, `decision-identity.json`, calibration rows
and model, proxy rows, promotion results, query ledger, truth audit,
`evaluation-journal/`, checkpoints, seed registry, cost ledger, summary, and a
content manifest. The final paper package copies the configuration, summary,
paired contrast table, and source hashes; it does not copy credentials or
private cloud lifecycle data.
