# HighwayEnv fixed-protocol replication, 2026-09-21

The single planned replication is complete. The primary B=2 result is
unresolved, and the secondary B=4 result is also unresolved. The original
statistically positive B=4 finding was not reproduced as a statistically
positive finding on the new cohort. An interval crossing zero does not prove
that the true effect is zero. No further benchmark or budget expansion is
part of this study.

The estimand is mean paired `ISR(Uniform HF) - ISR(Calibrated PIVOT-KG)`;
positive values favor PIVOT-KG. Each cohort has 60 test seeds and its own 20
calibration seeds. All seeds are disjoint between cohorts.

| Cohort | Budget | Mean contrast | 95% paired bootstrap CI |
| --- | --- | ---: | --- |
| Original | 1 | -0.022553 | [-0.043912, -0.000774] |
| Original | 2, primary | +0.004870 | [-0.008796, +0.019901] |
| Original | 4, secondary | +0.011482 | [+0.000849, +0.023684] |
| Replication | 2, primary | -0.000648 | [-0.014274, +0.014498] |
| Replication | 4, secondary | +0.003373 | [-0.000740, +0.009931] |

The replication retains the registered five actions, simulator configuration,
covariance shrinkage, sequential query rule, all 60 test roots, and the 4,000
bootstrap draws. B=4 does not replace B=2. Intervals condition on each fitted
calibration model and are not simultaneous across budgets. The comparison
includes calibration, acquisition, and final selection; it does not isolate
acquisition alone. The first two test roots were exposed by a technical smoke
run and retained without tuning or exclusion. This is external DEV evidence.

Execution completed with 3,120 journaled evaluations, zero failures,
420 decisions, 1,380 logical query records, and 2,100 candidate truth records.
The physical ledger counts 6,240 simulator calls and 124,800 environment steps;
end-to-end runner time is 3,847.58 seconds. Calibration completion preceded the
first full-run test evaluation. Logical B excludes calibration and final truth
audits; all physical work is included in the cost ledger.

`evidence/highway/{original,replication}` retains hash-bound public evidence.
`scripts/build_highway_evidence.py` independently recomputes both cohorts,
checks all query and truth records and frozen source hashes, and builds the
paper table, figure, paired CSV, and audit. `reproduction/highway` supplies
the exact source snapshot, dependency lock, and a launcher that works without
historical Git objects or a cloud account. Algorithm files are unchanged from
the executed commits.

The full worker archive has 12,496 files and 17,677,486 bytes. Two independent
local copies and a restore were verified before the temporary worker was
released. The local cloud receipt retains the inventory and release evidence;
private infrastructure details are excluded from the anonymous paper package.
