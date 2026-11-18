# Highway independent-calibration physical replication audit

Date: 2026-11-09  
Status: `DEV_INDEPENDENT_PHYSICAL_REPLICATION`, ready for review.  
Scope: a single-decision HighwayEnv replication with a new calibration cohort
and a disjoint new test cohort. It is not a multi-round or universal
knowledge-gradient claim.

## Frozen protocol

The preregistration is in
`HIGHWAY_INDEPENDENT_PHYSICAL_PREREGISTRATION_2026-11-09.md`. Calibration uses
40 roots with seeds `4100003 + 53*i`; testing uses 120 disjoint roots with
seeds `4300003 + 59*i`. The smoke-only seeds are disjoint from both cohorts.
The runtime is Python 3.12, HighwayEnv 1.12.1, Gymnasium 1.3.0, NumPy 2.5.3,
and PyYAML 6.0.3. The panel has eight scheduled action candidates, the
primary query budget is `B=4`, and all four selectors use the same correction
model, paired actor-query rule, posterior update, and posterior-mean terminal
selection rule.

The calibration model was fit once on the new 40-root cohort and sealed before
test outcomes were opened. Its canonical JSON digest (not the on-disk file hash) is
`6ec65e6be4aede9ae119273bbd0713792f7278f25ac3efd7a67b18e35fd553c7`.

## Physical execution audit

Four disjoint test shards cover all 120 expected seeds. The independent
aggregator verifies every shard manifest, source identity, seed partition,
eight proxy rows per root, four unique logical queries per method, query-order
identity, truth-audit equality, terminal selection, ISR recomputation, and
post-decision exact Uniform subset rows. It returned exit code 0 with
`failures=[]`.

The test-stage physical journal records 3,141 completed evaluations and 6,282 simulator
calls, with 125,640 environment steps. Failed, unresolved, and unmeasured
evaluations are all zero and `cost_complete=true`.

## Results

Contrasts are `ISR(method, B=4) - ISR(PIVOT-KG, B=4)`; positive values favor
PIVOT-KG. Intervals are two-sided 95% percentile root-bootstrap intervals
over the 120 paired roots.

| Method | Mean contrast | 95% CI |
|---|---:|---:|
| IVR-Trace | 0.00160 | [-0.00420, 0.00708] |
| Top-Proxy | -0.00012 | [-0.00497, 0.00450] |
| Uniform-Random | 0.01668 | [0.00591, 0.02987] |
| Uniform exact expectation (audit only) | 0.01670 | [0.00985, 0.02519] |

The independent calibration cohort therefore reproduces the qualitative
boundary of the earlier physical study: PIVOT-KG improves over Uniform in this
cohort, while the matched targeted comparisons against IVR-Trace and Top-Proxy
are inconclusive. This is evidence of cohort robustness for the stated
HighwayEnv task, not evidence of KG-specific superiority, a new environment,
or repeated co-adaptation.

## Reproducibility pointers

- Runner: `docs/rebuttal/audit/run_highway_independent_physical.py`
- Aggregator: `docs/rebuttal/audit/aggregate_highway_independent_physical.py`
- Protocol: `docs/rebuttal/audit/HIGHWAY_INDEPENDENT_PHYSICAL_PREREGISTRATION_2026-11-09.md`
- Calibration: `results/rebuttal/20261109-highway-independent-calibration-v1/calibration/`
- Authority aggregate: `results/rebuttal/20261109-highway-independent-calibration-v1/aggregate/summary.json`

Independent replay audit `HIGHWAY_INDEPENDENT_VERIFICATION_2026-11-09.json`
passes all seven checks: 1,920 query decisions, 480 terminal decisions, and
8,400 Uniform subsets; the new calibration model is refit exactly. Journal
timestamps place every physical HF query before the root truth audit.
Calibration adds 720 evaluations / 1,440 simulator calls: the full
calibration-plus-test run totals 3,861 evaluations / 7,722 simulator calls.
The separate smoke gate adds 90 evaluations / 180 calls, excluded from the
scientific cohort. The two physical studies change both calibration and test
roots, so their difference cannot isolate calibration variability alone.
