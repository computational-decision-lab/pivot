# Highway matched-selector physical replication audit

Date: 2026-11-09

Status: `DEV_INDEPENDENT_PHYSICAL_REPLICATION`, ready for review. This is a
single-decision HighwayEnv experiment. It does not establish a multi-round or
general self-improvement claim.

## Frozen protocol

The preregistration is in
`HIGHWAY_MATCHED_PHYSICAL_PREREGISTRATION_2026-11-09.md`. The frozen calibration
cohort has 40 roots with seeds `3700003 + 43*i`; the fresh held-out cohort has
120 roots with seeds `3900003 + 47*i`. This run uses new test roots but reuses
the calibration model collected before the incomplete v1 test attempt; it is
not independent calibration validation. Each root has eight scheduled
action candidates and budget `B=4`. PIVOT-KG, IVR-Trace, Top-Proxy, and
Uniform-Random use the same calibration model, paired-query observation rule,
posterior update, and posterior-mean terminal selector.

The physical backend uses a sealed shared observation bank. A deterministic
candidate/incumbent actor pair is executed once and may be reused by another
selector only after that selector makes the corresponding logical query. Each
selector retains an independent four-query ledger. The post-decision truth
audit is not available to acquisition. This saves duplicate simulator work
without changing any selector's information set.

The run used HighwayEnv 1.12.1, Gymnasium 1.3.0, NumPy 2.5.3, four lanes,
20 traffic vehicles, horizon 20, simulation frequency 5, and action hold 1.
The calibration model hash is
`91d1dc56a4ce6ef5ae7b1cc845bd2170047eb68e70135cd21742878527828d8d`; the
full-study configuration hash is
`f260f962267e14e8b3b153c17d1df1a13eb769d5e12252c03e92576a00c722f2`.

## Physical execution audit

The authoritative results are under
`results/rebuttal/20261109-highway-matched-physical-v2/`. Four disjoint test
shards cover all 120 expected seeds. The aggregate script was rerun from the
four shard manifests and the calibration manifest into a clean verification
directory; it returned exit code 0 and reproduced the authority summary,
including every contrast and the study hash.

The evaluation journal records 3,147 completed physical evaluations and 6,294
simulator calls, with 125,880 environment steps. It reports zero failed,
unresolved, or unmeasured evaluations and `cost_complete=true`. The aggregate
validator also checks eight proxy rows per root, four unique logical queries
per method, query-order/ledger identity, truth-audit equality, terminal
selection, ISR recomputation, disjoint shard seed coverage, and content
manifest hashes.

## Results

Contrasts are `ISR(method, B=4) - ISR(PIVOT-KG, B=4)`; positive values favor
PIVOT-KG. Intervals are two-sided 95% percentile root-bootstrap intervals over
the 120 paired roots.

| Method | Mean contrast | 95% CI |
|---|---:|---:|
| IVR-Trace | -0.00298 | [-0.00729, 0.00079] |
| Top-Proxy | 0.01351 | [0.00238, 0.03069] |
| Uniform-Random | 0.02953 | [0.01662, 0.04556] |
| Uniform exact expectation (audit only) | 0.03330 | [0.02495, 0.04199] |

The primary targeted comparison is therefore null/inconclusive: this run does
not show PIVOT-KG outperforming IVR-Trace. The positive Top-Proxy and Uniform
contrasts are descriptive evidence for this cohort and do not establish a
KG-specific advantage over all targeted allocation rules. Formal LUCB was not
run because one initialization pull per each of eight arms exceeds `B=4`.

## Reproducibility pointers

- Protocol: `docs/rebuttal/audit/HIGHWAY_MATCHED_PHYSICAL_PREREGISTRATION_2026-11-09.md`
- Aggregator: `docs/rebuttal/audit/aggregate_highway_physical_results.py`
- Runner: `docs/rebuttal/audit/run_highway_matched_physical.py`
- Authority summary: `results/rebuttal/20261109-highway-matched-physical-v2/physical-aggregate-v2/summary.json`
- Paired-root rows: `results/rebuttal/20261109-highway-matched-physical-v2/physical-aggregate-v2/paired-root-results.json`

The result is new physical evidence for the rebuttal, while the previously
reported 0.0380 exact-Uniform contrast remains the saved-truth offline replay
used in the submitted redesign analysis. The two estimates should not be
merged or presented as the same experiment.
