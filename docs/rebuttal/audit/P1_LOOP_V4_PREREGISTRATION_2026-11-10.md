# P1 multi-round loop v4 preregistration

Date: 2026-11-10  
Status: development protocol; confirmation roots remain untouched.

V3 separated repeated-rollout observation noise from contextual model error, but
its persistent online regression update can extrapolate after a deployment
raises pressure beyond the calibration range. V4 tests a narrower intervention:
reset the contextual correction model at the start of each round, use one
common sentinel query, and spend the remaining query on the declared
acquisition rule. This is a new method variant and does not replace v1--v3.

## Fixed protocol

- Five rounds, four candidates, and response strengths `0.35`, `0.60`, `0.85`.
- The calibration cohort is 240 roots (`520000`--`520239`), with eight paired
  HF rollouts per candidate context and the v3 repeated-rollout noise model.
- Calibration pressure covers `[0.05, 2.20]`, and the contextual feature set
  excludes round fraction because it is constant in the calibration cohort.
- Development uses 50 roots per regime from the `800000` blocks; confirmation
  uses 120 roots per regime from the disjoint `900000` blocks.
- Every method receives exactly two HF queries per round: a common sentinel at
  `candidate_index = round_id mod 4`, then one method-specific query.
- All three v4 methods reset the same contextual Bayesian linear correction
  model at each round; the sentinel and method-specific observation update only
  that round's model. Their second-query rule is the sole method difference:
  exact one-step Gaussian KG, IVR, or descending proxy delta.
- V3's persistent-update comparisons remain a separate diagnostic and are not
  used as the matched v4 baselines.
- Deployment, independent evaluation, and query seeds are shared by root,
  round, and candidate identity. The primary outcome is five-round cumulative
  ISR; utility, harmful adoption, and retention are secondary.
- The six paired contrasts are baseline cumulative ISR minus v4 cumulative ISR.
  Report root bootstrap intervals and sign-flip p-values with Holm correction.
  V4 can claim superiority only within a declared response regime where both
  baseline contrasts have positive 95% lower bounds and both adjusted p-values
  are below `0.05`. All regimes and negative findings are reported.

No confirmation result is used to select a method, tune a threshold, or alter
the response regimes. The protocol is limited to the declared controlled
PerformativeWorld, proposer, candidate generator, and five-round horizon.

### Amendment A1 (before rerun)

The first exploratory v4 development run exposed calibration extrapolation: its
pressure support ended at `0.70`, while later deployments reached about `2.0`.
The `round_fraction` feature was also constant in the calibration cohort. Before
any confirmation run, the calibration cohort is therefore moved to the unused
`520000`--`520239` block, pressure support is expanded to `[0.05, 2.20]`, and
the constant feature is removed. No confirmation outcome was inspected.

The amended protocol id is `P1-LOOP-V4A-2026-11-10`; the original exploratory
v4 output remains archived and is not pooled with the amended run.

Runner: `run_multiround_loop_v4.py`  
Analyzer: `analyze_multiround_loop_v4.py`
