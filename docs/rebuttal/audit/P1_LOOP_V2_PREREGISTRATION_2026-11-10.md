# P1 multi-round loop v2 preregistration

Date: 2026-11-10  
Status: development protocol; the confirmation split remains sealed until the development run and code audit are complete.

## Question

Does contextual online correction calibration improve five-round selection regret in the declared PerformativeWorld loop, and does the PIVOT-KG acquisition rule outperform matched IVR and Top-Proxy allocation under the same observation model and budget?

The earlier loop-v1 result is retained unchanged. This is a new algorithmic variant, `PIVOT-KG-Loop-v2`, not a replacement result for the submitted PIVOT-KG algorithm.

## Hypothesis and limits

The working hypothesis is that v1's correction model is poorly calibrated after deployment moves the environment state away from the narrow initial calibration range. The v1 protocol also assigned method-specific deployment random streams, which weakened paired comparisons. V2 tests contextual calibration and cross-round online updates while using common deployment and evaluation streams. A win is not assumed. A null or negative result will be reported as such.

This controlled study can support claims only for the declared PerformativeWorld dynamics, proposal rule, candidate panel, response strengths, budget, and five-round horizon. It does not establish general self-improving-agent performance or physical HighwayEnv multi-round performance.

## Fixed protocol

- Five deployment rounds, four candidates per round, five methods, and three response regimes: low `0.35`, medium `0.60`, and high `0.85`.
- Calibration uses 240 roots, IDs `500000` through `500239`, with pressure stratified over `[0.05, 0.70]`, incumbent intensity and bias drawn from fixed root-seeded distributions, and all three response strengths balanced by root.
- Development uses 50 roots per regime, IDs beginning at `600000` with disjoint 10,000-ID regime blocks.
- Confirmation uses 120 roots per regime, IDs beginning at `700000` with the same disjoint regime blocks. Confirmation roots are not used for model or protocol selection.
- The test initial pressure is `0.10 + 0.015 * (root_offset mod 7)`; deployment carries the resulting final state into the next round.
- The online methods receive two paired high-fidelity queries per round. One is a common-rule sentinel at candidate index `round_id mod 4`; the second is allocated by the named method. Thus the sentinel consumes the same part of the total budget for PIVOT-KG-Loop-v2, IVR-Online, and Top-Proxy-Online.
- The primary online methods use the same contextual correction-model family, one common-rule sentinel plus one acquisition-specific query, and posterior-mean terminal selector. PIVOT-KG-Loop-v2 uses exact one-step Gaussian KG; IVR-Online uses integrated variance reduction; Top-Proxy-Online queries the largest remaining proxy delta. `Contextual-KG` is a contextual-calibration ablation that conditions its posterior on both within-round observations but resets to the calibration fit at the next round. `Static-PIVOT-KG-v1` is a legacy reference using the original narrow-state static prior; it also receives two queries per round but is not part of the matched contextual-model comparison.
- Evaluation random streams are shared by root, round, and candidate index. Deployment streams are shared by root and round. Query streams are shared by root, round, and candidate ID.
- Primary outcome: per-root cumulative five-round ISR. Secondary outcomes: cumulative selected deployment utility, harmful-adoption rate, and incumbent-retention rate.
- Primary pairwise contrasts: IVR-Online minus PIVOT-KG-Loop-v2 ISR and Top-Proxy-Online minus PIVOT-KG-Loop-v2 ISR. The family covers both contrasts in all three response regimes; paired-root bootstrap 95% intervals and sign-flip tests are reported, with Holm correction over all six tests.
- A confirmatory superiority claim in a regime requires both paired ISR interval lower bounds to exceed zero and both Holm-adjusted p-values to be below `0.05` in that regime. Results that do not meet this rule remain inconclusive. Utility is reported alongside ISR; no utility non-inferiority claim is planned.

## Model

The contextual correction is `g = HF delta - proxy delta`. A Bayesian linear model uses pressure, response strength, incumbent parameters, candidate parameter changes, proxy delta, pressure/action interaction, response/action interaction, and normalized round as features. All methods condition the current-round posterior on each selected observation. For the three primary online methods, a Gaussian rank-one update also carries every observed correction into the next round. Each query's standardized prediction residual is logged for drift diagnosis; no test-tuned residual threshold is used. The common sentinel is a prespecified equal-budget information control, not shared private information: each method maintains its own observation and model state.

Ridge is fixed at `0.5`. The calibration observation variance is estimated from five-fold root-level cross-validation. The contextual model is fit on the full, separate calibration cohort before either test split is run.

## Execution order

1. Run the development split and audit the complete ledger, model fit, shared seeds, and matched budgets.
2. Any code or protocol change after development must be recorded as a versioned amendment before confirmation. Development outcomes remain exploratory.
3. Freeze the runner and protocol hashes, then run the untouched confirmation split once.
4. Report all response regimes, every comparator, uncertainty intervals, and negative findings.

Runner: `run_multiround_loop_v2.py`  
Analyzer: `analyze_multiround_loop_v2.py`
