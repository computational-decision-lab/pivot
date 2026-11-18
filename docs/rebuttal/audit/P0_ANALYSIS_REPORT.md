# P0 Offline Analysis Report

Date: 2026-11-09

## Matched decision replays

`results/rebuttal/20261109-p0-offline-v1` replays query allocation and terminal
selection on the frozen 120-root test cohort. All methods use the same eight
candidate proxy rows, fitted correction prior, observation model, query budget,
and posterior-mean terminal rule. Candidate outcomes come from the saved
post-decision actor truth audit. The runner does not call HighwayEnv.

`IVR-Trace` selects the next unobserved candidate by maximum expected reduction
in the trace of posterior covariance,
`||Sigma[:,j]||^2 / (Sigma[j,j] + s_j^2)`, with fixed candidate-order tie
breaking. It then conditions on that queried candidate's saved actor delta.
`Top-Proxy` queries the `B` candidates with largest proxy deltas, then applies
the shared posterior and terminal rule. `Uniform-Exact` is the exact average
over all `choose(8,B)` subsets, as in the saved primary analysis. Contrasts
are `baseline ISR - PIVOT-KG ISR`; positive values favor PIVOT-KG. Intervals
resample paired roots 4,000 times using common bootstrap indices per budget.

| Baseline minus PIVOT-KG | B=2 | B=4 (primary) |
|---|---:|---:|
| Uniform-Exact | `0.0456 [0.0343, 0.0587]` | `0.0380 [0.0300, 0.0474]` |
| IVR-Trace | `0.0383 [0.0257, 0.0512]` | `-0.0028 [-0.0073, 0.0009]` |
| Top-Proxy | `0.0449 [0.0334, 0.0567]` | `0.0121 [0.0057, 0.0198]` |

At B=4, the IVR-Trace interval crosses zero and 107 of 120 roots have equal
ISR, so this replay does not establish a KG-specific advantage over IVR-Trace.
The Top-Proxy contrast favors PIVOT-KG on average. Both are retrospective
counterfactual replays, not new physical simulator evaluations or
pre-registered confirmatory comparisons.

The frozen formal paired-LUCB1 implementation cannot initialize eight arms
with budget four: its required initialization cost is `K=8`, while `B=4`.
We leave LUCB as not run rather than substitute a non-LUCB heuristic.

## Calibration-root bootstrap

We drew 500 bootstrap samples of 40 calibration roots with replacement, refit
the same shrinkage Gaussian correction model, then replayed the fixed 120 test
roots. Test proxy and actor truth rows stayed unchanged. These percentiles
describe calibration-fit sensitivity conditional on the test cohort; they are
not a full sampling interval and do not constitute an independent calibration
cohort.

| Budget | Mean fitted Uniform-minus-PIVOT contrast | 2.5-97.5% across fits | Positive sign | Mean selection flip rate |
|---:|---:|---:|---:|---:|
| B=2 | `0.0491` | `[0.0265, 0.0679]` | 500/500 | 7.9% |
| B=4 | `0.0392` | `[0.0245, 0.0542]` | 500/500 | 2.9% |

The 500-fit rerun is saved separately in
`results/rebuttal/20261109-p0-offline-v2-500`; it used the same frozen inputs
and bootstrap seed, while the primary offline replay remains preserved in v1.
The B=4 result is directionally stable to calibration-root resampling, while a
small fraction of root-level selected policies change across fitted priors.
This analysis does not answer independent-calibration validation.

The normalized input bundle is under
`docs/rebuttal/audit_inputs/highway_redesign/`. A 10-fit path replay from that
bundle produced byte-identical `root-rows.json` and identical matched-selector
contrasts; the formal 500-fit numbers above remain the v2-500 output.
