# P0-B Poker prior sensitivity

Date: 2026-11-09

This is an offline replay of the sealed Leduc v2b decision banks. It does not
run OpenSpiel or create new rollouts. Decisions were made from the saved proxy
and selection banks; the saved audit labels were read only after the decision
ledger was sealed. All configurations use the same 12 calibration roots, 30
held-out test roots, two paired high-fidelity queries, and the same terminal
selection rule.

The grid varies the observation-variance floor
`r_floor in {0.125, 0.25, 0.5, 1.0}` and covariance shrinkage
`alpha in {0, 0.25, 0.47826087, 0.75, 1}`. The default
`r_floor=0.25, alpha=0.47826087` reproduces the frozen long-horizon prior and
the frozen primary result: PIVOT-KG ISR `0.00927`, Uniform ISR `0.04575`, and
Uniform minus PIVOT-KG `0.03649 [0.01395, 0.06437]` over roots.

Across the 20 configurations, the long-horizon mean ISR ranges are:

| Method | Mean ISR range over 20 fits |
|---|---:|
| PIVOT-KG | `0.00844--0.07249` |
| Uniform-v2 | `0.03699--0.09494` |
| IVR-v2 | `0.00844--0.07249` |
| LUCB-v2 | `0.00927--0.07249` |

At the lower and default floors, Uniform remains worse than PIVOT-KG across
the recorded configurations. Under the most conservative configuration
(`r_floor=1.0`, `alpha=1.0`), the Uniform minus PIVOT contrast is
`0.02246 [-0.00018, 0.05480]`; therefore the grid does not support a claim of
uniform statistical significance under arbitrary regularization. Targeted
methods often make the same final selection as PIVOT-KG; this is evidence that
the formulation and decision-aware validation matter more than a universal
claim of KG-specific superiority.

The short-horizon rows are all ISR zero in this sealed cohort and do not
provide a useful sensitivity contrast. The grid is a hyperparameter sensitivity
analysis conditional on the saved calibration and test cohorts, not an
independent calibration-set validation.

Input hashes and all per-configuration rows are in
`results/rebuttal/20261109-p0-poker-sensitivity-v3/summary.json` and
`sensitivity_rows.json`.

The normalized inputs under `docs/rebuttal/audit_inputs/poker_v2b/` produce the
same sealed decision hash in `results/rebuttal/20261109-p0-poker-sensitivity-input-replay-v1/`.
The bundle records the public protocol file hash and the protocol's internal
hash field separately because they are different byte-level identifiers.

## Calibration-root resampling

The follow-up 500-refit bootstrap is reported separately in
`POKER_CALIBRATION_BOOTSTRAP_REPORT_2026-11-09.md`. It resamples the 12 frozen
calibration roots with replacement and keeps the same 30 test roots fixed.
Uniform-minus-PIVOT remains positive in all refits (median `0.03649`,
calibration-refit percentile range `[0.02278, 0.11266]`); IVR-minus-PIVOT ISR
is zero in every refit. These are calibration-resampling results conditional
on the saved test cohort, not an independent calibration cohort.
