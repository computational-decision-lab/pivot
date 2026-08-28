# Footprint Analysis

Status: **DEV_ONLY**; phase: **DEV**.

The registered features are computed before any gate or assessment query and
exclude deployment outcomes from the feature vector. The independent unit is
the trajectory/cluster, so transition rows are not treated as independent
replicates.

Current materialized rows: `4` total, with
`4` rows containing proxy and
actor deltas across `2`
trajectory units. Transition-error estimate:
`{"ci_high": 0.0, "ci_low": 0.0, "clusters": 2, "estimate": 0.0, "rows": 4}`.
Improvement-reversal estimate:
`{"ci_high": 0.0, "ci_low": 0.0, "clusters": 2, "estimate": 0.0, "rows": 4}`.

The full feature-by-feature associations and leakage audit are stored in
`artifacts/v15/footprint_analysis.json`. Before confirmatory opening this is a
DEV-only diagnostic, not a paper claim.
