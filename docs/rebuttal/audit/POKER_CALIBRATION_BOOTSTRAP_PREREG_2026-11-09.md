# Poker Calibration-Root Bootstrap Preregistration

Date: 2026-11-09  
Status: frozen before analysis.  
Protocol: `poker-v2b-calibration-root-bootstrap-v1`.

## Question

How much do the long-horizon PIVOT-KG versus matched-selector contrasts vary
when the correction prior is refit on a different resample of calibration
roots?

## Frozen design

- Sample with replacement 12 roots from the existing 12-root Leduc v2b
  calibration cohort for each of 500 refits, using NumPy generator seed
  `20261110`.
- Fit the existing correction model at the frozen default
  `r_floor=0.25`, covariance shrinkage `11/23`, and horizon `h=8`.
- Keep the same 30 held-out test roots, proxy rows, paired-query observation
  banks, and terminal selector for all refits.
- Compare PIVOT-KG against Uniform-v2, IVR-v2, LUCB-v2 (the source's heuristic
  implementation), and Proxy-Only. Primary descriptive contrasts are
  `ISR(baseline) - ISR(PIVOT-KG)` for Uniform-v2 and IVR-v2.
- Seal every selector decision for every resample before reading the saved
  audit labels for scoring.
- Report the 2.5th, 50th, and 97.5th percentiles across calibration refits,
  sign stability, the reference full-calibration contrast, and selection
  flips versus that reference.

This is a bootstrap sensitivity analysis over saved roots. It does not create
new simulator rollouts or an independent calibration cohort. Its percentile
range measures calibration-resample variation conditional on the fixed test
roots; it is not a confidence interval over new test cohorts.

