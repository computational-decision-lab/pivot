# Poker Calibration-Root Bootstrap

Protocol: `poker-v2b-calibration-root-bootstrap-v1`  
Status: `PASS`; 500 refits, independently audited.  
Scope: offline sensitivity replay over the sealed Leduc v2b evidence bank.

## Design

Each refit samples 12 roots with replacement from the existing 12-root
calibration cohort and fits the frozen default long-horizon correction prior
(`r_floor=0.25`, covariance shrinkage `11/23`). The same 30 held-out test roots,
proxy rows, paired-query observations, and terminal selection rule are used for
all 500 fits. PIVOT-KG is compared with Uniform-v2, IVR-v2, the source's
LUCB-v2 heuristic, and Proxy-Only. The decision ledger is sealed before audit
labels are used to score ISR.

## Results

The default full-cohort fit exactly reproduces the frozen prior and the
previous sensitivity-grid contrast. Across the 500 calibration resamples, the
Uniform-minus-PIVOT mean ISR contrast has median `0.03649` and a 2.5th-97.5th
percentile range of `[0.02278, 0.11266]`; it is positive in all 500 refits.
The IVR-minus-PIVOT ISR contrast is zero in all refits. For comparison, the
LUCB-v2 heuristic contrast is positive in 6.8% of refits and zero otherwise.
PIVOT-KG's selected candidate differs from its full-calibration reference on
10.1% of fixed test roots on average across refits (maximum 33.3%).

The IVR result does not establish identical selections or KG-specific
superiority; it shows that the ISR contrast is zero on this fixed test cohort.
The percentile range captures calibration-resample variation conditional on
these same 30 test roots. This is not a new independent calibration cohort,
not a new simulator run, and not a confidence interval over new test roots.

## Artifacts

- Preregistration: `docs/rebuttal/audit/POKER_CALIBRATION_BOOTSTRAP_PREREG_2026-11-09.md`
- Runner: `docs/rebuttal/audit/run_poker_calibration_bootstrap.py`
- Independent audit: `docs/rebuttal/audit/audit_poker_calibration_bootstrap.py`
- Verification output: `docs/rebuttal/audit/POKER_CALIBRATION_BOOTSTRAP_VERIFICATION_2026-11-09.json`
- Sealed inputs, decisions, scores, bootstrap rows, and hashes: `results/rebuttal/20261109-p0-poker-calibration-bootstrap-v1/`
