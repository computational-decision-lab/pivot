# P1 loop v3c: equal-budget no-sentinel confirmation protocol

This is a dated amendment after the v3b development run. It tests whether
the common sentinel, rather than the acquisition rule, accounts for the
remaining gap against Top-Proxy. The v3b results remain historical and are
not pooled with this protocol. The confirmation split uses roots not used by
v3b development or confirmation.

## Locked design

- Controlled `PerformativeWorld`, fixed proposer, four candidates, five rounds.
- Low, medium, and high response strengths are `0.35`, `0.60`, and `0.85`.
- Development uses root blocks `860000`, `870000`, and `880000`; confirmation
  uses `960000`, `970000`, and `980000`, 120 roots per block.
- Calibration uses the sealed v3b cohort `510000`--`510239`, eight paired
  rollouts per candidate, and the contextual Bayesian linear correction model.
- Every method receives exactly two sequential paired HF queries per round.
  There is no common sentinel; after each observation the within-round latent
  posterior is conditioned with its positive observation variance.
- PIVOT-KG-Loop-v3c, IVR-Online-v3c, and Top-Proxy-Online-v3c share the
  candidate panel, calibration data, response operator, terminal rule, seeds,
  and total budget. They differ only in the acquisition rule. The correction
  model is reset to the sealed calibration fit each round (`cross_round_update`
  is fixed to `none` for the primary confirmation).
- The terminal decision is posterior-mean selection over the incumbent and all
  four candidates. Deployment then carries the resulting world state to the
  next round.

## Analysis

The primary estimand is paired five-round cumulative ISR (lower is better).
The secondary deployment metric is the sum of the absolute actor-world return
of the actually deployed policy on the state-carrying deployment trajectory.
Selected policy delta, harmful adoption, and retention are also reported; a
delta sum is not treated as an absolute return sum.

For each regime, report baseline-minus-PIVOT paired contrasts for ISR and
PIVOT-minus-baseline contrasts for deployment return, using 10,000 paired-root
bootstraps and 100,000 two-sided sign flips. Holm correction is applied over
the six ISR contrasts. A regime-level superiority claim requires a positive
ISR confidence lower bound and adjusted sign-flip p below `0.05` against both
baselines. A joint deployment-return claim additionally requires both
two-sided return confidence lower bounds to be non-negative. All regimes,
including negative and unresolved outcomes, are retained.

The confirmation run is executed once after this amendment. No method,
threshold, root, response regime, or metric is selected from its outcomes.
The claim is limited to this controlled world, proposer, candidate generator,
budget, and five-round horizon.
