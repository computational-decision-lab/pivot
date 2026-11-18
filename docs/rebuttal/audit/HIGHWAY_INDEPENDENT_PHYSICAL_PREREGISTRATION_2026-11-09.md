# Highway Independent Calibration Physical Replication

Protocol ID: `highway-independent-calibration-v1`  
Status: frozen before collecting outcomes from the new calibration and test
cohorts.  
Scope: an independent random-seed replication of the matched physical
HighwayEnv selector study. This does not test a new environment or establish
general PIVOT-KG superiority.

## Question and estimands

Using a calibration model fit only on a new 40-root calibration cohort, does
PIVOT-KG have lower improvement-selection regret than matched IVR-Trace on a
separate 120-root HighwayEnv test cohort? All selectors use the same candidate
panel, fitted correction model, paired actor observations, query budget, and
posterior-mean terminal decision.

The primary paired contrast is

```text
ISR(IVR-Trace, B=4) - ISR(PIVOT-KG, B=4)
```

Positive values favor PIVOT-KG. The estimate is the mean over all 120 test
roots, with a two-sided percentile root-bootstrap 95% interval using 4,000
draws and the fixed seed recorded by the runner. Top-Proxy and one
root-seeded Uniform-without-replacement selector are secondary comparisons.
The exact expectation over all 70 Uniform subsets is an audit-only reference
computed after decisions are sealed; it is not an executed selector.

## Frozen protocol

- Calibration seeds: `4100003 + 53*i`, `i=0..39` (40 roots).
- Test seeds: `4300003 + 59*i`, `i=0..119` (120 roots).
- Smoke-only seeds: calibration `4090003 + 61*i` and test
  `4090503 + 67*i`, `i=0..1`. Smoke seeds do not enter either full cohort.
- The new calibration and test seeds are mutually disjoint and do not overlap
  the archived Highway cohorts or the previously registered physical
  replication seeds.
- Runtime: Python 3.12, HighwayEnv 1.12.1, Gymnasium 1.3.0, NumPy 2.5.3,
  PyYAML 6.0.3; four lanes, initial lane 1, 20 traffic vehicles, density 1.0,
  horizon 20, simulation frequency 5, and action hold 1.
- Candidate panel: the frozen incumbent and eight scheduled action
  candidates: four actions (`SLOWER`, `LANE_LEFT`, `LANE_RIGHT`, `FASTER`),
  each applied in the early or late half of the horizon.
- Fit one Gaussian proxy-to-actor correction model with covariance shrinkage
  0.5 on the 40 new calibration roots. Seal the model before test outcomes are
  opened. No calibration row from the earlier cohort is reused.
- Each test root is first scored in observer mode. PIVOT-KG, IVR-Trace,
  Top-Proxy, and Uniform-Random each receive exactly four sequential paired
  actor queries. A paired query is the candidate actor rollout minus the
  incumbent actor rollout under the same root seed.
- The physical backend may reuse the sealed deterministic actor result for a
  candidate/root pair only after a selector makes that logical query. Every
  selector has its own four-query ledger. Reuse does not expose unqueried
  outcomes or the post-decision truth audit.
- All selectors share the fitted correction model, posterior update, and
  terminal rule: choose the highest posterior-mean candidate among the eight
  candidates and the zero-gain incumbent; ties retain the incumbent.
- PIVOT-KG uses the exact one-step correlated-Gaussian knowledge-gradient
  score. IVR-Trace chooses the candidate with the largest one-step reduction
  in total posterior variance. Top-Proxy follows proxy gain; Uniform samples
  without replacement using a fixed root-derived seed.
- Formal LUCB is not run because its one-pull-per-arm initialization requires
  eight queries, exceeding the registered budget of four.
- Actor truth audits run only after all four decisions on a root are sealed.
  Failures are recorded and are not silently replaced with different seeds.
- Calibration and test may be partitioned into disjoint index ranges for
  execution. The final analysis requires exact coverage of all 40 calibration
  and 120 test seeds, and pools each test root exactly once.

## Interpretation boundary

This is an independent calibration-and-test seed cohort within the same
HighwayEnv task and frozen simulator implementation. It can assess sensitivity
to the random-root cohort and calibration fit. A null or negative IVR contrast
will be reported as such; favorable Uniform or Top-Proxy contrasts do not
establish a KG-specific advantage over targeted allocation methods. This
single-decision study does not establish multi-round self-improvement.

The run retains the source/dependency identity, calibration rows and model,
test proxy rows, per-selector query ledger, sealed decisions, post-decision
truth audit, evaluation journal, seed registry, cost ledger, paired-root
contrasts, and content manifests.
