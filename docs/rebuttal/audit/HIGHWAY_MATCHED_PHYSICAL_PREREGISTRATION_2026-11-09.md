# Highway Matched-Selector Physical Replication

Protocol ID: `highway-physical-reactive-matched-targeted-v2`  
Status: frozen before opening the new test cohort.  
Scope: independent external DEV replication; not a confirmatory promotion or a
replacement for the previously reported redesign.

## Question and estimands

On fresh HighwayEnv roots, does exact calibrated PIVOT-KG have lower
improvement-selection regret than a matched IVR-Trace allocator with the same
candidate panel, calibration model, observation rule, query budget, and
posterior-mean terminal decision?

For each held-out root, the primary paired contrast is

```text
ISR(IVR-Trace, B=4) - ISR(PIVOT-KG, B=4)
```

Positive values favor PIVOT-KG. The primary estimate is the arithmetic mean
over all 120 test roots with a two-sided percentile root-bootstrap 95% interval
(4,000 draws; fixed RNG seed recorded by the runner). Secondary contrasts use
Top-Proxy and one preregistered Uniform-without-replacement subset per root.
Their intervals are descriptive. A post-decision exact expectation over all
70 Uniform subsets is an audit-only reference and is not described as an
executed Uniform selector.

## Frozen protocol

- Calibration seeds: `3700003 + 43*i`, `i=0..39` (40 roots).
- Held-out test seeds: `3900003 + 47*i`, `i=0..119` (120 roots).
- The calibration and test seed sets are disjoint from each other and from the
  archived 2026-09 Highway cohorts. This v2 test namespace also excludes the
  incomplete, internally exposed v1 test attempt.
- Python `3.12`; HighwayEnv `1.12.1`; Gymnasium `1.3.0`; NumPy `2.5.3`;
  PyYAML `6.0.3`; four lanes; initial lane 1; 20 traffic vehicles;
  density 1.0; horizon 20; simulation frequency 5; action hold 1.
- The candidate panel is the frozen incumbent plus eight action/schedule
  candidates: four actions (`SLOWER`, `LANE_LEFT`, `LANE_RIGHT`, `FASTER`),
  each applied in the early or late half of the horizon.
- Fit one Gaussian correction model on the 40 calibration roots and freeze it
  before reading any held-out actor outcomes.
- On each test root, first score the full candidate panel in observer mode.
  PIVOT-KG, IVR-Trace, Top-Proxy, and Uniform then each receive exactly four
  sequential paired actor queries. A paired query is the candidate actor
  rollout minus the incumbent actor rollout under the same root seed.
- The physical backend uses a common observation bank: each deterministic
  seed/candidate actor pair is executed once and its sealed result may be reused
  when another selector queries the same pair. Each selector still has its own
  four-query logical ledger and posterior sequence; reuse never exposes an
  unqueried candidate or a post-decision truth audit to that selector.
- Calibration data use the same frozen 40-root cohort collected before the v1
  test attempt. The v2 selector comparison changes only physical observation
  reuse and uses a fresh, disjoint test-seed namespace; no v1 test result enters
  the v2 calibration model or analysis.
- All methods use the same fitted correction model, posterior update, and
  terminal selector (highest posterior mean over the eight candidates and the
  zero-gain incumbent; ties retain the incumbent). Test truth audits run only
  after all four decisions on that root are sealed.
- PIVOT-KG chooses the exact one-step correlated-Gaussian knowledge-gradient
  score. IVR-Trace chooses the candidate with greatest one-step reduction in
  total posterior variance, then updates the posterior with its observation.
  Top-Proxy queries candidates by descending proxy gain. Uniform samples a
  four-candidate subset without replacement using a fixed root-derived seed.
- Formal LUCB is not run: its one-pull-per-arm initialization requires eight
  queries, exceeding the registered budget of four.
- No seed, candidate, model, budget, or analysis change is permitted after the
  test outcomes are opened. A simulator or journal failure is recorded as a
  protocol failure and is not silently replaced with a different seed.
- Calibration and test execution may each be partitioned into four contiguous,
  disjoint seed shards. Shards share a durable checkpoint store and seed
  registry, and use an identical full-study config hash. Calibration rows are
  merged and the model is fitted once before any test shard starts. This is a
  compute scheduling choice only; the final analysis pools the 120 unique test
  roots once.

## Smoke gate and evidence

Before the full cohort, a smoke run uses two separate calibration seeds
`3690003 + 47*i` and two separate test seeds `3691003 + 47*i`, `i=0..1`.
Smoke roots are excluded from the full analysis. A failed smoke gate blocks
the full run until its cause is resolved and the protocol is versioned anew.

The run retains its configuration, source/dependency hashes, calibration
rows/model, test proxy rows, physical query ledger, sealed decisions,
post-decision actor truth audit, evaluation journal, seed registry, cost
ledger, per-root contrasts, summary, and content manifest. The full run is
labelled `DEV_INDEPENDENT_PHYSICAL_REPLICATION`; it does not establish
general-purpose KG superiority or multi-round self-improvement.
