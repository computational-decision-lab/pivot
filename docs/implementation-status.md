# Implementation Status

Updated 2026-08-19 after the first end-to-end implementation pass.

## Implemented

| Phase | Evidence in this checkout | Status |
| --- | --- | --- |
| P0 | frozen policy/transition schemas, paired evaluation, metrics, bootstrap CI, append-only JSONL + Parquet manifest | Implemented and tested |
| P1 | controlled performative observer/actor world, synthetic and RL-style update operators | Implemented and tested |
| P2 | registered response x footprint x optimization x seed sweep; E1/E2/E3 commands; five first-milestone artifacts | Implemented; smoke/registered run completed |
| P3 | matched-budget global value vs differential transfer models and E4 ledger | Implemented; clean run completed |
| P4/P5 | acquisition baselines, PIVOT heuristic, round harness, E5 frontier, figure validation pipeline | Implemented; clean run completed |
| P6/P7 | virtual F0/F1 replay and F2 interactive participation fixture | Implemented and tested |
| P8/P9 | S0/S1/S2 strategic wrapper, E7/E8/E9 scripts, closed-loop query ledger | Implemented; only smoke evidence so far |

## Scientific gates

The implementation does not silently promote smoke results to paper claims.

- Gate A/B: controlled reversal is observable and response-dependent in the
  registered fixture, but require independent frozen runs before a formal pass.
- Gate C: E4 budget matching is verified; the current baseline is intentionally
  simple and does not yet establish a final superiority claim.
- Gate D: stratified-calibration E5 smoke improves over Random and Top Proxy at
  the smallest HF budgets, but the method gate remains open until independent
  registered runs and paired intervals confirm the effect.
- Gate E: F2 changes delta at plausible fixture participation values and equals
  F1 at zero participation; calibration against a real execution source is
  still required.
- Gate F: the smoke E7 run exhibits the desired sign pattern, but needs the
  registered opponent/seed sweep and an incremental-effect analysis.

## Remaining before a submission-grade result

1. Add independent run aggregation and paired confidence intervals across the
   E1--E9 seed/config jobs.
2. Improve PIVOT calibration/acquisition until the fixed-budget comparison is
   either positive or the claim is narrowed to a valid null result.
3. Replace the synthetic finance fixture with versioned public-data execution
   replay and calibrate F2 impact/recovery.
4. Run the full strategic sweep, then freeze Gate F evidence.
5. Keep LLM/EvoQuant/M3 adapters deferred until these gates are recorded.
