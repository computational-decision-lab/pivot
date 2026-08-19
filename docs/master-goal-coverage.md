# Master Goal Coverage

This matrix audits the authoritative specification in [master-goal.md](master-goal.md) against the current project state. “Documented” means the contract is written down; it does not mean the experiment or code has passed.

| Master requirement | Authoritative location | Current status | Evidence required for completion |
| --- | --- | --- | --- |
| Policy update is the scientific object | `master-goal.md` sections 1-3; design section 2 | Implemented | Transition-level rows in Parquet; formal runs pending |
| PIVOT acronym and method boundary | master sections 9-12; master plan P4 | Implemented | Paired correction + active HF implementation |
| World 0/1/2 hierarchy | master section 5; design section 6 | Implemented through F4 | Separate evaluator IDs; F3 remains deferred |
| F0-F4 finance ladder | master section 18; design section 6 | Partial | F0/F1/F2/F4 fixtures; F3 `ground_truth=false` adapter deferred |
| S0/S1/S2 opponent ladder | master section 19; design section 6/P8 | Implemented as smoke | Fixed/reactive/adaptive opponent logs; formal sweep pending |
| IDE/ISC/IRR/SIRR/MTR/ISR/CTI/HF cost | master section 7; master plan schemas | Implemented and tested | Metrics table with counts and intervals; independent aggregation pending |
| Full `PolicyTransition` schema | master section 13; master plan schemas | Implemented and tested | Every required column present, nulls explicit |
| Generic and finance footprint components | master sections 14-15; design section 3.3 | Implemented and tested | Component-level footprint columns and tests |
| Controlled world before finance | master sections 17 and 29; plan P1-P2 | Implemented | P1/P2 harnesses and import boundary tests |
| E1-E9 strict experiment order | master section 21; master plan phase table | Harnesses implemented | Smoke manifests exist; registered gate runs pending |
| B1-B9 baseline matrix | master section 22; master plan P4 | Implemented | Matched-budget baseline table; independent intervals pending |
| Twelve required ablations | master section 23; master plan ablation matrix | Planned, not run | Ablation result rows and seed intervals |
| Statistical protocol and failed-run retention | master section 24/27; global constraints | Partially implemented | Clean-room rerun and failure ledger; multi-run aggregation pending |
| Seven production figures | master section 25; design section 11 | Pipeline implemented | Seven images plus source tables and hashes from complete input bundle |
| Repository and artifact structure | master sections 26-27; master plan repository contract | Implemented | Files exist and scripts reproduce outputs |
| Analytic toy tests and full integration round | master section 28; plan P0/P9 | Implemented and tested | Unit tests and one complete round test |
| P0-P10 implementation order | master section 29; master plan phases | P0-P9 implemented; P10 deferred | Commits and gate ledger follow order |
| First milestone | master section 30; plan P1/P2 | Implemented; registered run completed | `run_sweep.py`, 240-row Parquet/JSONL, five diagnostics |
| Second milestone | master section 31; plan P3 | Implemented; matched-budget smoke completed | `e4_global_vs_local.py`, disjoint IDs and budget ledger |
| Third milestone | master section 32; plan P4/P5 | Implemented; preliminary positive, Gate D pending | `e5_budget_frontier.py`; independent paired runs required |
| Fourth milestone | master section 33; plan P6/P7 | Implemented as virtual fixture; calibration pending | F0/F1/F2 participation sweep |
| Fifth milestone | master section 34; plan P9 | Implemented as smoke; formal gate pending | E7/E8/E9 strategic and closed-loop logs |
| Gates A-F | master section 35; `experiments/gates.md` | Not run | Evidence record per gate |
| Non-goals and no live trading | master section 36; global constraints | Documented | Boundary audit before release |

## Current conclusion

The documentation and implementation layers are aligned with the master
specification. No scientific gate is currently passed, and no smoke result is
being presented as a submission claim. P10 remains intentionally deferred;
the next research blocker is independent multi-run aggregation, paired
intervals, and calibration of PIVOT/finance fixtures.
