# Master Goal Coverage

This matrix audits the authoritative specification in [master-goal.md](master-goal.md) against the current project state. “Documented” means the contract is written down; it does not mean the experiment or code has passed.

| Master requirement | Authoritative location | Current status | Evidence required for completion |
| --- | --- | --- | --- |
| Policy update is the scientific object | `master-goal.md` sections 1-3; design section 2 | Implemented and exercised | Registered transition-level rows and public paired rows; external confirmation pending |
| PIVOT acronym and method boundary | master sections 9-12; master plan P4 | Implemented | Paired correction + active HF implementation |
| World 0/1/2 hierarchy | master section 5; design section 6 | Implemented through F4 | Separate evaluator IDs; F3 remains deferred |
| F0-F4 finance ladder | master section 18; design section 6 | Partial | F0/F1/F2/F4 fixtures; F3 `ground_truth=false` adapter deferred |
| S0/S1/S2 opponent ladder | master section 19; design section 6/P8 | Implemented; registered fixture sweep completed | External strategic environment validation pending |
| IDE/ISC/IRR/SIRR/MTR/ISR/CTI/HF cost | master section 7; master plan schemas | Implemented and tested | Metrics table with counts and registered intervals |
| Full `PolicyTransition` schema | master section 13; master plan schemas | Implemented and tested | Every required column present, nulls explicit |
| Generic and finance footprint components | master sections 14-15; design section 3.3 | Implemented and tested | Component-level footprint columns and tests |
| Controlled world before finance | master sections 17 and 29; plan P1-P2 | Implemented | P1/P2 harnesses and import boundary tests |
| E1-E9 strict experiment order | master section 21; master plan phase table | Harnesses and registered runs implemented | E1-E9 manifests exist; external-validity review pending |
| B1-B9 baseline matrix | master section 22; master plan P4 | Implemented | Registered matched-budget fixture table and independent intervals; external confirmation pending |
| Twelve required ablations | master section 23; master plan ablation matrix | Implemented; three clean registered runs | `docs/experiments/ablation-evidence-2026-08-19.md`, raw rows, failure ledgers, and cross-run bootstrap intervals; external validity pending |
| Statistical protocol and failed-run retention | master section 24/27; global constraints | Implemented and clean-room verified for registered P2-E9 | Failure ledger and independent aggregation recorded; external review pending |
| Seven production figures | master section 25; design section 11 | Pipeline implemented and clean-room validated | Seven images plus source tables and validation hash from a fresh complete input bundle |
| Repository and artifact structure | master sections 26-27; master plan repository contract | Implemented | Files exist and scripts reproduce outputs |
| Analytic toy tests and full integration round | master section 28; plan P0/P9 | Implemented and tested | Unit tests and one complete round test |
| P0-P10 implementation order | master section 29; master plan phases | P0-P9 implemented; P10 deferred | Commits and gate ledger follow order |
| First milestone | master section 30; plan P1/P2 | Implemented; registered run completed | `run_sweep.py`, 240-row Parquet/JSONL, five diagnostics |
| Second milestone | master section 31; plan P3 | Implemented; matched-budget registered evidence completed | `e4_global_vs_local.py`, disjoint IDs and budget ledger |
| Third milestone | master section 32; plan P4/P5 | Implemented; fixture Gate D pass | `e5_budget_frontier.py`; external-validity review required |
| Fourth milestone | master section 33; plan P6/P7 | Fixture and frozen 3-asset/4-block observational audit implemented | Causal response validation and confirmatory update-generation rule |
| Fifth milestone | master section 34; plan P9 | Implemented; fixture Gate F pass | E7/E8/E9 registered strategic and closed-loop logs |
| Gates A-F | master section 35; `experiments/gates.md` | Fixture-level passes recorded; paper promotion pending | Registered evidence record and external-validity review |
| Non-goals and no live trading | master section 36; global constraints | Documented | Boundary audit before release |

## Current conclusion

The documentation and implementation layers are aligned with the master
specification. Fixture-level registered passes, a complete controlled
twelve-ablation suite, and a checksum-bound public
finance audit across three assets and four calendar blocks are recorded, but
none is being presented as a final submission claim. The public percentage-depth
result is an observational execution proxy, not causal actor-world ground truth;
the expansion observed no reversal at any predeclared participation. P10 remains
intentionally deferred; the next blockers are a confirmatory update-generation
rule, causal response robustness, and external strategic validation.
