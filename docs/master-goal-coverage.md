# Master Goal Coverage

This matrix audits the authoritative specification in [master-goal.md](master-goal.md) against the current project state. “Documented” means the contract is written down; it does not mean the experiment or code has passed.

| Master requirement | Authoritative location | Current status | Evidence required for completion |
| --- | --- | --- | --- |
| Policy update is the scientific object | `master-goal.md` sections 1-3; design section 2 | Documented | Transition-level rows in Parquet |
| PIVOT acronym and method boundary | master sections 9-12; master plan P4 | Documented | Paired correction + active HF implementation |
| World 0/1/2 hierarchy | master section 5; design section 6 | Documented | Separate evaluator IDs and non-substitution checks |
| F0-F4 finance ladder | master section 18; design section 6 | Documented | F0/F1/F2/F4 artifacts and F3 `ground_truth=false` |
| S0/S1/S2 opponent ladder | master section 19; design section 6/P8 | Documented | Fixed/reactive/adaptive opponent logs |
| IDE/ISC/IRR/SIRR/MTR/ISR/CTI/HF cost | master section 7; master plan schemas | Documented | Metrics table with counts and intervals |
| Full `PolicyTransition` schema | master section 13; master plan schemas | Documented | Every required column present, nulls explicit |
| Generic and finance footprint components | master sections 14-15; design section 3.3 | Documented | Component-level footprint columns and tests |
| Controlled world before finance | master sections 17 and 29; plan P1-P2 | Documented | P1/P2 gate records and no finance imports |
| E1-E9 strict experiment order | master section 21; master plan phase table | Documented | Experiment run manifests in order |
| B1-B9 baseline matrix | master section 22; master plan P4 | Documented | Matched-budget baseline table |
| Twelve required ablations | master section 23; master plan ablation matrix | Documented | Ablation result rows and seed intervals |
| Statistical protocol and failed-run retention | master section 24/27; global constraints | Documented | Clean-room rerun and failure ledger |
| Seven production figures | master section 25; design section 11 | Documented | Seven images plus source tables and hashes |
| Repository and artifact structure | master sections 26-27; master plan repository contract | Documented | Files exist and scripts reproduce outputs |
| Analytic toy tests and full integration round | master section 28; plan P0/P9 | Documented | Unit tests and one complete round test |
| P0-P10 implementation order | master section 29; master plan phases | Documented | Commits and gate ledger follow order |
| First milestone | master section 30; plan P1/P2 | Not implemented | One command and five outputs |
| Second milestone | master section 31; plan P3 | Not implemented | Equal-budget global/local table |
| Third milestone | master section 32; plan P4/P5 | Not implemented | PIVOT beats B2/B3 on CTI or ISR |
| Fourth milestone | master section 33; plan P6/P7 | Not implemented | Plausible F0/F1 versus F2 response |
| Fifth milestone | master section 34; plan P9 | Not implemented | Systematic strategic effect or honest null |
| Gates A-F | master section 35; `experiments/gates.md` | Not run | Evidence record per gate |
| Non-goals and no live trading | master section 36; global constraints | Documented | Boundary audit before release |

## Current conclusion

The documentation layer is aligned with the master specification. No scientific gate is currently passed, and no empirical result is being claimed. Implementation must start at P0 and stop at each gate if the evidence is null or contradictory.
