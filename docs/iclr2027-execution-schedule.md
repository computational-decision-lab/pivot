# ICLR 2027 Execution Schedule

**Planning basis:** dates supplied in the research discussion: abstract 2026-09-18 AOE, full paper 2026-09-25 AOE, nine-page main text. Recheck the official venue pages before external submission.

## Critical Path

| Window | Deliverable | Stop condition |
| --- | --- | --- |
| Aug 19-22 | P0 contracts, schemas, paired evaluator, metrics, logging | No environment work until schema/unit tests pass |
| Aug 23-26 | P1 controlled world and first transition dataset command | No finance, LLM, or multi-agent imports |
| Aug 27-30 | P2 E1/E2, Gates A/B | Stop/narrow if reversal is pathological or noise-only |
| Aug 31-Sep 3 | P3 E4, Gate C | Narrow framing if global evaluator fully solves local updates |
| Sep 4-8 | P4/P5 PIVOT and E5 budget frontier, Gate D | Do not enter finance without matched-budget evidence |
| Sep 9-12 | P6/P7 F0/F1/F2 and participation sweep, Gate E | Keep finance secondary if only implausible footprints reverse |
| Sep 13 | P8 strategic environment freeze | One focal self-improver; no tuned zero crossing |
| Sep 14-15 | P9 E7/E8, Gate F, then E9 closed loop | Run E9 only after E1-E8; preserve equal HF budgets |
| Sep 16-17 | Freeze title, abstract, contributions, core figures | No new modules |
| Sep 18 AOE | Abstract deadline basis | Abstract and author metadata frozen |
| Sep 19-22 | Main-paper writing, theory audit, clean reruns | Only result corrections and required ablations |
| Sep 23 | Nine-page PDF freeze candidate | No new experiments unless a claim is invalidated |
| Sep 24 | Independent reproducibility and anonymity audit | Block on missing evidence or leaked metadata |
| Sep 25 AOE | Full-paper deadline basis | Submit only the verified artifact set |

## Scope Protection

P10 extensions (LLM/EvoQuant and F3 world models) are outside the submission critical path. They enter only if Gates A-D pass early and the main paper already has reproducible controlled evidence. E9 remains a required project milestone, but it must not displace reproducible E1/E2/E4/E5 evidence from the submission; if it cannot be completed before the paper freeze, record it as post-submission work rather than silently dropping it.

## Daily Evidence Rule

Each day closes with run IDs, config hashes, seed lists, gate status, failed runs, and the next falsifiable question. A passing test suite is implementation evidence, not scientific gate evidence.
