# PIVOT V9 Results Report

This report is generated from per-run scientific decision artifacts.

| Run | Status | Powered | Design valid | Reason |
| --- | --- | --- | --- | --- |
| `e2c-confirmatory` | `HYPOTHESIS_SUPPORTED` | True | True | operator shift increases operator-relative absolute improvement error |
| `e2c-development` | `UNDERPOWERED` | False | True | profile does not meet the preregistered 30-seed confirmatory rule |
| `e3c-confirmatory` | `HYPOTHESIS_SUPPORTED` | True | True | PIVOT-VOI reduces CISR versus Proxy Only in the registered aggregate |
| `e4c-confirmatory` | `HYPOTHESIS_NOT_SUPPORTED` | True | True | powered OOD comparison does not support a differential ISC gain |
| `e5c-confirmatory` | `HYPOTHESIS_SUPPORTED` | True | True | PIVOT-VOI has lower CISR than Proxy Only at the registered budget |
| `e7c-confirmatory` | `HYPOTHESIS_SUPPORTED` | True | True | independently adaptive opponent modes reduce focal update value |

## Claim boundary

Numbers are interpreted only within the registered environments, operator families, splits, budgets, and opponent mechanisms. Underpowered and null decisions are retained as outcomes. All-HF is an oracle reference and is not treated as a comparable acquisition method.

## Reversal and efficiency reading

E2C reports operator-relative improvement fidelity and reversal diagnostics. E3C reports closed-loop selection outcomes. E4C reports matched-evidence OOD/calibration diagnostics. E5C reports fixed-budget efficiency. E7C reports strategic response effects. No row supports a universal PIVOT-superiority claim.
