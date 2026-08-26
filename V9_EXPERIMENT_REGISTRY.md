# PIVOT V9 Experiment Registry

The machine-readable registry is `research/experiment_registry_v9.yaml`.

| Experiment | Independent unit | Main question | Confirmatory scale |
| --- | --- | --- | --- |
| E2C | seed | Does operator shift predict local improvement-fidelity loss? | 3 environments x 3 operators x 10 shifts x 30 seeds |
| E3C | seed/trajectory | Does transition-aware validation alter closed-loop improvement? | 2 new worlds, 40 rounds x 8 candidates x 10 methods x 30 seeds; frozen V7 MPE2 null panel |
| E4C | held-out split | Do differential evaluators transfer OOD? | trajectory/environment/operator/response splits |
| E5C | seed/candidate set | Which acquisition rule is efficient at matched HF cost? | K in {4,8,16}, fixed budgets, nine methods |
| E7C | opponent seed | Does strategic adaptation add reversal? | five opponent mechanisms, held-out seeds |

The registry is intentionally narrower than the surrounding project history:
M3, LLM candidate generation, live market data, and market-ground-truth claims
are not required for V9 completion.
