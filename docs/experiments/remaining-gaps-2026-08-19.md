# Remaining Gap Audit

This audit follows the registered twelve-ablation completion at commit
`e41ae73`. It distinguishes implementation work that is complete from claims
that require a new design decision or an external intervention source.

| Gap | Current state | Why it is not silently completed | Exit condition |
| --- | --- | --- | --- |
| Confirmatory update-generation and holdout rule | The public runs use the frozen exploratory `position_size: 0.2 -> 0.6` typed edit. | Choosing a new update after seeing the public rows would invalidate the confirmatory claim; the next rule must be frozen before outcomes. | Approve and commit a deterministic update generator, asset/regime labels, participation grid, and holdout split before running outcomes. |
| Causal interactive response | F2 fixture has explicit impact/recovery; Binance klines plus percentage depth provide an observational execution proxy only. | The public source does not identify replenishment, hidden liquidity, post-trade response, or strategic adaptation. | Obtain or build a versioned intervention world with an explicit matching/response mechanism, then compare it against the observational proxy without calling either universal ground truth. |
| External strategic validation | S0/S1/S2 and E7/E8 show a deterministic fixture result (`adaptive SIRR=1.0`). | The fixture is useful for mechanism tests but cannot establish behavior in a realistic competitive market. | Freeze a versioned external or independently calibrated strategic environment and reproduce the actor-positive/strategic-negative contrast without tuning a zero crossing. |
| Footprint and budget null follow-up | The ablation suite records a footprint null and a non-monotonic controlled budget curve. | These are honest current results, but one grid is insufficient to establish that footprint is unnecessary or that budget is generally non-monotonic. | Expand the predeclared controlled grid, preserve equal budgets, and report the null as a robustness boundary unless a replicated effect appears. |
| F3/M3 and LLM/EvoQuant adapters | Intentionally deferred; no core result depends on them. | Adding them now would expand scope and blur the update-fidelity estimand. | Only begin after the controlled estimand, PIVOT budget result, causal response, and strategic validation gates are reviewed. |

## Completed engineering checks

- Twelve ablation IDs are implemented and enforced by the runner.
- Three disjoint registered seed sets complete with `status=ok`, `exit_code=0`.
- Cross-run aggregation reports `valid_run_count=3` and `ablation_count=12`.
- Raw rows, empty failure ledgers, provenance, configuration hashes, and clean
  reproduction hashes are retained.
- Full local verification is `76 passed`; ruff and mypy are clean.

## Submission boundary

The repository is implementation-complete for the controlled P0-P9 harness and
its registered ablations. It is not yet a submission-grade external claim. The
next action must be a scientific decision about the confirmatory rule or an
approved external intervention source, not an unregistered module expansion.
