# Implementation Status

Updated 2026-08-21 after the public-data calibration audit, the F2 fill-only
impact semantics amendment, the clean multi-asset expansion run, and the
registered twelve-ablation suite. The V6 analytic theory checks are now
registered under `results/theory/e10-theory-empirical`. The IMPROVE-X V5 platform vertical slice is
also tracked in `docs/improve-x-v5-status.md`.

Paper delivery update 2026-08-21: the anonymous ICLR package now has nine
main-text pages and fourteen total pages, with the bibliography explicitly
excluded from the main-text limit at `refs:start`. The styled hash-indexed
snapshot, supplementary archive, table generator, and portable PDF reports are
tracked under `paper/`. Local machine checks pass; the external-validity and
author-side gates below remain open.

## Implemented

| Phase | Evidence in this checkout | Status |
| --- | --- | --- |
| P0 | frozen policy/transition schemas, paired evaluation, metrics, bootstrap CI, append-only JSONL + Parquet manifest | Implemented and tested |
| P1 | controlled performative observer/actor world, synthetic and RL-style update operators | Implemented and tested |
| P2 | registered response x footprint x optimization x seed sweep; E1/E2/E3 commands; five first-milestone artifacts | Implemented; smoke/registered run completed |
| P3 | matched-budget global value vs differential transfer models and E4 ledger | Implemented; clean run completed |
| P4/P5 | acquisition baselines, PIVOT heuristic, round harness, E5 frontier, figure validation pipeline | Implemented; clean run completed |
| P6/P7 | virtual F0/F1 replay, corrected fill-only F2 fixture, official-public-data acquisition, parser, calibration, and depth-aware execution proxy | Implemented and tested; single-asset and frozen multi-asset public audits completed |
| P8/P9 | S0/S1/S2 strategic wrapper, E7/E8/E9 scripts, closed-loop query ledger | Implemented; corrected three-run registered fixture evidence completed |
| Registered evidence | frozen P2-E9 registries, isolated run manifests, paired aggregation, failed-run retention | Implemented; fixture-level A-F summaries generated and clean-room rerun verified |
| Public finance audit | checksum-bound BTCUSDT USD-M one-minute kline and percentage-depth sessions, paired F0/F1/F2-depth sweep | Implemented; observational external audit partial |
| Clean-room reproduction | fresh P2/E4-E9 registries, fresh public cache, independent aggregation, seven-figure validation | Verified at commit `0ba4117`; artifact record in `clean-room-evidence-2026-08-19.md` |
| Public expansion | frozen BTC/ETH/BNB quarterly-start grid, 12 sessions, subconfig contract validation, pooled/asset/holdout aggregation | Complete observational run; no reversal observed; causal validation open |
| Twelve ablations | paired/unpaired, transition/global-value, footprint, acquisition, update size, response, F1/F2, competitors, response-model count, candidate count, HF budget | Complete controlled suite; three clean registered runs; external validity and null follow-up open |
| V6 analytic theory checks | constructive Global Fidelity Blindness adjacent-swap family; exact Response-Footprint Sensitivity bound; row-level data, figures, provenance, and SHA-256 manifest | Registered full grid; all theorem gates pass; claim boundary is analytic, not causal market evidence |
| IMPROVE-X V5 platform | operator batches, multi-round trajectories, ImprovementBench v1, three-world rows, failure taxonomy, sign/ranking/explanation tasks, seeded evolutionary operator, PIVOT-X query scores | Implemented and locally tested; confirmatory benchmark and external response validation remain open |
| ICLR spotlight narrative upgrade | transition-first abstract/introduction, four contribution framing, decision-preservation proposition, active-learning distinction, controlled value-vs-improvement E4 contrast, semantic Figure 1/2/4, portable snapshot paths | Implemented; local PDF/supplement gates pass, scientific/manual gates remain open |

## Scientific gates

The implementation does not silently promote smoke results to paper claims.

- Gate A/B: the registered fixture-level summaries pass the predeclared
  three-run criteria; external scientific promotion remains pending.
- Gate C: the registered matched-budget summary passes the ISC criterion, while
  ISR is tied in this fixture; the claim is therefore limited to the stated
  estimand evidence.
- Gate D: the registered E5 summary passes at budget 1 on paired ISR against
  Random and Top Proxy; this remains a controlled-fixture result, not a broad
  superiority claim.
- Gate E: corrected F2 equals F1 at zero participation and changes delta in the
  registered fixture. The expanded public BTC/ETH/BNB grid has a negative
  pooled depth mechanical effect but `0/7` reversals at primary participation,
  including `0/5` on the frozen holdout. This is an observational null for the
  reversal claim, not identified endogenous response, so paper promotion remains
  pending.
- Gate F: the corrected registered E7/E8 fixture summary has positive E8 actor
  improvement before adaptation and E7/E8 SIRR of `1.0`. External strategic
  validation is still required.
- Gate G: the V6 GFB construction covers all three epsilon targets, with
  operator IRR `1.0` at minimum global MAE `1.91e-6`; the RFS construction
  holds its bound on all 180 rows with a tight ratio of one. This closes the
  analytic theorem-check gate only; it does not close causal or strategic
  external-validity gates. The final paper now defines
  `IF(V,A;L)=E_{Q_A}[L(Delta_V,Delta_*)]`, and the artifact records the
  adjacent-swap `Q_A` support plus absolute-delta and sign-error IF losses.

## Remaining before a submission-grade result

1. Freeze a confirmatory update-generation rule and add predeclared
   volatility/liquidity labels; the current typed update remains exploratory.
2. Add a causal interactive response source or multiple intervention-model
   robustness audit. Percentage depth does not identify replenishment,
   post-trade response, hidden liquidity, or impact recovery.
3. Validate strategic reversal outside the deterministic fixture and design
   confirmatory follow-ups for the footprint and non-monotonic budget nulls.
4. Keep LLM/EvoQuant/M3 adapters deferred until the core claims survive those
   checks.
5. Re-run the IMPROVE-X benchmark and trajectory commands from a clean
   environment before treating their outputs as release artifacts.

The complete gap ledger is in
`docs/experiments/remaining-gaps-2026-08-19.md`.
