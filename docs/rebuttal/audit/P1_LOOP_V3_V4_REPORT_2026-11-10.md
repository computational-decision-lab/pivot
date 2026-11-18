# Multi-round loop v3/v4 audit report

Date: 2026-11-10

This report supersedes the earlier 30-root loop-v1 summary for the new
contextual calibration studies. The experiments use the controlled
`PerformativeWorld` only; they are not physical HighwayEnv runs and do not
instantiate a learned proposer.

## v3 protocol and audit

`P1-LOOP-V3-2026-11-10` uses 240 calibration roots (`510000`--`510239`),
eight paired HF rollouts per candidate context, five rounds, four candidates,
three response strengths (`0.35`, `0.60`, `0.85`), a common sentinel plus one
method-specific query per round, and 120 confirmation roots per response
strength (`900000`, `910000`, `920000` blocks). PIVOT-KG-Loop-v3 uses the
contextual Bayesian correction model with persistent cross-round rank-one
updates; IVR-Online and Top-Proxy-Online use the same model, seeds, budget and
terminal rule with their named allocation rules.

The confirmation ledger has SHA-256
`1d24e5c6ff84989c0d4bc5a94295517ab4ac30048ca6f3228e80e0ce99fe4b9a`.
The independent analysis reports `protocol_audit_checks.all_pass = true`,
including calibration-rollout replay, root disjointness, shared deployment and
evaluation seeds, query-budget equality, sentinel order, and ISR recomputation.

| Response | PIVOT CISR | IVR CISR | Top-Proxy CISR | IVR-PIVOT | Top-Proxy-PIVOT |
|---|---:|---:|---:|---:|---:|
| Low | 0.0233 | 0.0883 | 0.0860 | `0.0649 [0.0528, 0.0772]` | `0.0626 [0.0443, 0.0818]` |
| Medium | 0.3536 | 0.8529 | 0.1569 | `0.4992 [0.3397, 0.6351]` | `-0.1967 [-0.3467, -0.0744]` |
| High | 1.1983 | 1.1267 | 0.1578 | `-0.0716 [-0.6574, 0.4523]` | `-1.0405 [-1.5848, -0.5612]` |

Intervals are paired-root bootstrap 95% intervals for baseline minus PIVOT;
positive favors PIVOT. Holm-adjusted sign-flip p-values are below `0.001` for
both low contrasts and for medium versus IVR, `0.0009` for medium versus
Top-Proxy, `0.8052` for high versus IVR, and below `0.001` for high versus
Top-Proxy. The high-response PIVOT utility mean is `0.2466`, versus `0.7742`
for IVR and `0.7465` for Top-Proxy.

Thus v3 supports a bounded low-response result, not superiority across all
response regimes or a general multi-round self-improvement claim.

## v4 exploratory development

The first v4 development run tested a reset-per-round contextual model with a
common sentinel and matched acquisition-only baselines. Its development
protocol was amended before the second development run (A1) to move calibration
roots to `520000`--`520239`, cover pressure `[0.05, 2.20]`, and remove a
round feature constant in calibration. The amended run's audit also passed, but
it lost to Top-Proxy in all three regimes and was not promoted to confirmation.
The amended output is retained at
`results/rebuttal/20261110-p1-loop-v4a-development/` for negative evidence.

The v4/v4a development outputs must not be pooled with v3 confirmation or used
to claim superiority. They document why additional tuning was stopped before
confirmation rather than selecting a favorable subset of regimes.

## Reproduction files

- v3 runner: `docs/rebuttal/audit/run_multiround_loop_v3.py`
- v3 analyzer: `docs/rebuttal/audit/analyze_multiround_loop_v3.py`
- v3 protocol: `docs/rebuttal/audit/P1_LOOP_V3_PREREGISTRATION_2026-11-10.md`
- v3 confirmation: `results/rebuttal/20261110-p1-loop-v3-confirmation/`
- v3 confirmation analysis: `results/rebuttal/20261110-p1-loop-v3-confirmation/paired_analysis.json`
- v4 exploratory protocol and code: `P1_LOOP_V4_PREREGISTRATION_2026-11-10.md`, `run_multiround_loop_v4.py`, and `analyze_multiround_loop_v4.py`
