# V3C Protocol Clarification

This clarification accompanies, but does not replace or modify, the hash-bound
v3c protocol at `P1_LOOP_V3C_PROTOCOL_2026-11-10.md`. The original protocol and
source hashes remain unchanged.

The phrase "share the candidate panel" in the original protocol is imprecise.
The methods share the same proposer function and random-seed schedule. Their
first-round panels are identical because they start from the same incumbent.
After the first deployment, each method's panel is generated relative to its
own incumbent and may diverge as trajectories evolve. The methods share the
calibration data, response operator, terminal rule, seed schedule, and total
HF budget; their acquisition rules induce the subsequent differences.

Relative to v3b, v3c changes both the sentinel-query allocation and the
cross-round correction-model update policy. Therefore, the v3c comparison is a
protocol-level extension rather than an isolated sentinel ablation. The
`Online` suffix in `IVR-Online-v3c` and `Top-Proxy-Online-v3c` denotes
sequential within-round posterior conditioning after each HF query. It does not
denote persistent cross-round calibration-regression learning. The calibrated
correction model is reset at the start of every round
(`cross_round_update=none`); the deployment world state and incumbent carry
forward.

The v3c confirmation uses fresh roots conditional on a protocol selected after
earlier exploratory variants. Its six-test Holm correction covers the six
prespecified within-v3c regime-by-baseline contrasts only; it does not adjust
for the preceding v1--v4a variant search. This post hoc clarification changes
no data, metric, method, or outcome.
