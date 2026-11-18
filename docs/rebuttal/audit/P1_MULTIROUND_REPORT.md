# SUPERSEDED: P1 controlled multi-round Algorithm 1 replay

The 30-root v1 result below is retained as historical evidence. The expanded
v3 confirmation and v4 development results are reported in
`P1_LOOP_V3_V4_REPORT_2026-11-10.md`.

Date: 2026-11-09

This analysis answers the request to place Algorithm 1 inside a repeated
proposer/deployment/response loop. It is a new, explicitly controlled protocol,
not a claim about general self-improving agents. The frozen Gaussian posterior,
exact KG acquisition, paired query semantics, and posterior-mean terminal rule
are unchanged. An outer wrapper generates four candidates from the incumbent,
allocates two paired HF queries, permits incumbent retention, deploys the
selected policy, and carries the PerformativeWorld actor final state into the
next round. Independent evaluation seeds are used for scoring.

## Protocol audit

- 30 independent roots and 5 rounds per root.
- Methods: PIVOT-KG, Uniform-Random, IVR, Top-Proxy, Proxy-Only, and an
  all-HF reference.
- Every PIVOT-KG round uses exactly two queries; zero-query and all-HF records
  are explicit.
- The response state changes and is hashed before and after every deployment.
- Evaluation, proposer, environment, query, and deployment seeds are recorded
  separately.
- Formal LUCB is marked infeasible because four candidate initial pulls exceed
  the per-round budget of two.
- All audit checks pass. No physical simulator is called.

## Results

| Method | Cumulative ISR | Cumulative utility | Harmful adoption | Retention |
|---|---:|---:|---:|---:|
| PIVOT-KG | `2.602` | `0.052` | `0.280` | `0.333` |
| Uniform-Random | `2.170` | `0.454` | `0.360` | `0.113` |
| IVR | `1.986` | `0.497` | `0.433` | `0` |
| Top-Proxy | `1.999` | `0.447` | `0.440` | `0` |
| All-HF reference | `0.0036` | `0.753` | `0.040` | `0.413` |

Paired root-level cumulative-ISR contrasts are defined as baseline minus
PIVOT-KG, so positive would favor PIVOT-KG. Uniform, IVR, and Top-Proxy give
`-0.432 [-1.153, 0.265]`, `-0.616 [-1.287, 0.057]`, and
`-0.603 [-1.320, 0.094]`, respectively. Two-sided sign-flip p-values are
`0.254`, `0.088`, and `0.109`; Holm-adjusted values are `0.265` for each.
The controlled replay therefore does not support a claim of lower cumulative
multi-round regret for PIVOT-KG than these targeted baselines.

The complete ledger, hashes, per-round bootstrap summaries, and paired analysis
are in `results/rebuttal/20261109-p1-loop-v1/`. The result should be presented
as a scope-limiting controlled evaluation and not as evidence of convergence,
general adaptive-world safety, or KG-specific superiority.

Re-running the same loop from the normalized Highway source under
`docs/rebuttal/audit_inputs/highway_redesign/source/` produced an identical
ledger SHA256 in `results/rebuttal/20261109-p1-loop-input-replay-v1/`; only the
bootstrap draw count for that path check was reduced.
