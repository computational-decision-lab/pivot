# Experiment Protocol Matrix

This matrix is an audit aid for the manuscript and rebuttal. It records the
candidate generator, proxy, high-fidelity (HF) observation, response mechanism,
and decision structure for each evidence layer. The descriptions are taken from
the frozen protocols and the corresponding source snapshots. It separates the
fixed-panel single-decision studies from the state-carrying v3c extension.

| Environment | Candidate generation | Proxy verifier | HF verifier / paired observation | Response mechanism | Decision structure |
|---|---|---|---|---|---|
| Leduc poker | Nine alpha-mixture candidate slots, alpha in `{0, .125, ..., 1}`, including the alpha=0 incumbent copy, plus a separate no-update action; each mixes the CFR+ incumbent toward a best response to a root-specific proxy opponent. | Exact game value of each candidate against the perturbed proxy opponent. | Paired common-random-number poker rollouts: candidate against its deployment responder minus incumbent against its incumbent responder; 4,096 hands per query and primary cap 16,384 hands (two queries). | Latent equilibrator/exploiter responder; short and long are one and eight response steps (`eta=.15`). | Fixed candidate panel and one decision per root; primary PIVOT-KG versus Uniform, with IVR/LUCB and all-HF diagnostics replayed from the sealed bank. |
| Kuhn poker | Same nine alpha-mixture slots and explicit incumbent option as Leduc, with the Kuhn CFR+ incumbent and proxy opponent. | Exact game value against the root-specific perturbed proxy opponent. | The same paired common-random-number rollout difference, 4,096 hands per query and primary two-query cap. | The same latent equilibrator/exploiter response family; short/long are one/eight response steps. | Fixed candidate panel and one decision per root; the v2b cohort changes HF precision relative to the retained 256-hand boundary cohort. |
| Melting Pot | Eight fixed specialist-mixture candidates with probabilities `{.125, .225, .325, .425, .575, .675, .775, .875}` and incumbent probability `.5` on the repeated stag-hunt substrate. | Frozen-responder proxy block with paired stratified native rollouts. | Sealed selection bank of paired stratified native rollout estimates for each candidate; held-out audit A/B blocks are generated after selection inputs are frozen and are not visible to selectors. | Latent cooperative/competitive responder selected by seed parity; short/long use 4/32 adaptation episodes. | Fixed candidate panel and one decision per root; primary HF episode cap 192, with one/two/ four-query caps and offline selector replay from the sealed bank. |
| HighwayEnv | Eight scheduled candidates crossing four actions (`SLOWER`, `LANE_LEFT`, `LANE_RIGHT`, `FASTER`) with early/late half-horizon windows, plus the incumbent. | Observer-mode proxy score corrected by a Gaussian model fit on a disjoint calibration cohort. | Paired actor rollouts of candidate minus incumbent under the same root seed; each selector gets four sequential logical queries at the primary budget, with posterior-mean terminal selection. | Reactive traffic simulator response after deployment; this is a single-decision reactive-control study. | Fixed candidate panel and one decision per root; saved-truth offline replay, fresh simulator development runs with frozen calibration, and an independent calibration/test cohort are distinct evidence layers. The simulator cohorts retain IVR and Top-Proxy null/mixed results. Formal LUCB is infeasible at `K=8, B=4`. |
| PerformativeWorld v3c | A fixed proposer generates four candidates from the current incumbent at each round. The first-round panel matches across methods; later panels can differ after method-specific deployments change the incumbent. | Contextual correction model over proxy deltas, pressure, incumbent/candidate features, response strength, interactions, and round fraction. | Two sequential method-specific paired HF deployment observations per round, conditioned with the positive observation variance; independent evaluation and deployment-return calls occur after the decision. | Controlled response world carries pressure/state and incumbent across five rounds; response strengths are `0.35/0.60/0.85`. | State-carrying five-round loop, 120 fresh roots per regime, shared proposer function/seeds/operator/terminal rule/budget, no sentinel, and calibration-model reset each round (`cross_round_update=none`). This is a bounded synthetic extension, not the submitted static Algorithm 1. |

## Measurement notes

- A paired HF observation compares a candidate and the incumbent under the
  same root, randomness, and opponent or environment initialization. The exact
  rollout count or native episode accounting is environment-specific as shown
  above.
- In the poker studies, `short` and `long` are response steps. In Melting Pot,
  the corresponding response-horizon labels are adaptation episodes. They are
  not interchangeable cost units.
- The designated allocation interaction is the paired long-minus-short
  contrast `(ISR_Uniform - ISR_PIVOT)_long - (ISR_Uniform - ISR_PIVOT)_short`.
  The long-response allocation gains are reported separately. The normalized
  gain divides a root-level allocation gain by that root's deployment-delta
  range.
- Leduc, Kuhn, Melting Pot, and HighwayEnv compare selectors on a fixed panel
  for a single decision. The v3c study instead measures a complete controlled
  trajectory whose later candidate panels are method-induced.

## Source anchors

- Poker: `protocols/openspiel_v2b_leduc_confirm.json`,
  `protocols/openspiel_v2b_kuhn_confirm.json`, and
  `code/openspiel_v2/openspiel_v2_panel.py` in the frozen response-world
  source snapshot.
- Melting Pot: `protocols/melting_v4_confirm.draft.json`,
  `code/melting_native_v4_hetero.py`, and the sealed v4 confirmation bank.
- HighwayEnv: `HIGHWAY_MATCHED_PHYSICAL_PREREGISTRATION_2026-11-09.md`,
  `HIGHWAY_INDEPENDENT_PHYSICAL_PREREGISTRATION_2026-11-09.md`, and
  `audit_inputs/highway_redesign/source/experiments/revision/`.
- PerformativeWorld: `P1_LOOP_V3C_PROTOCOL_2026-11-10.md`,
  `P1_LOOP_V3C_PROTOCOL_CLARIFICATION_2026-11-10.md`, and
  `results/rebuttal/20261110-p1-loop-v3c-confirmation/protocol.json`.
