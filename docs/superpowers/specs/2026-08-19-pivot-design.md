# PIVOT Research Design

**Status:** Frozen research design for implementation planning
**Date:** 2026-08-19
**Target venue:** ICLR 2027
**Planning deadlines supplied for this freeze:** abstract 2026-09-18 AOE; full paper 2026-09-25 AOE; main text limit 9 pages
**Working title:** *When Better Gets Worse: Improvement Fidelity for Self-Improving Agents in Adaptive Worlds*
**Method name:** PIVOT — *Paired Interventional Verification of Optimization Transitions*

## 1. Design lock

The paper studies whether a self-improving agent's update remains an improvement after deployment changes the world. The statistical object is the transition

```text
pi_t -> pi_{t+1}
```

and not an isolated state, trajectory, or policy. The paper is intentionally limited to one phenomenon, one method, one theory program, and two broad evidence levels: controlled performative experiments first, then a finance testbed.

| Item | Frozen choice |
| --- | --- |
| Core phenomenon | Improvement Reversal |
| Core estimand | Improvement Fidelity of a policy update |
| Core metrics | IDE, ISC, IRR, MTR, ISR, CTI, and high-fidelity cost |
| Core method | PIVOT: paired differential modeling plus budgeted interventional evaluation |
| World ladder | Observer -> Actor -> Strategic |
| Main theory | Improvement error scales with update footprint x environment sensitivity |
| Finance result | Proxy improvement -> actor improvement -> strategic reversal |
| Primary agent | One focal self-improving agent |
| Deferred extensions | LLM/EvoQuant and M3, only after controlled gates pass |

## 2. Central claim

An evaluator can be correct about the wrong world. A fixed, external, correctly implemented verifier can accept an update because it evaluates dynamics that do not include the update's mechanical footprint or the strategic response of other agents.

The paper therefore asks:

```text
Does sign(Delta_V) equal sign(Delta_*)?
```

where `Delta_V` is the proxy-world improvement and `Delta_*` is the improvement after deployment-induced dynamics are applied.

The intended message is:

> A self-improving agent should be judged by whether its update remains better in the world, and among the agents, that its deployment causes to exist.

## 3. Formal setup

Let `pi` be the incumbent policy and `pi'` a candidate update. A proxy evaluator reports:

```text
Delta_V(pi, pi') = J_V(pi') - J_V(pi)
```

When deployment changes the environment, the true deployed value is:

```text
J_*(pi) = J(pi; M[pi])
Delta_*(pi, pi') = J(pi'; M[pi']) - J(pi; M[pi])
```

The canonical failure event is **Improvement Reversal**:

```text
Delta_V > 0 and Delta_* < 0
```

### 3.1 Response decomposition

The decomposition is explanatory and experimental; it is not a claim that all domains admit an exact additive causal decomposition.

```text
Delta_direct     = J(pi'; M) - J(pi; M)
Delta_actor      = J(pi'; M[pi']) - J(pi; M[pi])
Delta_mechanical = Delta_actor - Delta_direct
Delta_strategic  = J_i(pi_i', BR_-i(pi_i'))
                   - J_i(pi_i, BR_-i(pi_i))
Delta_competition = Delta_strategic - Delta_actor
```

The empirical ladder is:

```text
Observer / replay -> Actor / endogenous footprint -> Strategic / adaptive opponents
```

### 3.2 Metrics

For update pairs sampled from an improvement operator `A`:

```text
IDE  = E[ |Delta_V - Delta_*| ]
ISC  = P[ sign(Delta_V) = sign(Delta_*) ]
IRR  = P[ Delta_* < 0 | Delta_V > 0 ]
SIRR = P[ Delta_strategic < 0 | Delta_actor > 0 ]
MTR  = Delta_* / Delta_V, only when |Delta_V| > tau_mtr
ISR_t = max_j Delta_{*,t,j} - Delta_{*,t,j_hat}
CTI_T = sum_t Delta_{*,t}
```

All sign metrics must specify a zero tolerance `tau_sign`; values in `[-tau_sign, tau_sign]` are treated as ties and reported separately. Every result must also record high-fidelity transition count, rollout count, environment-step count, simulator-call count, and compute cost when meaningful. No PIVOT result is valid without its high-fidelity budget.

### 3.3 Update footprint

The footprint is a feature of the transition, not a replacement for the policy:

```text
z_Delta = [ KL(pi' || pi), occupancy shift, action shift,
            entropy shift, support expansion ]
```

Finance-specific features are:

```text
[ turnover, position size, participation, holding time, urgency,
  rebalance frequency, order-size distribution, aggressive/passive ratio,
  liquidity consumption, concentration, inventory duration,
  spread-crossing frequency ]
```

The implementation must expose `compute_update_footprint(pi, pi_prime, evaluation_states)` and preserve component columns rather than collapsing them to one scalar. It must preserve the distinction between strategy frequency and simulation frequency. The first implementation uses synthetic edits and one-component typed edits so footprint can be controlled and audited.

## 4. Theory targets

### Proposition 1: value fidelity is sufficient

If a proxy satisfies

```text
sup_pi |J_V(pi) - J_*(pi)| <= epsilon
```

then every update satisfies

```text
|Delta_V - Delta_*| <= 2 epsilon
```

This establishes that global value fidelity is sufficient, but stronger than necessary for self-improvement.

### Proposition 2: differential fidelity can hold without global fidelity

If

```text
J_V(pi) = J_*(pi) + C
```

for a policy-independent constant `C`, global value error can be large while all paired improvement deltas are exact. This motivates estimating the differential object directly.

### Theorem target: footprint-sensitivity bound

Under explicitly stated reward/value regularity assumptions, and an environment response metric satisfying

```text
D(M[pi'], M[pi]) <= L_M d(pi', pi),
```

the implementation will attempt to establish a bound of the form

```text
|Delta_actor - Delta_direct| <= C L_M d(pi', pi)
```

where `C` depends on the chosen discount and value bounds. This is a target to prove or falsify in the controlled environment, not a result to assume in the paper.

### Strategic extension target

Define opponent sensitivity:

```text
S_-i = D(BR_-i(pi_i'), BR_-i(pi_i))
       / (d(pi_i', pi_i) + epsilon)
```

The exploratory extension is an error relationship involving both `L_M d` and `L_S S_-i d`. If competition violates a useful Lipschitz regime, that result is reported as a boundary condition rather than hidden by an over-strong theorem.

## 5. PIVOT method

Each round has an incumbent and `K` candidates. PIVOT uses a cheap evaluator to produce proxy deltas and footprints, then spends high-fidelity queries where the update ordering or sign is most likely to change.

```text
incumbent pi_t
  -> improvement operator A
  -> candidates pi'_1 ... pi'_K
  -> cheap evaluator V0: Delta_V and z_Delta
  -> PIVOT acquisition
       low-response / confident: retain proxy decision
       ambiguous / high-response: paired high-fidelity rollout
  -> differential transfer model g_theta
  -> corrected Delta_true and update selection
  -> next incumbent
```

The high-fidelity unit is a paired rollout sharing initial state, exogenous path, random seed, and opponent initialization whenever the world permits. PIVOT predicts the correction:

```text
g_theta(context, z_Delta, Delta_V) -> Delta_* - Delta_V
```

It does not first fit a global policy-value model and subtract two independently predicted values.

The first acquisition baselines are:

1. Random high-fidelity query.
2. Top-proxy query.
3. Largest-footprint query.
4. Highest-uncertainty query.
5. Highest-reversal-probability query.
6. PIVOT decision-change query.

The initial VOI approximation is decision-centric: prioritize candidates whose high-fidelity result can change the selected update, normalized by query cost. A full Bayesian VOI implementation is out of scope for the first version.

### 5.1 Canonical transition record

Every candidate transition is persisted as one row (Parquet for experiment tables; YAML/JSON for configuration and provenance). The schema must include these fields, with unavailable values represented as explicit `null`:

```text
transition_id, round_id, incumbent_policy_id, candidate_policy_id,
candidate_index, improvement_operator, edit_type,
proxy_world_id, high_fidelity_world_id,
proxy_incumbent_value, proxy_candidate_value, delta_proxy,
actor_incumbent_value, actor_candidate_value, delta_actor,
strategic_incumbent_value, strategic_candidate_value, delta_strategic,
mechanical_effect, competition_effect,
improvement_reversal, strategic_improvement_reversal,
update_footprint, footprint_components,
response_strength, competition_strength, opponent_context,
hf_queried, hf_query_reason, hf_query_cost,
seed, paired_seed_ids, config_id, git_commit, timestamp
```

The record must never silently substitute one fidelity level for another. It is the source table for metrics, figures, gate evidence, and the first milestone.

## 6. World hierarchy and finance ladder

The generic scientific hierarchy has exactly three worlds:

| World | Definition | Output |
| --- | --- | --- |
| World 0: Observer | `s_(t+1) ~ P_0(s_(t+1) | s_t)`; focal actions do not materially alter future dynamics | `Delta_proxy` |
| World 1: Actor | `s_(t+1) ~ P(s_(t+1) | s_t, a_t, pi_i)`; focal policy changes subsequent dynamics | `Delta_actor` |
| World 2: Strategic | opponents respond through fixed, reactive, or finite-step adaptive policies | `Delta_strategic` |

Finance is a testbed ladder inside this hierarchy:

| Level | World | Required role |
| --- | --- | --- |
| F0 | Historical backtest | Fixed path and standard costs; no endogenous response |
| F1 | Historical execution replay | Spread, partial fills, queue, slippage, fees, and execution constraints |
| F2 | Interactive actor market | Impact, liquidity depletion, recovery/reversion, and execution-state feedback |
| F3 | Alternative generative world model | Optional disagreement/interventional proxy; never ground truth |
| F4 | Strategic multi-agent market | Noise traders, liquidity provider, and one adaptive competitor |

The strategic opponent ladder is S0 fixed opponents, S1 reactive rules, and S2 finite-step adaptive responses. Only the focal policy uses PIVOT self-improvement. A learned world model such as M3 is an alternative intervention model and disagreement signal, never a ground-truth label.

## 7. Experimental program

Run the following experiments in strict order. The controlled environment must be complete before finance integration.

| Experiment | Required evidence |
| --- | --- |
| E1: Improvement Reversal | `Delta_proxy` vs `Delta_true`, ISC, IRR, IDE, confidence intervals, reversal quadrant |
| E2: Response x footprint | IRR heatmap, stratified curves, and regression over response strength and update footprint |
| E3: Performative overoptimization | `J_V(pi_t)` and `J_*(pi_t)` over rounds; do not assume deterioration must appear |
| E4: Global vs Improvement Fidelity | Equal HF budget; policy-value MAE/rank correlation, IDE, ISC, IRR, ISR |
| E5: PIVOT budget frontier | Proxy Only, Random HF, Top Proxy HF, Largest Footprint HF, Uncertainty HF, PIVOT, All-HF Oracle; CTI and ISR vs budget |
| E6: Financial mechanical reversal | F0 -> F1 -> F2; vary participation, size, urgency, turnover; compare proxy and actor deltas |
| E7: Strategic reversal | Same transition; compare actor and strategic deltas; do not tune to force the result |
| E8: Competition strength | Sweep opponent count, adaptation steps/rate, market-share sensitivity; report SIRR and strategic sensitivity |
| E9: Closed-loop self-improvement | Only after E1-E8; compare final `J_H(pi_T)` and `CTI_T` under equal HF budgets |

The mandatory finance causal knob is:

```text
rho = Agent Trading Volume / Market Volume
```

Hold `pi -> pi'` fixed, evaluate F0/F1/F2/F4, and test whether participation changes magnitude, ordering, or sign. Do not tune the simulator merely to force a zero crossing. The desired but not guaranteed strategic case is `Delta_proxy > 0`, `Delta_actor > 0`, `Delta_strategic < 0`.

EvoQuant-style typed candidate generation is an improvement operator, not the paper's novelty. M3 is an F3 alternative world adapter, not PIVOT. LLM generation and learned-world-model integration begin only after the controlled estimand and PIVOT budget gates pass.

## 8. Baselines and ablations

### Main baselines

| Baseline | Tests |
| --- | --- |
| B1 Proxy Only | No interventional evaluation |
| B2 Random HF | Value of extra high-fidelity budget |
| B3 Top Proxy | Standard cheap-screening strategy |
| B4 Largest Footprint | Whether a simple footprint heuristic explains gains |
| B5 Global Value Model | Policy-value model followed by subtraction |
| B6 Global Ranking Model | Policy-ranking fidelity rather than update fidelity |
| B7 Uncertainty Sampling | Generic active-learning reference |
| B8 All-HF Oracle | Expensive upper reference |
| B9 PIVOT | Transition-level differential modeling plus active intervention |

Performative policy-gradient methods are controlled-environment references, not direct PIVOT baselines.

### Required ablations

```text
paired vs unpaired rollouts
transition model vs global value model
footprint vs no footprint
active vs random high-fidelity queries
PIVOT acquisition vs Top Proxy
small vs large policy updates
weak vs strong environment response
F1 replay vs F2 interactive environment
fixed vs adaptive competitors
single vs multiple response models
candidate count
high-fidelity budget
```

## 9. Claim boundary

| Work | Core object |
| --- | --- |
| PEG | Prediction/evaluation ranking |
| EPV | Selected action |
| FinAuth | Authorization evidence |
| Intent-Cert | Execution route |
| LATTICE | External capability intervention |
| EvoQuant | Strategy evolution loop |
| PePG | Performative optimal policy |
| WorldGym | Policy evaluation fidelity |
| Policy-Aware Simulator Learning | Robust policy-value simulator |
| FinEvo | Multi-agent financial ecology |
| This paper | Improvement fidelity of `pi -> pi'` |

The paper does not certify actions, claim a universal equilibrium solver, or claim that learned world models are ground truth. It validates learning progress under induced world response.

## 10. Go/No-Go gates

These are pre-registration-style engineering gates, not guaranteed findings:

1. Reversal must occur at non-extreme response strengths and persist across seeds.
2. IRR must vary structurally with response strength and update footprint; a noise-only effect is a No-Go.
3. A global value/rank evaluator must not completely eliminate local sign errors under the same high-fidelity budget.
4. PIVOT must outperform Random HF and Top Proxy HF at matched high-fidelity query counts, with uncertainty intervals reported.
5. Finance reversal must occur in a defensible participation range and survive cost/slippage controls.
6. Strategic reversal is preferred evidence. If competitors only add variance without a systematic effect, move that result to the appendix and narrow the claim.

## 11. Main-paper page budget

The ICLR 2027 main paper is planned for nine pages. Production code must generate seven figures; the page budget selects the five most central figures for the main text and places the remaining diagnostic figures in the appendix if space requires.

| Page | Content |
| --- | --- |
| 1 | Motivation and Figure 1 |
| 2 | Related work and problem statement |
| 3 | Improvement Fidelity and metrics |
| 4 | Response decomposition and theory |
| 5 | PIVOT |
| 6 | Controlled phenomenon |
| 7 | PIVOT budget result |
| 8 | Finance and strategic response |
| 9 | Ablation, limitations, conclusion |

Appendix material cannot carry the core theorem or the main evidence.

The production figure set is:

1. When Better Gets Worse: proxy versus true delta.
2. Improvement Reversal Phase Diagram: IRR over response and footprint.
3. Optimizing the Wrong World: proxy and adaptive-world curves.
4. Policy Fidelity Is Not Improvement Fidelity: global value/rank versus local update quality.
5. PIVOT Budget Frontier: HF budget versus CTI or ISR.
6. Observer -> Actor -> Strategic: identical transition across worlds.
7. Strategic Improvement Reversal: competition strength versus true update gain.

## 12. Strict implementation order and milestones

Do not change this order without a documented scientific reason.

| Phase | Deliverable | Hard boundary |
| --- | --- | --- |
| P0 | `PolicyTransition`, world interface, paired evaluator, metrics, transition logging | No finance, LLM, or multi-agent |
| P1 | Controlled performative environment | No finance, LLM, or multi-agent |
| P2 | E1 and E2: establish whether reversal exists and is structured | Stop if the effect is pathological or noise-only |
| P3 | E4: global policy fidelity versus local improvement fidelity | Record a null result honestly if global fidelity solves it |
| P4 | Minimal PIVOT: differential model, active querying, update selection | Match HF budgets across baselines |
| P5 | E5 budget frontier | Gate before finance |
| P6 | F0 -> F1 finance replay | Offline and virtual fills only |
| P7 | F2 interactive actor market and participation sweep | Stop if only absurd footprints reverse |
| P8 | Add F4 strategic opponents through S0/S1/S2 | One focal self-improver only |
| P9 | Run E7/E8, then E9 closed loop after prior experiments | Do not force a zero crossing |
| P10 | Optional LLM/EvoQuant and F3/M3 adapters | Deferred extensions only |

### First milestone

One command must create a transition-level dataset containing `round`, `incumbent_id`, `candidate_id`, proxy incumbent/candidate values, `delta_proxy`, true incumbent/candidate values, `delta_true`, `improvement_reversal`, `update_footprint`, `response_strength`, and `seed`; it must also produce the proxy-versus-true scatter, IRR versus response, IRR versus footprint, response-by-footprint heatmap, and confidence intervals. No finance, LLM, or multi-agent code is allowed in this milestone.

### Second milestone

Using the same HF budget, compare a strong policy-value evaluator with a transition-level differential evaluator and produce policy-value MAE, policy rank correlation, IDE, ISC, IRR, and ISR. If the global evaluator completely solves the update problem, record that outcome and narrow the claim.

### Third milestone

PIVOT must beat Random HF and Top Proxy HF at matched budgets on CTI or ISR over multiple seeds with paired evaluation where possible.

### Fourth and fifth milestones

Finance must show a structured F0/F1 versus F2 difference at a physically interpretable, economically plausible footprint. Strategic adaptation should add a systematic effect beyond mechanical response; if it only adds variance, keep it secondary and do not redesign the paper around it.

The full 40-section master specification is archived at `docs/master-goal.md` and is authoritative for requirements not repeated in this condensed design.

## 13. Engineering, testing, and non-goals

The implementation targets Python 3.10+ with typed interfaces, dataclasses or Pydantic-style schemas, centralized configuration, explicit seeds, and deterministic modes where possible. Every run persists configuration, random seed, git commit, dependency versions, environment version, dataset/version ID, timestamp, and machine information where relevant. Constants must not be hidden in implementation files. Missing worlds, datasets, or failed runs must be recorded explicitly and fail loudly rather than being silently substituted or dropped.

Mandatory unit coverage includes transition serialization, metric formulas, reversal detection, paired deltas, confidence intervals, footprint calculations, and improvement decomposition. Analytically solvable toy environments must make direct, mechanical, and competition effects known. At least one integration test must complete `pi_t -> candidates -> proxy -> PIVOT -> HF query -> pi_{t+1}`.

Non-goals are reproducing all EvoQuant experiments, building another factor-mining or HFT system, maximizing Sharpe, training a market foundation model, reproducing M3 from scratch, solving general multi-agent equilibrium, fully co-evolving all agents, live or production trading, action authorization, abstention, EPV-style gating, KAIROS integration, or adding complexity for novelty optics.

## 14. Reference leads supplied with the design

The following links were supplied in the preceding research discussion. They are research leads for the implementation bibliography; each must be independently checked for version, claims, and citation details before submission.

- ICLR 2027 call and author guidelines: https://iclr.cc/Conferences/2027/CallForPapers and https://iclr.cc/Conferences/2027/AuthorGuidelines
- EvoQuant: https://arxiv.org/html/2607.12455v1
- EvoPolicyGym: https://arxiv.org/html/2607.02440v1
- Self-Authored Verification: https://arxiv.org/html/2607.24300v1
- Performative Policy Gradient: https://arxiv.org/pdf/2512.20576
- WorldGym: https://arxiv.org/html/2506.00613v2
- Policy-Aware Simulator Learning: https://arxiv.org/abs/2605.29032v2
- Interactive LOB simulation: https://arxiv.org/pdf/2603.24137v1
- Multi-Agent Performative Prediction: https://arxiv.org/pdf/2502.08063v1
- Performative Markov Potential Games: https://arxiv.org/pdf/2504.20593v1
- M3: https://arthurzhang02.github.io/m3-market-microstructure/M3_paper.pdf
- TradeFM: https://arxiv.org/abs/2602.23784v1
- ABIDES-MARL: https://arxiv.org/abs/2511.02016v1
- FinEvo: https://arxiv.org/html/2602.00948v1
