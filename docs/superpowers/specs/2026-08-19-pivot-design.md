# PIVOT Research Design

**Status:** Frozen research design for implementation planning
**Date:** 2026-08-19
**Target venue:** ICLR 2027
**Planning deadlines supplied for this freeze:** abstract 2026-09-18 AOE; full paper 2026-09-25 AOE; main text limit 9 pages
**Working title:** *When Better Gets Worse: Improvement Fidelity for Self-Improving Agents in Adaptive Worlds*
**Method name:** PIVOT — *Paired Interventional Validation of Optimization Transitions*

## 1. Design lock

The paper studies whether a self-improving agent's update remains an improvement after deployment changes the world. The statistical object is the transition

```text
pi_t -> pi_{t+1}
```

and not an isolated state, trajectory, or policy. The paper is intentionally limited to one phenomenon, one method, one theory, and two experimental levels.

| Item | Frozen choice |
| --- | --- |
| Core phenomenon | Improvement Reversal |
| Core estimand | Improvement Fidelity of a policy update |
| Core metric | Improvement Sign Consistency (ISC) |
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
ISR_t = max_j Delta_{*,t,j} - Delta_{*,t,j_hat}
CTI_T = sum_t Delta_{*,t}
```

All sign metrics must specify a zero tolerance `tau_sign`; values in `[-tau_sign, tau_sign]` are treated as ties and reported separately.

### 3.3 Update footprint

The footprint is a feature of the transition, not a replacement for the policy:

```text
z_Delta = [ KL(pi' || pi), occupancy shift, action shift,
            entropy shift, support expansion ]
```

Finance-specific features are:

```text
[ turnover, order size, participation, holding time, urgency,
  rebalance magnitude, liquidity consumption ]
```

The first implementation uses synthetic edits and one-component typed edits so footprint can be controlled and audited.

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

## 6. World fidelity ladder

| Level | World | What it captures | What it does not prove |
| --- | --- | --- | --- |
| W0 | Observer | Historical state transitions or backtest | Endogenous response |
| W1 | Execution replay | Fees, fills, queue, partial fills, slippage | Counterfactual future market response |
| W2 | Interactive actor | Focal actions alter the market or transition process | Adaptive competitor response |
| W3 | Generative counterfactual | Alternative order-flow or state trajectories | Ground truth; it remains a model |
| W4 | Strategic | Competitors adapt to the focal policy | Universal equilibrium correctness |

W3 adapters such as M3 are alternative intervention models and disagreement signals, not ground truth labels. The paper must not claim that a single learned simulator is the true world.

## 7. Experimental program

### Level A: controlled performative environment

Use a fully known environment with response strength `lambda_response` and controlled update footprint `d(pi, pi')`.

Required experiments:

1. **Improvement Reversal:** scatter `Delta_proxy` against `Delta_true`; report all four sign quadrants and confidence intervals for IRR.
2. **Performative overoptimization:** plot `J_V(pi_t)` and `J_*(pi_t)` over self-improvement rounds.
3. **Global vs differential fidelity:** hold high-fidelity data budget fixed; compare policy-value MAE/rank metrics with IDE/ISC/IRR.
4. **PIVOT budget frontier:** plot cumulative true improvement or update-selection regret against the number of high-fidelity queries.

The first heatmap is `IRR(lambda_response, footprint)`.

### Level B: finance testbed

Use the same fixed transition `pi -> pi'` while varying participation rate:

```text
rho = AgentVolume / MarketVolume
```

Compare backtest, replay, interactive actor, and strategic worlds. The main response plot reports proxy, actor, and strategic deltas against participation and opponent adaptation strength.

The target strategic result is:

```text
Delta_proxy > 0, Delta_actor > 0, Delta_strategic < 0
```

If the result only appears at implausibly large participation, it fails the finance Go/No-Go gate.

### Deferred extensions

EvoQuant-style typed candidate generation is an operator `A`, not the paper's novelty. M3 is a `V3` world adapter, not PIVOT. LLM generation and M3 integration begin only after Level A and the PIVOT budget gate pass.

## 8. Baselines and ablations

### Main baselines

| Baseline | Tests |
| --- | --- |
| Proxy Only | No interactive verification |
| Random HF | Value of the high-fidelity budget itself |
| Top Proxy HF | Original cheap-screening strategy |
| Global Value Model | Whether policy-level fidelity is enough |
| Policy-Aware Simulator | Strong global simulator fidelity reference |
| PIVOT | Paired transition-level fidelity |

Performative policy-gradient methods are controlled-environment references, not direct PIVOT baselines.

### Required ablations

```text
paired vs unpaired rollouts
transition model vs global value model
footprint vs no footprint
active vs random high-fidelity queries
mechanical vs strategic response
fixed vs adaptive opponents
small vs large updates
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

The ICLR 2027 main paper is planned for nine pages:

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

## 12. Implementation order

1. Define `PolicyTransition` and the paired evaluator.
2. Build the controlled performative environment and reproduce reversal.
3. Test global fidelity versus improvement fidelity.
4. Implement PIVOT and the budget frontier.
5. Add historical replay and execution-aware finance adapters.
6. Add interactive impact.
7. Add adaptive competitors.
8. Only then evaluate EvoQuant/LLM candidate operators.
9. Add M3 and cross-world disagreement as the final robustness extension.

## 13. Reference leads supplied with the design

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
