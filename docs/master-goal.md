# CODEX MASTER GOAL

## Project: PIVOT

**PIVOT = Paired Interventional Verification of Optimization Transitions**

### Target Paper

**When Better Gets Worse: Improvement Fidelity for Self-Improving Agents in Adaptive Worlds**

---

# 1. Mission

Build a rigorous, reproducible research framework for studying whether an apparent improvement made by a self-improving agent remains an improvement after deployment changes the environment and, in the strongest setting, causes other agents to strategically adapt.

The fundamental research object is not:

* a prediction;
* an individual action;
* a selected winner;
* a policy in isolation;
* a simulator trajectory;
* an authorization decision.

The fundamental object is a **policy update**:

[
\boxed{
\pi_t\rightarrow\pi'_{t,j}
}
]

The central scientific question is:

> **When does a policy update that looks better under a cheap or fixed verifier remain better in the adaptive world created by deploying that update?**

The project must distinguish three notions:

[
\text{Proxy Improvement}
]

[
\text{Endogenous Improvement}
]

[
\text{Strategically Robust Improvement}.
]

The key failure mode is:

[
\boxed{
\Delta_{\mathrm{proxy}}>0
\quad\text{but}\quad
\Delta_{\mathrm{true}}<0.
}
]

Call this:

## Improvement Reversal

A stronger multi-agent failure is:

[
\boxed{
\Delta_{\mathrm{actor}}>0
\quad\text{but}\quad
\Delta_{\mathrm{strategic}}<0.
}
]

Call this:

## Strategic Improvement Reversal

---

# 2. Core Thesis

Self-improving agents are usually evaluated using proxies such as:

* offline benchmarks;
* backtests;
* fixed simulators;
* world models;
* validation environments;
* external verifiers.

These tools generally estimate quantities such as:

[
J_V(\pi).
]

But a self-improving system does not merely ask:

> Is policy (\pi) good?

It repeatedly asks:

> Is candidate (\pi') better than incumbent (\pi)?

Therefore the directly relevant quantity is:

[
\Delta_V(\pi,\pi')
==================

J_V(\pi')-J_V(\pi).
]

If policy deployment changes the environment, true performance is:

[
\mathcal J(\pi)
===============

J(\pi;\mathcal M[\pi]).
]

The true improvement is therefore:

[
\boxed{
\Delta_*(\pi,\pi')
==================

## J(\pi';\mathcal M[\pi'])

J(\pi;\mathcal M[\pi]).
}
]

The central thesis of this project is:

> **For self-improving agents, the appropriate target of environment fidelity is the policy transition, not necessarily the absolute policy value.**

---

# 3. Scientific Identity and Claim Boundary

This project belongs to a new research line:

## Valid Improvement in Adaptive Worlds

It must remain clearly distinct from prior research on:

* prediction validity;
* model ranking;
* execution-aware evaluation;
* selected-action risk;
* action authorization;
* route certification;
* abstention;
* capability gating.

Do not formulate PIVOT as:

* “Should this action execute?”
* “Should the system abstain?”
* “Is this action authorized?”
* “Is this selected action safe?”

Those belong to an Evidence-Before-Action paradigm.

This project studies the next part of the loop:

[
\boxed{
Action
\rightarrow
World Response
\rightarrow
Learning
\rightarrow
Improvement
}
]

The conceptual distinction is:

> **Previous work asks whether an action is justified before it changes the world. PIVOT asks whether an apparent improvement remains an improvement after it changes the world.**

The project must never collapse back into an EPV-style permission or gating method.

---

# 4. Explicit Non-Novelty Boundaries

Do not claim novelty from any of the following.

## 4.1 Self-Evolving Strategy Generation

Existing work already demonstrates iterative:

[
Strategy
\rightarrow
Diagnosis
\rightarrow
Edit
\rightarrow
Verification
\rightarrow
Promotion.
]

Therefore:

**Do not claim that generating and verifying repeated strategy edits is novel.**

Self-improvement algorithms such as EvoQuant should be treated as:

[
\boxed{
\text{Improvement Operators}
}
]

that produce candidate transitions.

---

## 4.2 Evaluating Policy Evolution

Existing policy-evolution benchmarks already evaluate repeated revision-feedback trajectories.

Therefore:

**Do not claim that policy evolution itself is a new evaluation object.**

Our object is specifically:

[
\boxed{
\text{the fidelity of each local improvement transition under endogenous response}.
}
]

---

## 4.3 Verifier–Deployment Gap

External or sealed evaluation of self-improving agents already exists.

Therefore:

**Do not claim simply that self-verification may fail or that external verification is needed.**

Our stronger hypothesis is:

> An external verifier may be perfectly honest and internally correct yet still approve a harmful update because it evaluates the wrong environment dynamics.

The desired message is:

[
\boxed{
\text{A verifier can be correct about the wrong world.}
}
]

---

## 4.4 Performative RL

Policy-dependent transition and reward distributions are already studied.

Therefore:

**Do not claim that agents changing their environment is novel.**

Our information regime is different:

* high-fidelity performative evaluation is expensive;
* most candidate updates are evaluated using cheap proxies;
* the system must allocate limited interventional evaluation budget;
* the central quantity is local update fidelity.

---

## 4.5 Policy-Aware Simulator Learning

Existing research already studies simulators optimized for downstream policy-value fidelity and simulator exploitation.

Therefore:

**Do not make “policy-aware simulation” the main novelty.**

The distinction is:

[
\text{Policy-value fidelity}
]

versus:

[
\boxed{
\text{Improvement fidelity}.
}
]

---

## 4.6 Multi-Agent Financial Evolution

Do not build a general financial multi-agent ecology and claim novelty from competition.

Competition exists only as the strongest form of environment response.

The focal research question remains:

[
\boxed{
\text{Does one specific update remain beneficial after competitors respond?}
}
]

---

# 5. Three-World Abstraction

Implement all experimental environments using a common hierarchy.

---

## World 0 — Observer World

The environment is effectively exogenous.

[
s_{t+1}
\sim
P_0(s_{t+1}\mid s_t).
]

The focal agent observes a historical or fixed world.

Its actions do not materially alter future dynamics.

Examples:

* historical backtest;
* fixed validation environment;
* historical price replay;
* fixed-distribution RL environment.

Output:

[
\Delta_{\mathrm{proxy}}.
]

---

## World 1 — Actor World

The focal policy affects subsequent environment dynamics.

[
s_{t+1}
\sim
P(
s_{t+1}
\mid
s_t,a_t,\pi_i
).
]

Examples:

* performative gridworld;
* endogenous resource allocation;
* market impact;
* interactive LOB;
* liquidity depletion.

Output:

[
\Delta_{\mathrm{actor}}.
]

This is the first genuinely interventional world.

---

## World 2 — Strategic World

Other agents respond to the focal agent.

The focal update is:

[
\pi_i\rightarrow\pi_i'.
]

Opponent behavior may change:

[
\pi_{-i}
\rightarrow
BR_{-i}^{K}(\pi_i').
]

The world therefore depends jointly on:

[
\mathcal M[
\pi_i,
\pi_{-i}
].
]

Output:

[
\Delta_{\mathrm{strategic}}.
]

Only the focal agent should self-improve in the first implementation.

Do not initially allow all agents to self-evolve simultaneously.

---

# 6. Improvement Decomposition

For each transition:

[
\pi_i\rightarrow\pi_i'
]

compute whenever possible the following components.

---

## 6.1 Direct Policy Effect

Freeze environment dynamics and opponent policies:

[
\Delta_{\mathrm{direct}}
========================

J_i(
\pi_i',
\pi_{-i};
M
)
-

J_i(
\pi_i,
\pi_{-i};
M
).
]

This approximates what an ordinary verifier sees.

---

## 6.2 Mechanical Environment Response

Allow the focal agent to alter the environment while opponent policies remain fixed:

[
\Delta_{\mathrm{actor}}
=======================

J_i(
\pi_i';
M[\pi_i']
)
-

J_i(
\pi_i;
M[\pi_i]
).
]

Define:

[
\boxed{
\Delta_{\mathrm{mechanical}}
============================

## \Delta_{\mathrm{actor}}

\Delta_{\mathrm{direct}}.
}
]

Examples include:

* market impact;
* liquidity consumption;
* queue changes;
* resource depletion;
* changed state visitation.

---

## 6.3 Strategic Response

Allow opponents to adapt.

[
\Delta_{\mathrm{strategic}}
===========================

J_i(
\pi_i',
BR_{-i}(\pi_i')
)
-

J_i(
\pi_i,
BR_{-i}(\pi_i)
).
]

Define:

[
\boxed{
\Delta_{\mathrm{competition}}
=============================

## \Delta_{\mathrm{strategic}}

\Delta_{\mathrm{actor}}.
}
]

The conceptual decomposition is therefore:

[
\boxed{
\Delta_{\mathrm{true}}
======================

\Delta_{\mathrm{direct}}
+
\Delta_{\mathrm{mechanical}}
+
\Delta_{\mathrm{competition}}.
}
]

Do not force this decomposition in environments where counterfactual components are not identifiable.

Record unavailable components explicitly rather than fabricating them.

---

# 7. New Evaluation Target: Improvement Fidelity

Implement the following metrics as first-class objects.

---

## 7.1 Improvement Differential Error

[
IDE
===

\mathbb E[
|\Delta_V-\Delta_*|
].
]

---

## 7.2 Improvement Sign Consistency

[
ISC
===

P[
\operatorname{sign}(\Delta_V)
=============================

\operatorname{sign}(\Delta_*)
].
]

---

## 7.3 Improvement Reversal Rate

[
\boxed{
IRR
===

P(
\Delta_*<0
\mid
\Delta_V>0
).
}
]

This is one of the primary metrics.

---

## 7.4 Strategic Improvement Reversal Rate

[
\boxed{
SIRR
====

P(
\Delta_{\mathrm{strategic}}<0
\mid
\Delta_{\mathrm{actor}}>0
).
}
]

This measures updates that survive mechanical response but fail under strategic adaptation.

---

## 7.5 Magnitude Transfer Ratio

When numerically stable:

[
MTR
===

\frac{
\Delta_*
}{
\Delta_V
}.
]

Do not use this metric when the denominator is close to zero.

---

## 7.6 Update Selection Regret

At improvement round (t), suppose there are (K) candidates.

If candidate (\hat j) is selected:

[
ISR_t
=====

\max_j
\Delta^H_{t,j}
--------------

\Delta^H_{t,\hat j}.
]

---

## 7.7 Cumulative True Improvement

[
\boxed{
CTI_T
=====

\sum_{t=1}^{T}
\Delta^H_t.
}
]

Also report:

[
J_H(\pi_T).
]

---

## 7.8 High-Fidelity Evaluation Cost

Track:

* number of high-fidelity transitions;
* number of high-fidelity rollouts;
* environment steps;
* simulator calls;
* compute cost where meaningful.

Never report PIVOT performance without reporting its high-fidelity budget.

---

# 8. Theory Program

The implementation must support empirical verification of the following theory program.

---

## Theory 1 — Absolute Fidelity Is Sufficient but Not Necessary

If:

[
\sup_\pi
|
J_V(\pi)-J_*(\pi)
|
\le\epsilon,
]

then for any update:

[
\pi\rightarrow\pi'
]

we have:

[
\boxed{
|
\Delta_V-\Delta_*
|
\le2\epsilon.
}
]

Therefore:

[
\text{Global Value Fidelity}
\Rightarrow
\text{Improvement Fidelity}.
]

But the converse does not hold.

For example:

[
J_V(\pi)
========

J_*(\pi)+C
]

may have arbitrarily large absolute error while preserving:

[
\Delta_V=\Delta_*.
]

The empirical system must test the consequence:

> A globally inaccurate evaluator may still be excellent for self-improvement, while a globally strong evaluator may still perform poorly on locally visited improvement transitions.

---

## Theory 2 — Response Sensitivity × Update Footprint

Define update footprint:

[
d(\pi,\pi').
]

Assume environment response satisfies an appropriate regularity condition such as:

[
D(
M[\pi'],
M[\pi]
)
\le
L_M
d(\pi,\pi').
]

Attempt to derive a bound of the form:

[
\boxed{
|
\Delta_{\mathrm{actor}}
-----------------------

\Delta_{\mathrm{direct}}
|
\le
C
L_M
d(\pi,\pi').
}
]

Exact constants depend on the environment class and objective.

Do not hard-code or claim a bound that has not been proven.

The code must nevertheless make the quantities empirically measurable:

* update footprint;
* environment response strength;
* improvement error.

The main empirical hypothesis is:

[
\boxed{
Improvement\ Error
\uparrow
\text{ as }
Update\ Footprint
\times
Environment\ Sensitivity
\uparrow.
}
]

---

## Theory 3 — Strategic Sensitivity

For multi-agent experiments define:

[
S_{-i}
======

\frac{
D(
BR_{-i}(\pi_i'),
BR_{-i}(\pi_i)
)
}{
d(\pi_i',\pi_i)+\epsilon
}.
]

Test whether:

[
S_{-i}\uparrow
]

is associated with:

[
SIRR\uparrow.
]

Investigate whether small focal updates can induce disproportionately large competitor adaptation.

This is an extension, not a prerequisite for the basic paper.

---

# 9. Core Method: PIVOT

PIVOT solves:

> Given multiple candidate policy updates and a limited budget for high-fidelity interventional evaluation, which candidate transitions should receive expensive evaluation and which candidate should become the next incumbent?

Pipeline:

```text
Incumbent policy π_t
        │
        ▼
Improvement Operator
        │
        ▼
π'_{t,1},...,π'_{t,K}
        │
        ▼
Cheap Proxy Evaluation
        │
        ├── Δ_proxy
        ├── update footprint
        ├── context
        └── cheap uncertainty
        │
        ▼
      PIVOT
        │
        ├───────────────┐
        │               │
 low-value HF query   high-value HF query
        │               │
        │               ▼
        │        paired interventional
        │            evaluation
        │               │
        └───────┬───────┘
                ▼
    Differential Transfer Model
                │
                ▼
      corrected improvement
                │
                ▼
       choose π_{t+1}
                │
                └──────→ next round
```

---

# 10. PIVOT Component A — Paired Evaluation

Whenever possible, evaluate:

[
\pi
]

and:

[
\pi'
]

using matched conditions.

Use identical:

* initial states;
* scenario IDs;
* external random seeds;
* market days;
* exogenous order flows;
* opponent initialization.

Implement:

`PairedEvaluator`

returning:

```text
incumbent_value
candidate_value
delta
standard_error
confidence_interval
num_rollouts
paired_seed_ids
```

Prefer estimating:

[
\Delta
]

directly from paired differences.

Do not separately estimate noisy policy values and subtract them unless pairing is impossible.

---

# 11. PIVOT Component B — Differential Transfer Model

Primary model target:

[
\boxed{
\Delta_*-\Delta_{\mathrm{proxy}}.
}
]

Train:

[
g_\theta(
z_{\Delta},
context,
\Delta_{\mathrm{proxy}}
)
\rightarrow
\widehat{
\Delta_*-\Delta_{\mathrm{proxy}}
}.
]

Then:

[
\tilde\Delta_*
==============

\Delta_{\mathrm{proxy}}
+
\hat\Delta_{\mathrm{correction}}.
]

Implement initially:

1. linear/ridge regression;
2. gradient-boosted trees;
3. small MLP only if justified by data.

Do not begin with transformers or large neural models.

Optional diagnostic classifier:

[
P(
\Delta_*<0
\mid
\Delta_{\mathrm{proxy}}>0
).
]

This classifier is diagnostic only.

Do not turn it into an authorization gate.

---

# 12. PIVOT Component C — Active High-Fidelity Querying

At each round estimate which candidate transition has the highest value of high-fidelity evaluation.

Implement progressively.

### Baseline 1 — Random

Uniform selection.

### Baseline 2 — Top Proxy

Query the largest:

[
\Delta_{\mathrm{proxy}}.
]

### Baseline 3 — Largest Footprint

Query largest:

[
d(\pi,\pi').
]

### Baseline 4 — Highest Uncertainty

Query maximum transfer-model uncertainty.

### Baseline 5 — Highest Reversal Probability

Query candidate most likely to change sign under HF evaluation.

### PIVOT — Decision-Change / Value-of-Information

Preferred acquisition target:

[
VOI_j
\approx
\frac{
E[
\text{reduction in update-selection regret}
\mid query(j)
]
}{
Cost_j
}.
]

An exact Bayesian VOI implementation is not required for the first complete version.

A tractable approximation is acceptable.

---

# 13. Policy Transition Object

Create a canonical `PolicyTransition` data structure.

Required fields:

```text
transition_id
round_id
incumbent_policy_id
candidate_policy_id
candidate_index
improvement_operator
edit_type
proxy_world_id
high_fidelity_world_id

proxy_incumbent_value
proxy_candidate_value
delta_proxy

actor_incumbent_value
actor_candidate_value
delta_actor

strategic_incumbent_value
strategic_candidate_value
delta_strategic

mechanical_effect
competition_effect

improvement_reversal
strategic_improvement_reversal

update_footprint
footprint_components

response_strength
competition_strength
opponent_context

hf_queried
hf_query_reason
hf_query_cost

seed
paired_seed_ids
config_id
git_commit
timestamp
```

Unavailable values must be explicit `null`/missing fields.

Never silently substitute one fidelity level for another.

Use Parquet for experiment tables.

Use YAML/JSON for experiment configurations and provenance.

---

# 14. Generic Update Footprint

Implement:

`compute_update_footprint(pi, pi_prime, evaluation_states)`

Potential features:

* mean policy KL divergence;
* max policy KL divergence;
* action distribution shift;
* action magnitude shift;
* entropy change;
* occupancy-measure divergence;
* state-support expansion;
* trajectory divergence;
* episode-length change.

Expose both:

[
d(\pi,\pi')
]

and its components.

Do not collapse everything into one scalar before analysis.

---

# 15. Finance-Specific Update Footprint

Implement additional finance features:

* turnover change;
* position-size change;
* participation-rate change;
* holding-period change;
* rebalance-frequency change;
* order-size distribution shift;
* execution urgency;
* aggressive/passive order ratio;
* liquidity consumption;
* concentration;
* inventory duration;
* spread-crossing frequency.

Preserve the distinction:

[
\boxed{
Strategy\ Frequency
\neq
Simulation\ Frequency.
}
]

A minute-level strategy may be evaluated in an event-level LOB environment.

Do not place an LLM in the event-level execution hot path.

---

# 16. Improvement Operators

Expose a common API:

```python
ImprovementOperator.propose(
    incumbent_policy,
    context,
    num_candidates
) -> list[CandidatePolicy]
```

Implement in the following order.

---

## Operator A — Controlled Synthetic Perturbations

Mandatory.

Generate explicitly controllable updates:

* tiny;
* small;
* medium;
* large.

The purpose is causal study of:

[
d(\pi,\pi').
]

---

## Operator B — RL Policy Updates

Use a performative RL environment.

Support at minimum:

* ordinary policy-gradient update;
* stronger optimization rounds;
* optionally performative-aware policy optimization as a reference.

The goal is to generate realistic adaptive transitions.

Do not make state-of-the-art RL performance the research objective.

---

## Operator C — Typed Financial Strategy Edits

Create a lightweight strategy genome.

Editable components:

```text
signal
entry
exit
threshold
position_size
risk_size
holding_horizon
rebalance_frequency
execution_urgency
participation_rate
```

First experiments should modify one component at a time.

This enables mechanism attribution.

---

## Operator D — Optional LLM Improver

Add only after PIVOT works without LLMs.

LLM proposals must map into:

* typed edits;
* executable strategies;
* persistent structured diffs.

Persist every:

* prompt;
* response;
* edit;
* compilation result;
* strategy artifact.

LLM stochasticity must never be conflated with environment-response effects.

---

# 17. Controlled Environment

The first complete scientific result must come from a controlled environment where the true policy-dependent dynamics are known.

Requirements:

* reproducible environment;
* policy-dependent transition/reward dynamics;
* explicit response-strength parameter;
* cheap fixed-world proxy;
* true performative evaluator;
* configurable update footprint.

Sweep:

[
\text{response strength}
\times
\text{update footprint}
\times
\text{optimization strength}.
]

This environment must be completed before finance integration.

---

# 18. Finance World Ladder

Finance is the strongest real-world-motivated testbed, not the sole evidence source.

---

## F0 — Historical Backtest

Fixed historical price path.

Include standard transaction costs where appropriate.

No endogenous market response.

---

## F1 — Historical Execution Replay

Add:

* bid/ask spread;
* partial fills;
* queue logic when available;
* slippage;
* fees;
* execution constraints.

Future market trajectory remains mostly exogenous.

F1 is an execution-fidelity environment, not yet an endogenous market.

---

## F2 — Interactive Actor Market

The focal agent's actions must affect future market state.

At minimum model:

* endogenous impact;
* liquidity depletion;
* recovery/reversion;
* execution-state feedback.

F2 is the main finance high-fidelity target.

Do not rebuild an unnecessarily complete exchange simulator if a minimal validated response mechanism is sufficient for the scientific hypothesis.

---

## F3 — Alternative Generative World Model

Optional.

A learned market world model may be used as an alternative interventional proxy.

It must never be labeled ground truth.

Use it to measure:

* simulator disagreement;
* sign disagreement;
* update-fidelity disagreement.

Do not train a large market foundation model from scratch for this project.

---

## F4 — Strategic Multi-Agent Market

Add competitors.

Initial population:

* exogenous/noise traders;
* liquidity provider;
* one additional adaptive competitor.

Only the focal policy uses the PIVOT self-improvement process.

---

# 19. Strategic Opponent Ladder

Implement gradually.

## S0 — Fixed Opponents

[
\pi_{-i}'=\pi_{-i}.
]

---

## S1 — Reactive Opponents

Opponent behavior follows predefined response rules.

Examples:

* spread undercutting;
* quote adjustment;
* inventory response;
* liquidity withdrawal;
* execution-timing response.

---

## S2 — Adaptive Opponents

After focal update:

[
\pi_i\rightarrow\pi_i',
]

allow competitors to perform a finite number of adaptation steps:

[
BR_{-i}^{K}(\pi_i').
]

Sweep:

[
K.
]

Do not initially perform simultaneous unbounded co-learning.

---

# 20. Finance Causal Knob: Participation Rate

This is a mandatory finance experiment.

Hold a transition:

[
\pi\rightarrow\pi'
]

fixed.

Vary only:

[
\rho
====

\frac{
Agent\ Trading\ Volume
}{
Market\ Volume
}.
]

For identical transitions evaluate:

[
F0
\rightarrow
F1
\rightarrow
F2
\rightarrow
F4.
]

Plot:

[
x=\rho
]

and:

[
y=\Delta.
]

Curves:

```text
Δ_proxy
Δ_replay
Δ_actor
Δ_strategic
```

The scientific test is whether increasing footprint changes:

* magnitude;
* ordering;
* sign.

Do not tune the simulator merely to force a zero crossing.

---

# 21. Main Experimental Program

Run in strict order.

---

## E1 — Does Improvement Reversal Exist?

Controlled performative environment.

Produce:

[
\Delta_{\mathrm{proxy}}
]

versus:

[
\Delta_{\mathrm{true}}.
]

Report:

* ISC;
* IRR;
* IDE;
* confidence intervals.

Primary visualization:

scatter plot with the reversal quadrant highlighted.

---

## E2 — Response Strength × Update Footprint

Measure:

[
IRR
===

f(
response\ strength,
update\ footprint
).
]

Produce:

* heatmap;
* stratified curves;
* regression analysis.

Goal:

show whether reversal has systematic structure rather than random noise.

---

## E3 — Performative Overoptimization

Run:

[
\pi_0\rightarrow\pi_1\rightarrow\dots\rightarrow\pi_T.
]

Track:

[
J_V(\pi_t)
]

and:

[
J_*(\pi_t).
]

Test whether proxy optimization can continue improving while true adaptive-world performance plateaus or deteriorates.

Do not assume this pattern must appear.

---

## E4 — Global Fidelity vs Improvement Fidelity

Train:

### Model A

[
f(\pi)\rightarrow J_H(\pi)
]

### Model B

[
g(\pi,\pi')\rightarrow\Delta_H.
]

Equalize high-fidelity data budget.

Compare:

* policy-value MAE;
* policy rank correlation;
* IDE;
* ISC;
* IRR;
* update-selection regret.

This experiment is mandatory.

---

## E5 — PIVOT Budget Frontier

Compare:

* Proxy Only;
* Random HF;
* Top Proxy HF;
* Largest Footprint HF;
* Uncertainty HF;
* PIVOT;
* All-HF Oracle.

Sweep HF budget.

Plot:

[
HF\ Cost
\rightarrow
CTI_T.
]

Also report update-selection regret.

---

## E6 — Financial Mechanical Reversal

Use:

[
F0\rightarrow F1\rightarrow F2.
]

Sweep:

* participation;
* size;
* urgency;
* turnover.

Measure:

[
\Delta_{\mathrm{proxy}}
]

versus:

[
\Delta_{\mathrm{actor}}.
]

---

## E7 — Strategic Improvement Reversal

For exactly the same transition compare:

[
\Delta_{\mathrm{actor}}
]

and:

[
\Delta_{\mathrm{strategic}}.
]

The strongest desired empirical case is:

[
\boxed{
\Delta_{\mathrm{proxy}}>0,
\quad
\Delta_{\mathrm{actor}}>0,
\quad
\Delta_{\mathrm{strategic}}<0.
}
]

Do not fabricate or tune directly toward this result.

---

## E8 — Competition Strength

Sweep:

* opponent count;
* adaptation steps;
* adaptation learning rate;
* market-share sensitivity.

Measure:

[
SIRR
]

and strategic sensitivity.

---

## E9 — Closed-Loop Self Improvement

Only after all prior experiments work.

Run:

[
\pi_0
\rightarrow
\pi_1
\rightarrow
\dots
\rightarrow
\pi_T.
]

Compare final:

[
J_H(\pi_T)
]

and:

[
CTI_T
]

under equal high-fidelity evaluation budgets.

---

# 22. Baselines

Mandatory baseline families:

### B1 — Proxy Only

No interventional evaluation.

### B2 — Random High Fidelity

Controls for additional evaluation budget.

### B3 — Top Proxy

High-fidelity evaluation of apparently best candidate.

Represents standard multi-fidelity screening.

### B4 — Largest Footprint

Tests whether simple heuristics explain PIVOT.

### B5 — Global Value Model

Learn policy value, then derive update difference.

### B6 — Global Ranking Model

Optimize policy-ranking fidelity rather than update fidelity.

### B7 — Uncertainty Sampling

Generic active-learning baseline.

### B8 — All-HF Oracle

Expensive upper reference.

### B9 — PIVOT

Transition-level differential modeling plus active interventional evaluation.

Do not include unnecessary weak baselines.

---

# 23. Required Ablations

At minimum:

1. paired vs unpaired evaluation;
2. transition model vs global value model;
3. with vs without footprint features;
4. active vs random HF querying;
5. PIVOT acquisition vs Top Proxy;
6. small vs large policy updates;
7. weak vs strong environment response;
8. F1 replay vs F2 interactive environment;
9. fixed vs adaptive competitors;
10. single vs multiple response models;
11. candidate count;
12. HF budget.

---

# 24. Statistical Protocol

All final experiments must:

* use multiple independent random seeds;
* preserve transition-level raw data;
* report confidence intervals;
* distinguish paired from unpaired uncertainty;
* store sample counts;
* log discarded/failed runs;
* preserve failed transitions.

Prefer paired bootstrap or analytically justified paired confidence intervals.

Never report a reversal based on one trajectory.

Distinguish:

[
\text{sampling noise}
]

from:

[
\text{systematic response effect}.
]

Do not filter inconvenient seeds unless the exclusion rule was registered beforehand.

---

# 25. Required Figures

Production code must automatically generate the following.

---

## Figure 1 — When Better Gets Worse

Scatter:

[
\Delta_{\mathrm{proxy}}
]

vs:

[
\Delta_{\mathrm{true}}.
]

Highlight:

[
\Delta_{\mathrm{proxy}}>0,
\quad
\Delta_{\mathrm{true}}<0.
]

---

## Figure 2 — Improvement Reversal Phase Diagram

Heatmap:

[
IRR
]

as a function of:

[
Update\ Footprint
\times
Environment\ Response.
]

---

## Figure 3 — Optimizing the Wrong World

Across self-improvement rounds:

```text
proxy performance
true adaptive-world performance
```

---

## Figure 4 — Policy Fidelity Is Not Improvement Fidelity

Show:

* global policy-value/ranking quality;
* local update-sign quality.

---

## Figure 5 — PIVOT Budget Frontier

[
HF\ Budget
]

vs:

[
CTI
]

or:

[
Update\ Selection\ Regret.
]

---

## Figure 6 — Observer → Actor → Strategic

For identical updates show:

[
\Delta_{\mathrm{proxy}},
\quad
\Delta_{\mathrm{actor}},
\quad
\Delta_{\mathrm{strategic}}.
]

---

## Figure 7 — Strategic Improvement Reversal

Competition strength versus true update gain.

Show zero crossing if empirically present.

---

# 26. Repository Structure

Use approximately:

```text
pivot/
├── README.md
├── pyproject.toml
├── configs/
│   ├── controlled/
│   ├── finance/
│   ├── strategic/
│   └── sweeps/
│
├── src/
│   └── pivot/
│       ├── core/
│       │   ├── policy.py
│       │   ├── transition.py
│       │   ├── candidate.py
│       │   ├── world.py
│       │   └── result.py
│       │
│       ├── improvers/
│       │   ├── perturbation.py
│       │   ├── rl_update.py
│       │   ├── typed_finance.py
│       │   └── llm_optional.py
│       │
│       ├── environments/
│       │   ├── performative/
│       │   ├── finance_backtest/
│       │   ├── execution_replay/
│       │   ├── interactive_market/
│       │   └── strategic_market/
│       │
│       ├── footprint/
│       │   ├── generic.py
│       │   └── finance.py
│       │
│       ├── evaluation/
│       │   ├── paired.py
│       │   ├── decomposition.py
│       │   └── uncertainty.py
│       │
│       ├── transfer/
│       │   ├── global_value.py
│       │   ├── differential.py
│       │   └── reversal.py
│       │
│       ├── acquisition/
│       │   ├── random.py
│       │   ├── top_proxy.py
│       │   ├── footprint.py
│       │   ├── uncertainty.py
│       │   └── pivot.py
│       │
│       ├── algorithms/
│       │   └── pivot.py
│       │
│       ├── metrics/
│       │   └── improvement.py
│       │
│       └── logging/
│           └── transition_store.py
│
├── experiments/
│   ├── e1_reversal.py
│   ├── e2_phase_diagram.py
│   ├── e3_overoptimization.py
│   ├── e4_global_vs_local.py
│   ├── e5_budget_frontier.py
│   ├── e6_finance_actor.py
│   ├── e7_strategic_reversal.py
│   ├── e8_competition.py
│   └── e9_closed_loop.py
│
├── scripts/
│   ├── run_sweep.py
│   ├── aggregate_results.py
│   └── make_paper_figures.py
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── results/
│   ├── raw/
│   ├── processed/
│   ├── figures/
│   └── tables/
│
└── docs/
    ├── research_question.md
    ├── estimands.md
    ├── claim_boundary.md
    ├── experiment_protocol.md
    ├── theory_notes.md
    └── reproducibility.md
```

Notebooks may be used for exploration only.

Every final figure and table must be reproducible from scripts.

---

# 27. Engineering Requirements

Use:

* Python 3.10+ unless dependency constraints require another version;
* typed interfaces;
* dataclasses/Pydantic-style schemas;
* centralized configuration;
* explicit seeds;
* deterministic modes where possible.

Every run must persist:

```text
config
random seed
git commit
dependency versions
environment version
dataset/version ID
timestamp
machine information where relevant
```

Never hide experimental constants inside implementation files.

Never silently:

* change environment fidelity;
* replace unavailable components;
* drop failed runs;
* substitute datasets;
* reduce simulation realism.

Fail loudly.

---

# 28. Tests

Mandatory unit tests:

* `PolicyTransition` serialization;
* metric formulas;
* Improvement Reversal detection;
* paired delta calculation;
* confidence intervals;
* footprint calculations;
* improvement decomposition.

Create analytically solvable toy environments where:

[
\Delta_{\mathrm{direct}},
\quad
\Delta_{\mathrm{mechanical}},
\quad
\Delta_{\mathrm{competition}}
]

are known.

Mandatory integration test:

One complete:

[
\pi_t
\rightarrow
{\pi'*{t,j}}
\rightarrow
Proxy
\rightarrow
PIVOT
\rightarrow
HFQuery
\rightarrow
\pi*{t+1}
]

round.

---

# 29. Strict Implementation Order

Do not change this order without a scientific reason.

## P0

Implement:

* `PolicyTransition`;
* world interface;
* paired evaluator;
* metrics;
* logging.

---

## P1

Controlled performative environment.

No finance.

No LLM.

No multi-agent.

---

## P2

Complete E1 and E2.

Establish whether Improvement Reversal is real and structured.

---

## P3

Complete E4:

Global Fidelity vs Improvement Fidelity.

---

## P4

Implement minimal PIVOT:

* differential model;
* basic active querying;
* update selection.

---

## P5

Complete budget frontier.

---

## P6

Implement finance:

[
F0\rightarrow F1.
]

---

## P7

Implement interactive actor market:

[
F2.
]

Run participation-rate experiment.

---

## P8

Add strategic opponents:

[
F4.
]

---

## P9

Run Strategic Improvement Reversal experiments.

---

## P10

Only then consider:

* LLM strategy edits;
* EvoQuant integration;
* alternative learned market world model.

---

# 30. First Milestone

The first milestone is complete only when one command creates a transition dataset containing at least:

```text
round
incumbent_id
candidate_id

proxy_incumbent_value
proxy_candidate_value
delta_proxy

true_incumbent_value
true_candidate_value
delta_true

improvement_reversal

update_footprint
response_strength

seed
```

and automatically produces:

1. proxy-vs-true improvement scatter;
2. IRR vs environment response;
3. IRR vs update footprint;
4. response×footprint heatmap;
5. confidence intervals.

No finance.

No LLM.

No multi-agent.

---

# 31. Second Milestone

Demonstrate experimentally whether:

[
\boxed{
Global\ Policy\ Fidelity
\neq
Local\ Improvement\ Fidelity.
}
]

Train a strong policy-value evaluator and a transition-level differential evaluator using the same HF budget.

Produce a table with:

```text
policy value MAE
policy rank correlation
improvement differential error
improvement sign consistency
improvement reversal rate
update selection regret
```

If a global evaluator completely solves the update problem, explicitly record that result.

Do not manipulate the setting to preserve the paper hypothesis.

---

# 32. Third Milestone

PIVOT must beat:

* Random HF;
* Top Proxy HF;

under identical high-fidelity budgets on at least one primary metric:

[
CTI
]

or:

[
Update\ Selection\ Regret.
]

The comparison must use multiple seeds and paired evaluation where possible.

---

# 33. Fourth Milestone

Finance must show a structured difference between:

[
F0/F1
]

and:

[
F2.
]

The key variable should be a physically interpretable footprint parameter such as:

[
Participation\ Rate.
]

Do not proceed if differences only appear at absurdly unrealistic order sizes.

---

# 34. Fifth Milestone

Add strategic adaptation.

The strongest desired observation is:

[
\boxed{
\Delta_{\mathrm{proxy}}>0,
\quad
\Delta_{\mathrm{actor}}>0,
\quad
\Delta_{\mathrm{strategic}}<0.
}
]

If multi-agent competition merely increases noise without producing systematic effects:

* report it honestly;
* move multi-agent results to secondary analysis;
* do not redesign the entire project around forcing Strategic Improvement Reversal.

---

# 35. Go / No-Go Gates

## Gate A — Phenomenon

Proceed only if Improvement Reversal appears under non-pathological controlled settings.

---

## Gate B — Structure

Proceed only if reversal probability relates systematically to:

* update footprint;
* environment response;
* or both.

---

## Gate C — New Estimand

Proceed with the Improvement Fidelity framing only if global policy evaluation fails to fully explain local transition quality.

---

## Gate D — Method

PIVOT must outperform simple HF allocation strategies under fixed budget.

---

## Gate E — Finance

Financial reversal must occur under simulator-calibrated and economically plausible footprint levels.

---

## Gate F — Competition

Strategic adaptation must add a meaningful systematic effect beyond mechanical response.

---

# 36. Non-Goals

Do not prioritize:

* reproducing all EvoQuant experiments;
* building another factor-mining agent;
* creating a new HRT architecture;
* developing a new RL trading algorithm;
* maximizing Sharpe;
* training a new microstructure foundation model;
* reproducing M3 from scratch;
* solving general multi-agent equilibrium;
* fully co-evolving all market agents;
* live trading;
* production trading infrastructure;
* action authorization;
* abstention;
* EPV-style gating;
* KAIROS integration in the first paper;
* adding complexity purely for novelty optics.

---

# 37. Paper-Oriented Success Criteria

The project should ultimately support three defensible contributions only.

## Contribution 1 — New Evaluation Object

Introduce:

# Improvement Fidelity

as the fidelity of local policy transitions generated along a self-improvement trajectory.

---

## Contribution 2 — New Method

Introduce:

# PIVOT

which combines:

* paired differential evaluation;
* update-footprint modeling;
* active interventional querying;
* transition-level correction.

---

## Contribution 3 — New Empirical/Structural Result

Show whether:

[
\boxed{
Proxy\ Improvement
\not\Rightarrow
Endogenous\ Improvement
\not\Rightarrow
Strategically\ Robust\ Improvement.
}
]

and characterize when these failures occur.

---

# 38. Main Paper Message

Everything in the codebase should ultimately support or falsify the following statement:

> **A self-improving agent should not be evaluated only by whether a candidate performs better in the world used to generate or verify it. The relevant question is whether the update remains better in the adaptive world—and among the agents—that its deployment causes to exist.**

The concise scientific principle is:

[
\boxed{
\text{For self-improvement, fidelity should follow the update.}
}
]

---

# 39. Final Research Loop

The final conceptual architecture is:

```text
Observe World
     ↓
Build / Update Policy
     ↓
Generate Candidate Improvements
     ↓
Cheap Proxy Verification
     ↓
Estimate Update Footprint
     ↓
PIVOT
     ↓
Selective Paired Interventional Evaluation
     ↓
Estimate True Improvement
     ↓
Promote Best Update
     ↓
Deploy
     ↓
World Responds
     ↓
Competitors May Adapt
     ↓
Observe Consequences
     ↓
Next Improvement Round
```

Mathematically:

[
\boxed{
\pi_t
\rightarrow
{\pi'*{t,j}}
\rightarrow
\Delta*{\mathrm{proxy}}
\rightarrow
\Delta_{\mathrm{actor}}
\rightarrow
\Delta_{\mathrm{strategic}}
\rightarrow
\pi_{t+1}.
}
]

---

# 40. Final Instruction to Codex

Implement this project as a sequence of falsifiable scientific milestones.

Do not optimize for demo quality.

Do not optimize for system complexity.

Do not assume the central hypothesis is true.

Do not fabricate missing results.

Do not add LLMs, large world models, LOB complexity, or multi-agent ecology before the core estimand has been validated.

At every stage ask:

1. What is the exact policy transition?
2. What world produced the proxy improvement?
3. What world defines the higher-fidelity improvement?
4. Is evaluation paired?
5. What part of the error is explained by update footprint?
6. What part is explained by environment response?
7. Does strategic adaptation add a distinct effect?
8. Would an alternative policy-level evaluator already solve the problem?
9. Does expensive evaluation actually change update selection?
10. Is PIVOT reducing true update-selection regret per unit HF budget?

The first objective is not to prove PIVOT works.

The first objective is to determine whether **Improvement Fidelity is a real, distinct, measurable problem**.

Only after that is established should the project optimize PIVOT.

The final scientific target is:

[
\boxed{
\textbf{Understand and improve how self-improving agents measure progress when their progress changes the world.}
}
]
