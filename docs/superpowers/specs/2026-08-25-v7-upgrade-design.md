# PIVOT V7 Upgrade Design

## Goal

Upgrade the V9 controlled-mechanism paper into an auditable study of
operator-relative Improvement Fidelity, with a principled PIVOT-VOI acquisition
rule and independently validated adaptive/strategic experiments. Valid nulls
remain publishable outcomes; only invalid or underpowered designs may be
redesigned.

## Frozen scientific object

Every result is indexed by a policy transition
`tau = (pi_t, pi_{t+1})`. The proxy and deployment-induced improvements are
`Delta_V = J_V(pi') - J_V(pi)` and
`Delta_* = J(pi'; M[pi']) - J(pi; M[pi])`. Improvement Reversal is the event
`Delta_V > 0` and `Delta_* < 0`.

An improvement operator induces a transition law `Q_A`. For any transition
loss `L`, the estimand is

```text
IF(V, A; L) = E_{tau ~ Q_A}[L(Delta_V(tau), Delta_*(tau))].
```

Global Fidelity Blindness remains an impossibility result: arbitrarily small
global policy-value MAE and near-perfect Spearman ranking do not rule out
`IRR(Q_A)=1` for an adaptively concentrated operator. The new Operator Shift
Bound complements it. If `Q_A << P`, `w=dQ_A/dP`, and
`ell(tau)=L(Delta_V, Delta_*)`, then

```text
IF(V, A; L) <= sqrt(E_P[ell^2]) * sqrt(1 + chi2(Q_A || P)).
```

The response-footprint result remains mechanistic:
`|Delta_actor - Delta_direct| <= L_J L_M d(pi, pi')`.

## Components

1. `pivot.research.state` owns the explicit experiment-state machine and
   design/result classification. It accepts only the five V7 result classes
   and writes append-only decision records.
2. `pivot.theory.operator_shift` computes discrete density ratios, chi-square
   divergence, the second-moment bound, and the observed IF losses without
   claiming that the bound is causal evidence.
3. `pivot.acquisition.pivot_voi` owns a dependency-free Bayesian linear
   posterior over correction targets. It computes posterior expected simple
   regret, Monte-Carlo EVSI per cost, and the V7 stopping rule. Existing
   `select_pivot` is renamed/documented as PIVOT-H and remains a baseline.
4. `pivot.theory.sample_complexity` provides the explicit sub-Gaussian
   best-update identification bound and required sample-size calculation.
5. `experiments/e2_operator_shift.py` is a registered controlled diagnostic;
   it varies operator temperature beta and records distribution shift, IF,
   effective sample size, and construct-validity diagnostics.
6. `experiments/e3b_closed_loop.py`, `e4b_global_vs_transition.py`, and
   `e7b_external_strategic.py` are separate V7 experiment entry points. They
   reuse frozen transition/paired-evaluation contracts but cannot mutate the
   V9 fixture outputs.
7. `pivot.research.validity` implements the five E3b gates and classifies a
   run before any confirmatory result is consumed.
8. `pivot.benchmark.improvementbench_v7` exports leakage-safe transition
   records and group splits from the final pipeline.

## Data flow

```text
operator -> CandidateBatch -> proxy paired evaluation
         -> Q_A / shift features -> posterior correction model
         -> EVSI/cost acquisition -> paired HF intervention
         -> posterior update or stopping -> selected transition
         -> trajectory metrics and ImprovementBench record
```

All rollout comparisons are paired on initial state, exogenous draw, seed,
and (where applicable) opponent initialization. Trajectory-level bootstrap or
clustered uncertainty is mandatory for final claims.

## Integrity rules

- V9 is copied to `archive/submission_v9` before new outputs are generated.
- Development, validation, and confirmatory settings are separate and frozen
  before confirmatory outcomes are inspected.
- A design-invalid result can be redesigned only with a recorded construct
  validity reason. A powered confirmatory null is frozen as
  `HYPOTHESIS_NOT_SUPPORTED`.
- No result is promoted to an external claim merely because it comes from a
  public repository; the environment version, license, seed, and independent
  implementation boundary must be recorded.
- Finance remains a negative observational boundary test and is never tuned to
  create a reversal.

## Acceptance evidence

The V7 package is complete only when theory modules have unit tests, E2 has a
registered artifact, PIVOT-VOI has matched-cost decision-regret evidence, E3b,
E4b, and E7b each have a frozen validity/result classification, the final
ImprovementBench has leakage-safe splits, and `make reproduce-paper` verifies
all generated hashes and manuscript claim gates.
