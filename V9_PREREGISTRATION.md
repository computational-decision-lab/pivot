# PIVOT V9 Preregistration

Version `v9.0`, frozen before confirmatory execution. Configuration hashes are
stored in each experiment's `provenance.json` and `scientific_decision.json`.

## Primary estimand

The unit is the policy replacement transition `pi -> pi'`. For a proxy world
`V` and deployment world `*`,

`Delta_V = J_V(pi') - J_V(pi)` and `Delta_* = J_*(pi') - J_*(pi)`.

Primary outcomes are IDE, ISC, IRR, ISR/CISR, CTI, and HF cost. Causal layers
remain null when the environment does not define them.

## Confirmatory design

- E2C: three environment families, three operator families, ten frozen shift
  levels, eight-candidate batches, and 30 independent seeds.
- E3C: two newly implemented worlds (performative-control and
  congestion/resource), 40 rounds, 8 candidates per round, 30 independent
  seeds, and the ten registered acquisition methods. The powered V7 MPE2
  result is retained as a frozen external null panel; it is not rerun or
  relabeled as a V9 observation.
- E4C: trajectory, environment, operator, and response-regime holdouts with
  matched training evidence for Bayesian linear and bootstrap ensemble global
  and differential evaluators.
- E5C: candidate counts `{4, 8, 16}`, budgets including zero and all-HF, fixed
  high-fidelity costs, EVSI sample-grid stability, and cost misspecification.
- E7C: fixed, reactive, finite-step best-response, gradient-adaptive, and
  RL/evolutionary opponent mechanisms with held-out seeds and cluster-level
  inference.

## Decision rule

Every run emits one terminal state. `UNDERPOWERED` is a valid outcome and is
not promoted to a positive or negative scientific claim. A positive claim is
allowed only within the registered environments, operators, budgets, splits,
and opponent families. No universal PIVOT superiority claim is permitted.

## Multiplicity and uncertainty

Seeds are the independent unit. Bootstrap intervals resample seed or trajectory
clusters, never individual transitions as if independent. The primary family
uses the registered Holm correction; secondary diagnostics are labelled
descriptive.
