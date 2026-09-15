# V7 Redesign Policy

Every experiment produces a state and an evidence record. The only allowed
states are `IMPLEMENTATION_FAILURE`, `DESIGN_INVALID`, `UNDERPOWERED`,
`HYPOTHESIS_SUPPORTED`, and `HYPOTHESIS_NOT_SUPPORTED`.

`IMPLEMENTATION_FAILURE` means a reproducible code, interface, numerical,
data-integrity, seed, or statistical error. Fix the implementation and rerun
the same registered design.

`DESIGN_INVALID` means the intervention cannot identify the stated quantity.
Examples include reward-ceiling saturation, no policy-dependent response,
degenerate candidates, identical proxy and deployment worlds, non-adaptive
opponents, or unstable measurement. Redesign is allowed only after recording
the failed gate, the reason it blocks identification, the changed component,
and the frozen components. The redesigned run uses development seeds.

`UNDERPOWERED` means the design is valid but its uncertainty cannot distinguish
the predeclared minimum effect. Increase sample size using the registered
power rule; do not change dynamics, candidate generation, or the primary
estimand.

`HYPOTHESIS_SUPPORTED` requires the frozen primary test, the predeclared alpha
and minimum effect, and clustered confidence intervals to pass.

`HYPOTHESIS_NOT_SUPPORTED` is a valid, powered confirmatory null. Never
redesign it to seek a preferred direction.

Finance is permanently `FROZEN_NEGATIVE` unless a separately approved causal
interactive source is added; public observational data cannot be relabeled as
endogenous response evidence.
