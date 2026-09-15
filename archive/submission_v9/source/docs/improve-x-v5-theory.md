# IMPROVE-X V5 Theory Notes

This note fixes the exact scope of the theory claims before they enter the
paper. Let `V` be a proxy world and `*` a deployment world. For an update
`tau=(pi,pi')`, write

```text
Delta_V = J_V(pi') - J_V(pi)
Delta_* = J_*(pi') - J_*(pi)
```

## 1. Global value fidelity is sufficient

If

```text
sup_pi |J_V(pi) - J_*(pi)| <= epsilon,
```

then `|Delta_V - Delta_*| <= 2 epsilon`. Therefore a non-tied update with
`|Delta_*| > 2 epsilon` has the correct sign in the proxy. This is a
sufficient condition, not a necessary one: a policy-independent offset can
be arbitrarily large while cancelling in every paired difference.

## 2. Ranking wording must be precise

The phrase “ranking fidelity is insufficient” is only defensible for a finite,
noisy, aggregate, or distribution-shifted ranking metric. If a verifier
preserves the exact pairwise order of every policy in the same deployment
world, then every non-tied pairwise difference has the same sign by definition.
That exact premise cannot be used to prove a reversal.

The valid claim is instead:

> High policy-level rank correlation on a sampled policy set does not imply
> local update sign consistency for candidate transitions generated later or
> for policies whose deployment changes the world.

A minimal counterexample is a training set containing policies `a,b` with
`J_V(a)>J_V(b)` and `J_*(a)>J_*(b)`, while an unseen candidate `c` has
`J_V(c)>J_V(b)` but `J_*(c)<J_*(b)`. The sampled ranking is perfect, yet the
update `b -> c` reverses. The missing assumption is coverage of the actual
update distribution and response world, not the ranking operation itself.

## 3. Response-footprint bound

Let `d(pi,pi')` be an update footprint and suppose the environment response
obeys `D(M[pi'],M[pi]) <= L_M d(pi,pi')`. If the value functional is
`L_J`-Lipschitz in its world argument, then

```text
|Delta_actor - Delta_direct| <= L_J L_M d(pi,pi').
```

This is a local diagnostic bound. It does not assume or imply that strategic
best responses are globally Lipschitz. In a competitive world the relevant
response can instead be summarized by an empirical strategic sensitivity,
which must be reported with its intervention model and confidence interval.

## 4. What the experiments may claim

- Controlled worlds may establish a mechanism-level reversal and test the
  footprint/sensitivity relationship.
- ImprovementBench measures sign, ranking, and explanation tasks on explicit
  transition rows; its scores are not external validity.
- Observational order-book/depth data cannot identify endogenous replenishment
  or strategic adaptation without an intervention design.
- PIVOT-X is an acquisition heuristic whose decision-change scores must be
  evaluated against matched-budget baselines; implementation is not evidence
  of superiority.
