# E7b Confirmatory Result

This frozen result evaluates one focal self-improving policy against two
opponent families in version-pinned Farama MPE2. Family A is the development
reactive-observation family; family B is held out and uses an explicit
finite-action best response. The pool uses 180 pre-registered opponent seeds,
three adaptation strengths, and eight typed candidate updates per seed.

The strategic test has 140 valid family-B opponent-seed clusters and requires
135 under the registered clustered-mean power rule. Its state is
`HYPOTHESIS_SUPPORTED`: the family-B cluster-level strategic effect is
`-0.5884429`, below the frozen `-0.5` threshold; the actor-positive row-level
strategic effect is `-0.5987190` with bootstrap interval
`[-0.7404119, -0.4617742]`. The result supports strategic improvement
reversal for this held-out opponent family only. It does not establish an
equilibrium result, a realistic market ecology, or universal strategic
failure.
