# V7 Power Analysis Contract

The primary closed-loop effect is

```text
Delta_CTI = CTI(PIVOT-VOI) - CTI(Proxy Only).
```

The confirmatory minimum practically important effect is `5.0` CTI units,
two-sided alpha is `0.05`, and target power is `0.80`. Development estimates
the standard deviation of paired trajectory-level `Delta_CTI`; the primary
confirmatory count uses the registered normal clustered-mean rule in
`required_cluster_samples`. The explicit sub-Gaussian best-update rule remains
reported as a conservative secondary diagnostic, not as the number of
independent trajectories. If the planned count is insufficient, the
confirmatory count may increase before freezing, but the environment,
operators, budget, stopping rule, and metric remain unchanged.

All transition-level observations within a trajectory are clustered. Final
intervals use trajectory-level bootstrap or clustered standard errors, and
strategic intervals additionally cluster on opponent family and opponent
seed. No transition is treated as an independent trajectory.
