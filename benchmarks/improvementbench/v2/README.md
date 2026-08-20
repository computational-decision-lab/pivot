# IMPROVE-X ImprovementBench v2

This directory contains the hash-bound, controlled multi-round release used
for transition-level development. It has 243 rows:

```
3 splits x 3 rounds x 3 operators x 3 scales x 3 worlds
```

The operators are `synthetic`, `rl-update`, and `evolutionary-mutation`.
Candidates from all operators share a ranking pool within a trajectory,
round, incumbent, and world. The split seeds are `train=31`,
`validation=41`, and `test=51`; each split contains 81 rows.

Build and evaluate it with:

```bash
.venv/bin/python scripts/build_improvementbench.py \
  --config configs/improve_x/benchmark_v2.yaml \
  --output /tmp/improvementbench-v2

.venv/bin/python scripts/evaluate_improvementbench.py \
  --input benchmarks/improvementbench/v2 \
  --output /tmp/improvementbench-v2-metrics \
  --split test
```

Rows preserve every candidate and record `trajectory_id`, `split`, operator
scale, source transition identity, and collection metadata. The actor-oracle
promotion is only a deterministic collection policy; it is not a PIVOT-X
result. This controlled release is not evidence of causal market impact,
strategic equilibrium, or paper-level method superiority.

`v1` remains a separate frozen release and is not overwritten by v2 builds.
