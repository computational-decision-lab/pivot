# ImprovementBench v2 Design

**Status:** approved extension of the user-provided IMPROVE-X Goal V5 on
2026-08-20

## Purpose

ImprovementBench v1 is a deliberately small, single-round fixture. Version 2
adds the minimum controlled coverage needed for confirmatory transition-level
studies: multiple update operators, sequential incumbents, and frozen
train/validation/test splits. It remains a synthetic performative benchmark;
it does not establish external market impact or a finance result.

## Fixed release shape

The v2 configuration contains one deterministic seed per split, three rounds,
three operators, and three registered footprint scales. Every candidate is
evaluated in observer, actor, and strategic worlds using matched contexts.
This yields exactly 243 rows:

```
3 splits x 3 rounds x 3 operators x 3 scales x 3 worlds = 243 rows
```

The operators are `synthetic`, `rl-update`, and `evolutionary-mutation`.
Candidates from all three operators share one ranking pool for the same
trajectory, round, incumbent, and world. Ranking must therefore not group by
operator.

## Collection protocol

Each split follows a sequential controlled trajectory. Candidate generation is
seeded and the actor-world oracle selects the incumbent for the next collection
round. This is only a deterministic data-collection policy. Each row records
`collection_promotion_world`, `collection_selected`, and the selected source
transition ID so no downstream user can mistake the release for a PIVOT-X
evaluation.

The frozen `v1` files are never regenerated or modified. Version 2 has its own
config, README, JSONL, metadata, and manifest. Both releases retain the same
row schema because their serialized fields are compatible.

## API and evaluator changes

`ImprovementBenchDataset` exposes `split_names` and `rows_for_split(split)`.
The evaluator command accepts an optional `--split`, validates the full frozen
release before filtering, and reports the requested split alongside the source
manifest hash. Unknown or empty splits are errors.

Candidate ranking uses this key:

```
(world_level, trajectory_id, round_id, incumbent_policy)
```

For legacy rows without a `trajectory_id`, the seed is the deterministic
fallback. This preserves v1 behavior while allowing v2 to compare candidates
across operators.

## Integrity and boundaries

- Rows retain nulls for worlds that were not observed.
- Every row keeps source transition identity, operator scale, split, and world
  level in hash-covered JSONL.
- The release contains no credentials, private data, live trading, or LLM
  calls.
- `v2` is a controlled engineering artifact. Paper claims remain conditional
  until held-out PIVOT-X comparisons and independently calibrated worlds pass.

## Verification

Tests must prove the deterministic row count, split membership, three operator
labels, three sequential rounds, per-round nine-candidate cross-operator
ranking pools, split filtering, manifest validation, and v1 compatibility.
The release is generated only after the builder and evaluator pass from the
committed source revision.
