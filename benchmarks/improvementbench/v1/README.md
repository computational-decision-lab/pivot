# IMPROVE-X ImprovementBench v1

This directory contains the small, hash-bound controlled release of the
IMPROVE-X benchmark. Build or verify it with:

```bash
.venv/bin/python scripts/build_improvementbench.py \
  --config configs/improve_x/benchmark.yaml \
  --output benchmarks/improvementbench/v1
```

Each row is one policy transition `(pi_t, pi_t+1)` at one world level. The
release contains observer, actor, and strategic rows for each candidate, with
explicit nulls where a layer is not observed. It is a controlled benchmark
artifact, not evidence of a causal external market claim.

Files:

- `transitions.jsonl`: canonical transition-level rows;
- `metadata.json`: schema, seed/config provenance, and row count;
- `manifest.json`: SHA-256 hashes for the data files.

Run `ImprovementBenchDataset.read(...).validate()` before consuming a copied
release. Do not overwrite a tracked release with a partial or failed run.
