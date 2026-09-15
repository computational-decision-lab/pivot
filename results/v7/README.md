# V7 Result Artifacts

Each experiment directory contains its frozen metrics, state, provenance,
trajectory summaries, and a manifest. Row-level transitions are stored as
`transition_rows.jsonl.gz` to keep the public release within repository file
limits. Restore a specific stream with `gzip -dk transition_rows.jsonl.gz`;
the manifest records both compressed and uncompressed SHA-256 values.

The combined, leakage-safe public release is
`benchmarks/improvementbench/v7/transitions.jsonl.gz`. The discrete E3b
observer failure is retained as a separate `DESIGN_INVALID` audit artifact and
is not used as confirmatory evidence.
