# Reproducibility Contract

## Run Artifacts

Every run persists:

```text
config snapshot
master and derived seeds
train/validation/test IDs
git commit
dependency versions
environment version
dataset/version ID
timestamp in UTC
machine information
HF budget and cost ledger
raw transition rows
processed metric tables
failed/discarded run records
SHA-256 manifest
```

Experiment tables use Parquet. Configuration and provenance use YAML/JSON. Figures are generated only from processed tables and each figure has an adjacent source-data table.

## Determinism and Pairing

Paired rollouts share initial state, scenario, exogenous seed/path, market day/order flow, and opponent initialization whenever supported. Train, validation, and test transition IDs are disjoint. Repeating the same config and seed must produce identical deterministic artifacts or a documented stochastic tolerance.

## Failure Policy

Fail loudly when a world, dataset, model, or fidelity component is unavailable. Never substitute a dataset, drop a failed transition, reduce simulator realism, or relabel a proxy world as ground truth without an explicit config/version change.

## Release Verification

Before a result is cited:

1. Validate the manifest.
2. Reproduce aggregates from raw rows.
3. Reproduce figures from processed tables.
4. Check seed and budget equality.
5. Confirm gate status and evidence links.
6. Audit for credentials, private logs, raw vendor L2, and live-order artifacts.
