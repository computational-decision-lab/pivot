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

## Public Data Acquisition

Public finance inputs are acquired only from URLs frozen in a versioned
manifest. The downloader restricts hosts and paths, verifies the expected
SHA-256 before moving a temporary file into the cache, revalidates reused
files, and refuses to overwrite corrupt cache entries. Binary archives and
download receipts remain outside Git; the manifest, parser, assumptions, and
result hashes remain versioned.

The current manifest uses the official Binance public archive. Its one-minute
kline and percentage-depth files support an observational execution audit;
they do not identify counterfactual replenishment, post-trade response, hidden
liquidity, strategic adaptation, or causal impact recovery. Every such run must
persist `ground_truth_for_endogenous_response: false`.

Exploratory update selection and confirmatory evaluation must be separated. The
seven-session single-asset calibration and the 12-session three-asset
expansion both use the exploratory typed update. Any future confirmatory
asset/regime evaluation must freeze its update generator, asset list, regime
labels, participation grid, and holdout rule before inspecting outcomes.

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
7. Confirm that observational depth results are not labeled causal actor-world
   ground truth.
