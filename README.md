# IMPROVE-X / PIVOT

IMPROVE-X is the reusable research platform for **Improvement Fidelity**:
whether a self-improvement update that looks beneficial in a cheap or fixed
proxy world remains beneficial after deployment changes the environment and,
later, after other agents adapt. PIVOT is the paired, budgeted validation
method implemented on top of the platform.

```text
PIVOT = Paired Interventional Validation of Optimization Transitions
```

Working paper title: *When Better Gets Worse: Improvement Fidelity of Self-Improvement Operators in Adaptive Worlds*.

## Current Status

- Research specification: frozen and documented.
- Implementation: the original P0-P9 PIVOT harnesses are present, and the
  IMPROVE-X vertical slice provides operator batches, multi-round trajectories,
  ImprovementBench v1, world-layer fidelity metrics, and failure taxonomy.
  LLM/EvoQuant/M3 adapters remain deferred.
- Scientific gates A-F: fixture-level registered passes are recorded, but none
  are promoted to final paper claims. A checksum-bound public Binance execution
  audit across three assets and four calendar blocks is
  complete; it observed no depth-proxy reversal, and causal response plus
  external strategic validity remain open.
- Live trading or external execution: out of scope.

## IMPROVE-X quick start

Build the deterministic controlled benchmark:

```bash
.venv/bin/python scripts/build_improvementbench.py \
  --config configs/improve_x/benchmark.yaml \
  --output /tmp/improvementbench-v1
```

Run a four-round trajectory while retaining every candidate:

```bash
.venv/bin/python scripts/run_improvement_trajectory.py \
  --config configs/improve_x/trajectory.yaml \
  --output /tmp/improve-x-trajectory
```

Evaluate the frozen transition benchmark's sign, ranking, explanation, and
world-layer fidelity tasks:

```bash
.venv/bin/python scripts/evaluate_improvementbench.py \
  --input benchmarks/improvementbench/v1 \
  --output /tmp/improvementbench-v1-metrics
```

The stable platform contracts live under
`src/improve_x/`. `ImprovementBench` rows are transition-level records, not
causal claims about real markets; see
`docs/improve-x-v5-status.md` and
`benchmarks/improvementbench/v1/README.md`.

The v2 controlled release adds three operators, three sequential rounds, and
frozen train/validation/test splits:

```bash
.venv/bin/python scripts/build_improvementbench.py \
  --config configs/improve_x/benchmark_v2.yaml \
  --output /tmp/improvementbench-v2

.venv/bin/python scripts/evaluate_improvementbench.py \
  --input benchmarks/improvementbench/v2 \
  --output /tmp/improvementbench-v2-metrics \
  --split test
```

Its actor-oracle collection promotion is explicitly metadata, not a claim
that PIVOT-X has won a held-out comparison. See
`benchmarks/improvementbench/v2/README.md`.

The matched-budget trajectory diagnostic is reproducible with:

```bash
.venv/bin/python scripts/run_improve_x_comparison.py \
  --config configs/improve_x/comparison.yaml \
  --benchmark benchmarks/improvementbench/v2 \
  --output /tmp/improve-x-comparison
```

Its checked-in result is a controlled diagnostic with a hash-bound query
ledger, not a general superiority claim.

## ICLR 2027 paper and architecture figure

Build the anonymous submission package from the pinned snapshot:

```bash
(cd paper/iclr2027 && ./build.sh)
.venv/bin/python scripts/bootstrap_opentikz.py
.venv/bin/python scripts/build_opentikz_architecture.py
```

OpenTikZ is locked to commit
`359befbf8e8af7ce08e7e387b2c2a198e0ca735d`; its adapted architecture source
is `paper/iclr2027/figures/fig3_pivot_architecture.tex`, and the paper uses
the hash-bound copy under `paper/snapshot/figures/`.

Start with:

- `/opt/projects/research/pivot/docs/pivot.md`
- `/opt/projects/research/pivot/docs/master-goal.md`
- `/opt/projects/research/pivot/docs/superpowers/plans/2026-08-19-pivot-master-implementation.md`
- `/opt/projects/research/pivot/docs/experiments/gates.md`
- `/opt/projects/research/pivot/docs/experiments/registered-protocol.md`
- `/opt/projects/research/pivot/docs/experiments/registered-evidence-2026-08-19.md`
- `/opt/projects/research/pivot/docs/experiments/public-finance-evidence-2026-08-19.md`
- `/opt/projects/research/pivot/docs/experiments/clean-room-evidence-2026-08-19.md`
- `/opt/projects/research/pivot/docs/experiments/public-expansion-evidence-2026-08-19.md`

Run the controlled first milestone with:

```bash
python3 scripts/run_sweep.py --config configs/sweeps/p2.yaml --output results/raw/controlled-first
```

The command writes a Parquet/JSONL transition table, provenance, confidence
intervals, CSV source tables, and PNG diagnostics. Finance and strategic
commands are separate (`experiments/e6_finance_actor.py`, `e7`, `e8`, `e9`),
and all fills remain virtual. See `docs/implementation-status.md` for the
current gate-aware status and known limitations.

Acquire and run the frozen public finance audit with:

```bash
python scripts/fetch_public_finance.py \
  --manifest configs/data/binance_btcusdt_um_2023-01-01_07.yaml \
  --output-root data/public

python experiments/e6_public_calibration.py \
  --config configs/finance/e6_public_calibration.yaml \
  --output results/raw/e6-public-calibration
```

The acquisition command only reads the official Binance public archive and
verifies every file against its frozen SHA-256. The resulting depth world is
an observational execution proxy and is never labeled endogenous ground truth.

Run the frozen multi-asset expansion with:

```bash
.venv/bin/python experiments/e6_public_expansion.py \
  --config configs/finance/e6_public_expansion.yaml \
  --output results/raw/e6-public-expansion
```

Its 12 asset/date pairs, primary holdout, and acceptance rules are frozen in
`docs/experiments/public-expansion-protocol.md`; the observed result is in
`docs/experiments/public-expansion-evidence-2026-08-19.md`.
