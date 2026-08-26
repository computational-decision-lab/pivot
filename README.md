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
- V7 scientific status: theory and operator-shift tests pass; powered E3b and
  E4b external comparisons are registered nulls; powered E7b supports
  strategic reversal for one held-out best-response opponent family. The
  Binance audit remains a negative, observational boundary (0/7 primary,
  0/5 holdout), not causal market evidence.
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

## V9 confirmatory package

V9 promotes the replacement transition `pi -> pi'` to the primary statistical
object. The registered experiments are E2C operator-shift scaling, E3C
closed-loop selection, E4C learned-evaluator OOD, E5C fixed-budget evidence
efficiency, and E7C strategic response. Results are isolated under
`results/v9/`; the V7 boundary is preserved in `snapshot/v9_preupgrade/`.

The confirmatory workflow is:

```bash
.venv/bin/python -m experiments.v9.run --experiment e2c --profile confirmatory --output results/v9/e2c-confirmatory --root .
.venv/bin/python -m experiments.v9.run --experiment e3c --profile confirmatory --output results/v9/e3c-confirmatory --root .
.venv/bin/python -m experiments.v9.run --experiment e4c --profile confirmatory --output results/v9/e4c-confirmatory --root .
.venv/bin/python -m experiments.v9.run --experiment e5c --profile confirmatory --output results/v9/e5c-confirmatory --root .
.venv/bin/python -m experiments.v9.run --experiment e7c --profile confirmatory --output results/v9/e7c-confirmatory --root .
make v9-analyze v9-figures v9-tables v9-audit
```

Each run emits a terminal scientific state, compressed raw transition rows,
provenance, a failure ledger, and a SHA-256 manifest. `UNDERPOWERED` and
`HYPOTHESIS_NOT_SUPPORTED` are retained as valid outcomes. PIVOT is not claimed
to dominate every acquisition method, and All-HF is an oracle reference only.

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

- `docs/pivot.md`
- `docs/master-goal.md`
- `docs/superpowers/plans/2026-08-19-pivot-master-implementation.md`
- `docs/experiments/gates.md`
- `docs/experiments/registered-protocol.md`
- `docs/experiments/registered-evidence-2026-08-19.md`
- `docs/experiments/public-finance-evidence-2026-08-19.md`
- `docs/experiments/clean-room-evidence-2026-08-19.md`
- `docs/experiments/public-expansion-evidence-2026-08-19.md`

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
