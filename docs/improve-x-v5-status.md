# IMPROVE-X V5 Status

Updated 2026-08-20. This page records the implementation state of the
user-provided CODEX MASTER GOAL V5. It is an engineering and evidence ledger,
not a claim that the ICLR submission has been uploaded.

## Platform now implemented

- `ImprovementOperator` and immutable `CandidateBatch` contracts wrap the
  existing `PolicyTransition` schema.
- `ImprovementTrajectory` records every candidate across multiple rounds,
  preserves unselected rows, and exposes proxy, actor, strategic, and true
  cumulative curves.
- `ImprovementBench v1` stores `(pi_t, pi_t+1)` rows as canonical JSONL with
  explicit nulls, schema version, metadata, file hashes, and validation.
- `ImprovementBench v2` adds three seeded operators, three sequential rounds,
  and frozen train/validation/test splits in a separate 243-row release. Its
  ranking task pools candidates across operators within one trajectory round.
- A held-out comparison runner evaluates `proxy_only`, `random_hf`,
  `top_proxy_hf`, and `pivot_x` under a matched two-query-per-round budget.
  The frozen test diagnostic contains 108 candidate records and a
  hash-bound query ledger; it is not yet a paper-level superiority claim.
- Sign, candidate-ranking, and failure-explanation task evaluators are
  available in `src/improve_x/benchmark/tasks.py`.
- Failure labels distinguish observer, environment-response, strategic, and
  optimization-drift failures when the corresponding layers are observed.
- Synthetic and legacy RL operators remain available through the PIVOT
  adapter; `EvolutionaryMutation` adds a seeded offline population operator.
- `select_pivot_x` and `score_decision_preservation` expose an auditable,
  cost-normalized decision-change query rule over the existing PIVOT round
  harness.

## Reproducible commands

```bash
.venv/bin/python scripts/build_improvementbench.py \
  --config configs/improve_x/benchmark.yaml \
  --output /tmp/improvementbench-v1

.venv/bin/python scripts/run_improvement_trajectory.py \
  --config configs/improve_x/trajectory.yaml \
  --output /tmp/improve-x-trajectory

.venv/bin/python scripts/evaluate_improvementbench.py \
  --input benchmarks/improvementbench/v1 \
  --output /tmp/improvementbench-v1-metrics
```

The first command emits 12 controlled rows (2 seeds x 2 candidates x 3
world levels). The second emits four rounds and 12 retained candidate rows.
The third reports the proxy sign/ranking baseline, oracle explanation sanity,
and layer fidelity without changing the dataset. All use matched contexts and a seeded controlled performative world. The
checked-in benchmark release is generated only after the full verification
run; generated temporary directories should not be treated as paper evidence.

Build the multi-round, multi-operator release with:

```bash
.venv/bin/python scripts/build_improvementbench.py \
  --config configs/improve_x/benchmark_v2.yaml \
  --output /tmp/improvementbench-v2

.venv/bin/python scripts/evaluate_improvementbench.py \
  --input benchmarks/improvementbench/v2 \
  --output /tmp/improvementbench-v2-metrics \
  --split test
```

This emits 243 rows (three splits x three rounds x three operators x three
scales x three worlds). The actor-oracle promotion is recorded only as a
collection policy; it is not a PIVOT-X superiority result.

Run the held-out comparison with:

```bash
.venv/bin/python scripts/run_improve_x_comparison.py \
  --config configs/improve_x/comparison.yaml \
  --benchmark benchmarks/improvementbench/v2 \
  --output /tmp/improve-x-comparison
```

The checked-in comparison artifact is under
`benchmarks/improvementbench/v2/comparison/`; its `manifest.json` binds the
result files to both the benchmark manifest and source commit.

## Evidence boundary

The controlled runner demonstrates that the platform can represent and audit
proxy, actor, and strategic deltas. It does not establish external causal
market impact, realistic equilibrium, or performance of Databento/Binance
data. No live orders, credentials, raw vendor L2, or LLM calls are required.
The existing PIVOT paper remains `CONDITIONAL GO` pending author checks and
the open scientific gates in the ICLR checklist.

## Verification checkpoint

At the 2026-08-20 checkpoint:

- `.venv/bin/pytest -q`: **114 passed**;
- `.venv/bin/ruff check .`: **clean**;
- `.venv/bin/mypy src scripts`: **clean**;
- ImprovementBench v1: **12 rows**, manifest validation **true**;
- ImprovementBench v2: **243 rows**, three frozen splits, manifest validation
  **true**, and nine cross-operator ranking groups per split;
- held-out comparison: **108 rows**, four methods, three rounds, and matched
  query ledgers with manifest validation **true**;
- trajectory smoke: **4 rounds / 12 retained rows**. The controlled run ended
  with proxy `2.5269201087`, actor `-3.0549010603`, and strategic
  `-5.1170298273`. These values are fixture diagnostics only;
- anonymous ICLR PDF: **9 main pages / 10 total**, local decision
  **CONDITIONAL GO**. PDF SHA-256:
  `ab0db0c44228df22a5bea6d7f1897d57ac7ada632ad5b4d6d45f9dc1bbd973af`.

## Remaining V5 work

1. Repeat the held-out comparison over additional frozen seeds and confirm the
   update rule before promoting any method-level number into the paper.
2. Add independently calibrated interactive and strategic response worlds;
   observational depth data cannot identify those responses.
3. Evaluate PIVOT-X against the existing matched-budget baselines on held-out
   ImprovementBench rounds; implementation alone is not a superiority claim.
4. Rebuild the anonymous PDF only after the new claims and tables are backed by
   regenerated artifacts.
