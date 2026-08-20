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
```

The first command emits 12 controlled rows (2 seeds x 2 candidates x 3
world levels). The second emits four rounds and 12 retained candidate rows.
Both use matched contexts and a seeded controlled performative world. The
checked-in benchmark release is generated only after the full verification
run; generated temporary directories should not be treated as paper evidence.

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
- trajectory smoke: **4 rounds / 12 retained rows**. The controlled run ended
  with proxy `2.5269201087`, actor `-3.0549010603`, and strategic
  `-5.1170298273`. These values are fixture diagnostics only;
- anonymous ICLR PDF: **9 main pages / 10 total**, local decision
  **CONDITIONAL GO**. PDF SHA-256:
  `ab0db0c44228df22a5bea6d7f1897d57ac7ada632ad5b4d6d45f9dc1bbd973af`.

## Remaining V5 work

1. Add confirmatory multi-round operator comparisons and held-out seeds before
   promoting ImprovementBench numbers into the paper.
2. Add independently calibrated interactive and strategic response worlds;
   observational depth data cannot identify those responses.
3. Evaluate PIVOT-X against the existing matched-budget baselines on held-out
   ImprovementBench rounds; implementation alone is not a superiority claim.
4. Rebuild the anonymous PDF only after the new claims and tables are backed by
   regenerated artifacts.
