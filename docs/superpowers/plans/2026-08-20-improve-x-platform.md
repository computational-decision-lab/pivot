# IMPROVE-X Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable IMPROVE-X platform layer for self-improvement operators, multi-round trajectories, ImprovementBench, three-world fidelity records, and explicit failure taxonomy while preserving the existing PIVOT implementation.

**Architecture:** `improve_x` is a thin platform facade over the existing typed `pivot` contracts. Operators produce immutable `CandidateBatch` objects; `ImprovementTrajectory` records every round; `ImprovementBenchDataset` stores transition-level rows with null-preserving JSONL and hashes; failure labels and fidelity tasks consume those rows without changing existing experiment semantics.

**Tech Stack:** Python 3.10+, frozen dataclasses, NumPy-backed existing worlds, JSONL/JSON SHA-256 manifests, pytest, Ruff, mypy.

## Global Constraints

- Preserve `pivot` imports and all existing paper claims.
- Never substitute missing actor/strategic values for true values; keep `None`.
- Keep every candidate row, including failed, tied, and unselected transitions.
- Use seeded deterministic generation and matched contexts.
- No credentials, live orders, private data, LLM dependency, or new causal claim.
- New benchmark artifacts must include schema version, source commit, row count, and file hashes.

---

### Task 1: Platform core contracts

**Files:**
- Create: `src/improve_x/__init__.py`
- Create: `src/improve_x/core/operator.py`
- Create: `src/improve_x/core/trajectory.py`
- Create: `tests/unit/test_improve_x_core.py`

**Interfaces:**
- `ImprovementOperator` protocol with `propose(incumbent: Policy, round_id: int | str, seed: int, **kwargs) -> CandidateBatch`.
- `CandidateBatch(incumbent, candidates, operator, round_id, seed)`.
- `ImprovementTrajectory.append_round(batch, selected_index, values, query_cost) -> None`.
- `ImprovementTrajectory.cumulative_true_improvement` and `to_records()`.

- [x] **Step 1: Write failing tests** for batch incumbent validation, deterministic round IDs, trajectory promotion, proxy/true curves, and null preservation.
- [x] **Step 2: Run** `.venv/bin/pytest -q tests/unit/test_improve_x_core.py` and observe import failure.
- [x] **Step 3: Implement** the frozen dataclasses and protocol by adapting existing `PolicyTransition` objects; reject empty batches, invalid selected indices, and mismatched incumbents.
- [x] **Step 4: Run** focused tests, Ruff, and mypy.
- [x] **Step 5: Commit** as part of the IMPROVE-X platform checkpoint.

### Task 2: Failure taxonomy and fidelity hierarchy

**Files:**
- Create: `src/improve_x/failures/taxonomy.py`
- Create: `src/improve_x/metrics/fidelity.py`
- Create: `tests/unit/test_improve_x_fidelity.py`

**Interfaces:**
- `FailureType` enum and `classify_failure(delta_proxy, delta_actor, delta_strategic, trajectory=None) -> FailureType`.
- `compute_layer_fidelity(rows) -> Mapping[str, float | int | None]` returning observer/actor/strategic sign consistency and reversal rates.

- [x] **Step 1: Write failing tests** for observer, environment-response, strategic, drift, none, and unknown precedence.
- [x] **Step 2: Run focused tests and confirm missing imports.**
- [x] **Step 3: Implement** deterministic classification and null-aware layer metrics using existing improvement metrics.
- [x] **Step 4: Verify** all taxonomy cases and the existing metric suite.
- [x] **Step 5: Commit** as part of the IMPROVE-X platform checkpoint.

### Task 3: ImprovementBench dataset and tasks

**Files:**
- Create: `src/improve_x/benchmark/dataset.py`
- Create: `src/improve_x/benchmark/tasks.py`
- Create: `tests/unit/test_improvementbench.py`

**Interfaces:**
- `ImprovementBenchRow.from_transition(record, world_level) -> ImprovementBenchRow`.
- `ImprovementBenchDataset.append(row)`, `.write(directory)`, `.read(directory)`, `.validate()`.
- `evaluate_sign_task(rows, predictions)`, `evaluate_ranking_task(rows, scores)`, `evaluate_explanation_task(rows, labels)`.

- [x] **Step 1: Write failing tests** for JSONL round trip, explicit nulls, grouped candidate ranking, sign accuracy, explanation macro accuracy, and tampered manifest rejection.
- [x] **Step 2: Run focused tests and observe missing modules.**
- [x] **Step 3: Implement** a versioned schema, canonical JSON serialization, deterministic hash manifest, and three benchmark task evaluators.
- [x] **Step 4: Run unit tests and a temporary round-trip fixture.**
- [x] **Step 5: Commit** as part of the IMPROVE-X platform checkpoint.

### Task 4: Controlled benchmark and trajectory commands

**Files:**
- Create: `scripts/build_improvementbench.py`
- Create: `scripts/run_improvement_trajectory.py`
- Create: `configs/improve_x/benchmark.yaml`
- Create: `configs/improve_x/trajectory.yaml`
- Create: `tests/integration/test_improve_x_commands.py`
- Create: `benchmarks/improvementbench/v1/README.md`

**Interfaces:**
- `build_improvementbench.py --config ... --output ...` creates `transitions.jsonl`, `metadata.json`, and `manifest.json`.
- `run_improvement_trajectory.py --config ... --output ...` creates `trajectory.json`, `rounds.jsonl`, and `provenance.json`.

- [x] **Step 1: Write integration tests** invoking both commands in `tmp_path`, asserting deterministic hashes, required columns, multiple rounds, and retained failures.
- [x] **Step 2: Run tests and confirm command failures.**
- [x] **Step 3: Implement** seeded generation from `PerformativeWorld` + `SyntheticPerturbation`; evaluate observer/actor/strategic modes with paired contexts; preserve all rows.
- [x] **Step 4: Run the commands and copy only the small, hash-bound benchmark artifact into `benchmarks/improvementbench/v1/`; keep generated large outputs ignored.
- [x] **Step 5: Commit** as the next artifact checkpoint after generation.

### Task 5: Documentation and final verification

**Files:**
- Modify: `README.md`
- Modify: `docs/implementation-status.md`
- Create: `docs/improve-x-v5-status.md`
- Modify: `paper/iclr2027/submission_checklist.md`

- [x] **Step 1: Document** the new platform commands, schema, and explicit boundary that no V5 benchmark result is a paper claim yet.
- [x] **Step 2: Run** `.venv/bin/pytest -q`, `.venv/bin/ruff check .`, `.venv/bin/mypy src scripts`, both commands, paper verifier, and supplementary auditor.
- [x] **Step 3: Inspect** benchmark manifest, trajectory JSON, PDF verification, and git diff.
- [x] **Step 4: Commit** `docs: record improve-x v5 platform status`.
