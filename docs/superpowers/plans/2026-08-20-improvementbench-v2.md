# ImprovementBench v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze a deterministic multi-round, multi-operator ImprovementBench v2 release with split-aware transition tasks while preserving v1 exactly.

**Architecture:** Keep the current JSONL schema and create a second release directory. The builder gains a narrowly scoped `multiround_multioperator` mode; the dataset and evaluator learn frozen split filtering; candidate ranking pools across operator labels when transitions share a trajectory, round, incumbent, and world.

**Tech Stack:** Python 3.10+, PyYAML, frozen dataclasses, NumPy-backed controlled world, JSONL/JSON SHA-256 manifests, pytest, Ruff, mypy.

## Global Constraints

- Do not edit `benchmarks/improvementbench/v1/`.
- Preserve explicit nulls and all unselected candidate rows.
- Use only deterministic seeds and matched contexts.
- Mark actor-oracle collection promotion as collection metadata, never as PIVOT-X evidence.
- Keep schema version `improvementbench-v1`; v2 is a release version, not an incompatible row schema.
- Make no external finance, MARL-equilibrium, or method-superiority claim.

---

### Task 1: Split and ranking contracts

**Files:**
- Modify: `tests/unit/test_improvementbench.py`
- Modify: `tests/integration/test_improve_x_commands.py`
- Modify: `src/improve_x/benchmark/dataset.py`
- Modify: `src/improve_x/benchmark/tasks.py`
- Modify: `scripts/evaluate_improvementbench.py`

**Interfaces:**
- `ImprovementBenchDataset.split_names -> tuple[str, ...]`.
- `ImprovementBenchDataset.rows_for_split(split_name: str) -> tuple[ImprovementBenchRow, ...]`.
- `evaluate_ranking_task` groups v2 candidates by world, trajectory, round, and incumbent, not operator.
- `evaluate_improvementbench.py --split NAME` evaluates exactly the named frozen split.

- [x] **Step 1: Write failing tests** for split discovery/filtering, unknown split rejection, cross-operator nine-candidate group size, and CLI split metrics.

```python
assert dataset.split_names == ("test", "train", "validation")
assert len(dataset.rows_for_split("test")) == 81
assert evaluate_ranking_task(rows, scores)["n_groups"] == 9
```

- [x] **Step 2: Run focused tests** and confirm the missing split API or wrong grouping failure.

Run: `.venv/bin/pytest -q tests/unit/test_improvementbench.py tests/integration/test_improve_x_commands.py`

- [x] **Step 3: Implement the minimal contract** with a seed fallback for legacy rows and full-release validation before CLI filtering.

- [x] **Step 4: Run focused tests** and confirm all pass.

- [x] **Step 5: Commit** the tested API checkpoint (`87d63c0`).

### Task 2: Deterministic v2 builder and release

**Files:**
- Modify: `scripts/build_improvementbench.py`
- Create: `configs/improve_x/benchmark_v2.yaml`
- Modify: `tests/integration/test_improve_x_commands.py`
- Create: `benchmarks/improvementbench/v2/README.md`
- Create: `benchmarks/improvementbench/v2/transitions.jsonl`
- Create: `benchmarks/improvementbench/v2/metadata.json`
- Create: `benchmarks/improvementbench/v2/manifest.json`

**Interfaces:**
- `mode: multiround_multioperator` selects the v2 builder without changing default v1 behavior.
- The release contains 243 rows and all three operator/world labels.
- Each v2 row has split, trajectory, operator scale, source transition, and collection metadata.

- [x] **Step 1: Write the failing v2 integration test** asserting 243 rows, split count 81, three rounds, nine candidates per group, and deterministic manifests.

- [x] **Step 2: Run it** and confirm the current builder cannot create the v2 dataset.

- [x] **Step 3: Implement the builder mode** using seeded synthetic, RL-update, and evolutionary operators. Promote only the actor-oracle collection winner after recording every candidate/world row.

- [x] **Step 4: Run the v1 and v2 integration tests**, then generate v2 to a temporary directory.

- [x] **Step 5: Commit source** (`87d63c0`), regenerate the tracked v2 artifact from that commit, validate its manifest, and stage the frozen release.

### Task 3: Documentation, supplement, and verification

**Files:**
- Modify: `README.md`
- Modify: `docs/improve-x-v5-status.md`
- Modify: `paper/iclr2027/supplementary/scripts/build_iclr_supplement.py`
- Modify: `paper/iclr2027/supplementary/README.md`

- [x] **Step 1: Document v2 commands and the collection/evidence boundary.**

- [x] **Step 2: Add and freeze the held-out matched-budget comparison** with
  `proxy_only`, `random_hf`, `top_proxy_hf`, and `pivot_x`; retain its
  108-row ledger and source/benchmark hashes.

- [x] **Step 3: Include the hash-bound v2 release and comparison artifact in
  the anonymous supplement without bundling generated temporary outputs.**

- [x] **Step 4: Run** `.venv/bin/pytest -q`, `.venv/bin/ruff check .`, `.venv/bin/mypy src scripts`, v1/v2 builder and evaluator commands, supplement auditor, and PDF verifier.

- [x] **Step 5: Inspect** row counts, manifests, split metrics, package contents, PDF page count, and git diff.

- [x] **Step 6: Commit** the documentation and package refresh as the final
  local V5 checkpoint.
