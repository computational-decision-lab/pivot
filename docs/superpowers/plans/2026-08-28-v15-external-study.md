# V15 External Self-Improvement Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the V15 external-study no-op boundaries with an auditable, runnable mini-SWE/Inspect pipeline while preserving sealed task planes, frozen protocol inputs, and honest terminal states.

**Architecture:** The project owns immutable policies, task planes, paired differences, promotion decisions, and canonical artifacts. Inspect AI is the control-plane logger/scorer; the pinned mini-SWE-agent runs each policy-task pair in a fresh Docker sandbox. Development probes are explicitly separate from confirmatory runs, and every confirmatory phase refuses to execute unless the pre-outcome lock and runtime manifest agree.

**Tech Stack:** Python 3.10 project environment, Python 3.11 external runtime, Inspect AI at the pinned source commit, mini-SWE-agent at the pinned source commit, Docker with an immutable image digest, LiteLLM-compatible model endpoint, JSON/CSV/Parquet canonical artifacts, pytest, Ruff, mypy, and LaTeX build/audit scripts.

## Implementation Status (2026-08-28)

The repository implementation for Tasks 1--6 is complete and verified.  The
pinned Inspect AI, mini-SWE-agent, and Pi runtimes are locally available; DEV
smoke, paired sandbox, candidate archive, promotion replay, closed-loop schema,
response audit, ablation, figure, PDF, and release checks all pass.  The
confirmatory protocol remains intentionally unopened: it requires the frozen
execution authorization, consumes the registered external model/container
budget, and has no DEV-to-confirmatory promotion path.

| Scope | Status | Evidence |
|---|---|---|
| Runtime, task planes, and lock | COMPLETE | `configs/v15/`, `experiments/v15/confirmatory_lock.json` |
| Inspect/mini-SWE DEV transition audit | COMPLETE (DEV only) | `results/v15/dev-external-transition-audit/manifest.json` |
| Paired promotion and closed-loop runners | COMPLETE (DEV only) | `results/v15/dev-external-promotion/manifest.json`, `results/v15/dev-external-closed-loop/manifest.json` |
| Pi adapter, Inspect bridge, and DEV replication | COMPLETE (DEV only) | `experiments/v15/run_pi_replication.py`, `results/v15/dev-pi-replication/manifest.json` |
| Identity-blind response and registered ablations | COMPLETE (DEV only) | `results/v15/dev-external-strategic-response/manifest.json`, `results/v15/dev-external-ablations/manifest.json` |
| Confirmatory scientific outcomes | NOT RUN | `V15_FINAL_REPORT.md` (preserved as a blocker) |

The local engineering terminal state is therefore reproducible but not a
substitute for the registered confirmatory evidence.

## Global Constraints

- Never alter a scientific configuration because an observed result is undesirable.
- Keep `D_proxy`, `D_gate`, and `D_assessment` disjoint and enforce role-based access.
- Treat trajectory or task cluster, not transition rows, as the independent unit.
- Do not persist credentials or return sealed task contents to the self-improver.
- Keep the existing pre-modern-agent snapshot immutable and usable as fallback.
- Do not expose the internal implementation-assistant token in reviewer-facing artifacts.
- Every completion statement requires fresh verification output.

### Task 1: External runtime and sealed task inputs

**Files:**
- Modify: `experiments/v15/external_runtime.py`
- Modify: `configs/v15/task_manifest.json`
- Modify: `configs/v15/confirmatory.yaml`
- Modify: `configs/v15/external_versions.json`
- Create: `configs/v15/external_runtime_requirements.txt`
- Test: `tests/unit/test_v15_external_runtime.py`

- [x] Add contract tests for task-family coverage, runtime settings validation, secret redaction, and immutable image metadata.
- [x] Run the focused tests and confirm the new contract tests fail for missing task/runtime inputs.
- [x] Implement the smallest runtime/config changes needed for the tests.
- [x] Run the focused tests, Ruff, and mypy.

### Task 2: Development external smoke

**Files:**
- Create: `experiments/v15/dev/external_smoke.py`
- Modify: `experiments/v15/__main__.py`
- Test: `tests/unit/test_v15_external_smoke.py`

- [x] Add a non-confirmatory smoke command that runs one proxy task through Inspect and mini-SWE in Docker.
- [x] Ensure it emits only DEV artifacts and records model/container counts without touching sealed assessment data.
- [x] Run the smoke with the pinned runtime and inspect its log and trajectory artifact.

### Task 3: Autonomous transition audit

**Files:**
- Create: `experiments/v15/external_study.py`
- Create: `experiments/v15/operators_external.py`
- Modify: `experiments/v15/commands.py`
- Modify: `experiments/v15/run_transitions.py`
- Test: `tests/unit/test_v15_external_study.py`

- [x] Add tests for operator output validation and transition rows that contain only pre-gate footprint features.
- [x] Implement two model-driven proposal operators with disjoint registered edit scopes and no hypothesis disclosure.
- [x] Implement a DEV-limited transition runner and a locked confirmatory runner using the same candidate archive format.
- [x] Run DEV transition smoke and verify no gate or assessment outcome is visible to operator prompts.

### Task 4: Paired promotion and closed loop

**Files:**
- Create: `experiments/v15/promotion.py`
- Modify: `experiments/v15/commands.py`
- Modify: `experiments/v15/run_promotion_replay.py`
- Modify: `experiments/v15/run_closed_loop.py`
- Modify: `experiments/v15/run_assessment.py`
- Test: `tests/unit/test_v15_promotion.py`

- [x] Add deterministic unit tests for Proxy, LUCB, Global-VOI, PIVOT-VOI, and All-HF selection orientation.
- [x] Implement paired hidden evaluation and update-selection regret with frozen candidate sets.
- [x] Implement closed-loop method trajectories and exactly-once terminal assessment access.
- [x] Run a DEV replay and validate canonical row counts and provenance.

### Task 5: Replication, response, and ablations

**Files:**
- Modify: `experiments/v15/adapters/pi.py`
- Modify: `experiments/v15/run_pi_replication.py`
- Modify: `experiments/v15/run_strategic.py`
- Modify: `experiments/v15/run_ablations.py`
- Test: `tests/unit/test_v15_replication.py`

- [x] Add capability checks and an explicit optional status for Pi when its runtime is unavailable.
- [x] Implement the frozen non-LLM response evaluator and paired/no-pairing/no-footprint ablations.
- [x] Run only registered DEV checks until primary mini-SWE outcomes are frozen.

### Task 6: Canonical artifacts, manuscript, and final audits

**Files:**
- Modify: `experiments/v15/canonical.py`
- Modify: `experiments/v15/reports.py`
- Modify: `paper/iclr2027/main.tex`
- Modify: `paper/iclr2027/README.md`
- Test: `tests/unit/test_v15_reports.py`

- [x] Generate canonical tables from frozen artifacts without hand-maintained scientific numbers.
- [x] Add the modern-agent section only after valid results exist, with claims bounded by terminal states; the current manuscript retains the pre-modern-agent fallback until confirmation is opened.
- [x] Render and inspect figures at print size and in paper context.
- [x] Run number, claim, reference, anonymity, language, reproducibility, and PDF audits.
