# PIVOT V9 Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a statistically rigorous V9 PIVOT package that tests operator-relative improvement fidelity across independent adaptive mechanisms while preserving the frozen V7 baseline and all null results.

**Architecture:** Add a versioned `src/pivot/v9` layer over the existing transition/evaluation contracts. All V9 runs write isolated artifacts under `results/v9`, figures/tables under `figures/v9` and `tables/v9`, and auditable manifests/decisions. Rewrite the paper only after confirmatory artifacts are frozen.

**Tech Stack:** Python 3.10, NumPy, PyYAML, Matplotlib, optional scikit-learn/pyarrow with deterministic standard-library fallbacks, LaTeX/pdfLaTeX, pytest, Ruff, mypy, SHA-256 manifests.

## Global Constraints

- Preserve `snapshot/v9_preupgrade` and never overwrite V7 results, figures, PDF, supplement, Binance audit, or claims.
- The scientific object is the replacement transition `pi -> pi'`, with explicit nulls for unavailable causal layers.
- Every experiment ends in exactly one registered scientific state.
- Confirmatory seeds, candidate generators, environments, horizons, proxy definitions, and metrics are frozen before confirmatory execution.
- Use grouped/hierarchical uncertainty; transitions within one trajectory are not independent.
- No credentials, live execution, capital authorization, raw vendor data, M3, LLM, or claimed market ground truth.
- No numerical value is manually typed into generated figures or result tables.
- Main PDF remains anonymous and at most nine main-text pages.

---

### Task 1: Freeze V7 baseline and create V9 design registry

**Files:**
- Create: `scripts/freeze_v9_baseline.py`
- Create: `snapshot/v9_preupgrade/README.md`
- Create: `snapshot/v9_preupgrade/manifest.json`
- Create: `configs/v9/profiles.yaml`
- Create: `research/claims_v9.yaml`
- Create: `research/experiment_registry_v9.yaml`
- Create: `research/research_state_v9.json`
- Test: `tests/unit/test_v9_contracts.py`

- [ ] Record the current parent commit, paper hashes, result hashes, and tracked source paths in a deterministic manifest.
- [ ] Refuse to overwrite an existing snapshot whose manifest differs.
- [ ] Define smoke/dev/confirmatory profiles and the H1--H5 registry before running confirmatory experiments.
- [ ] Test snapshot idempotence, profile validation, required terminal states, and null preservation.

### Task 2: Implement independent V9 environments and operators

**Files:**
- Create: `src/pivot/v9/environments.py`
- Create: `src/pivot/v9/operators.py`
- Create: `src/pivot/v9/schema.py`
- Create: `tests/unit/test_v9_environments.py`
- Create: `tests/unit/test_v9_operators.py`

- [ ] Implement observer, actor, and strategic paired rollouts for performative control and congestion/resource response.
- [ ] Implement local, gradient-informed, and population proposal families with frozen shift levels.
- [ ] Verify response signal, reward non-ceiling, candidate diversity, reproducibility, and absence of candidate-specific punishment.

### Task 3: Build the unified V9 runner and raw artifact ledger

**Files:**
- Create: `experiments/v9/__init__.py`
- Create: `experiments/v9/run.py`
- Create: `experiments/v9/validate.py`
- Create: `experiments/v9/artifacts.py`
- Create: `scripts/run_v9.py`
- Create: `tests/integration/test_v9_runner.py`

- [ ] Support `smoke`, `dev`, and `confirmatory` profiles with per-seed isolation, checkpoint/resume, and failed-seed ledger rows.
- [ ] Emit the complete V9 transition schema, config/provenance snapshots, compressed raw rows, metrics, decision, and manifest.
- [ ] Implement E2C and E3C first, preserving frozen MPE2 as an explicit null input.

### Task 4: Implement E2C operator-shift scaling and statistics

**Files:**
- Create: `experiments/v9/e2c_operator_shift.py`
- Create: `src/pivot/v9/statistics.py`
- Create: `tests/integration/test_v9_e2c.py`
- Create: `configs/v9/e2c.yaml`

- [ ] Run 3 environment families, 3 operator families, 10 frozen shift levels, and confirmatory independent seeds.
- [ ] Compute chi-square/MMD/ESS diagnostics where defined, global value metrics, operator-relative IDE/ISC/IRR/ISR, and bound diagnostics.
- [ ] Produce scientific decision and hierarchical bootstrap evidence without treating transitions as independent seeds.

### Task 5: Implement E3C multi-environment closed-loop comparison

**Files:**
- Create: `experiments/v9/e3c_closed_loop.py`
- Create: `src/pivot/v9/methods.py`
- Create: `configs/v9/e3c.yaml`
- Create: `tests/integration/test_v9_e3c.py`

- [ ] Preserve the V7 MPE2 null and add performative-control and congestion/resource worlds with shared candidate sets across methods.
- [ ] Compare Proxy Only, Global Value, Random HF, Top Proxy HF, Uncertainty HF, Paired LUCB, Global-VOI, PIVOT-H, PIVOT-VOI, and All-HF.
- [ ] Report final deployment value, CISR, CTI, HF cost, harmful promotion, and time-to-target with hierarchical intervals.

### Task 6: Implement E4C learned evaluator OOD and calibration

**Files:**
- Create: `experiments/v9/e4c_learned_ood.py`
- Create: `src/pivot/v9/evaluators.py`
- Create: `configs/v9/e4c.yaml`
- Create: `tests/integration/test_v9_e4c.py`

- [ ] Implement matched-evidence Bayesian linear and nonlinear bootstrap ensemble global/differential evaluators.
- [ ] Enforce trajectory, environment, operator, and response-regime holdouts without row leakage.
- [ ] Emit MAE/RMSE/rank, IDE/ISC/IRR/ISR, NLL/Brier/coverage/reliability, and downstream selection metrics.

### Task 7: Implement E5C evidence-efficiency and posterior robustness

**Files:**
- Create: `experiments/v9/e5c_efficiency.py`
- Create: `experiments/v9/calibration.py`
- Create: `configs/v9/e5c.yaml`
- Create: `tests/integration/test_v9_e5c.py`

- [ ] Compare all required acquisition methods across K in {4,8,16} and fixed HF budgets including zero and all-HF.
- [ ] Measure EVSI ranking stability for posterior/hypothetical sample grids and cost-misspecification sensitivity.
- [ ] Report Pareto non-dominance rather than assuming PIVOT-VOI dominates.

### Task 8: Implement E7C multi-family strategic adaptation

**Files:**
- Create: `experiments/v9/e7c_strategic.py`
- Create: `src/pivot/v9/opponents.py`
- Create: `configs/v9/e7c.yaml`
- Create: `tests/integration/test_v9_e7c.py`

- [ ] Implement fixed, reactive, finite-step best-response, gradient-adaptive, and RL/evolutionary adaptive opponents.
- [ ] Use held-out opponent seeds/initialization/strengths and cluster-level inference.
- [ ] Report direct/actor/strategic effects, SIRR, focal CTI/CISR, opponent reward changes, and policy shifts.

### Task 9: Rebuild figures, tables, and audits

**Files:**
- Create: `scripts/build_v9_figures.py`
- Create: `scripts/build_v9_tables.py`
- Create: `scripts/audit_v9_figures.py`
- Create: `scripts/audit_v9_claims.py`
- Create: `scripts/audit_v9_statistics.py`
- Create: `V9_RESULTS_REPORT.md`
- Create: `V9_FAILURE_LEDGER.md`
- Create: `V9_STATISTICAL_AUDIT.md`
- Create: `V9_FIGURE_AUDIT.md`
- Create: `V9_CLAIM_AUDIT.md`
- Create: `V9_REPRODUCIBILITY.md`

- [ ] Generate vector PDF/SVG, PNG, CSV/Parquet source, and metadata JSON for each main/appendix figure.
- [ ] Replace the old Figure 2 with E2C, add multi-environment trajectory and evidence-efficiency figures, and demote raw heatmaps to the appendix.
- [ ] Verify readability at 100/75/50%, grayscale distinguishability, source hashes, bootstrap grouping, and claim scope.

### Task 10: Rewrite and rebuild the anonymous submission

**Files:**
- Modify: `paper/iclr2027/main.tex`
- Modify: `paper/iclr2027/references.bib`
- Modify: `paper/iclr2027/submission_checklist.md`
- Modify: `scripts/reproduce_paper.py`
- Modify: `scripts/verify_iclr_submission.py`
- Create: `paper/iclr2027/supplementary/v9/`

- [ ] Rewrite only after V9 decisions are frozen, retaining V7 nulls and narrowing claims to supported scopes.
- [ ] Add E2C/E3C/E4C/E5C/E7C figures and generated tables without manual numbers.
- [ ] Rebuild the PDF/supplementary ZIP and pass anonymity, page, font, citation, overfull-box, manifest, and clean-clone checks.

### Task 11: Final release verification

- [ ] Run unit/integration tests, Ruff, mypy, all smoke checks, and paper rebuild.
- [ ] Verify every V9 manifest and decompressed raw table against SHA-256.
- [ ] Review the claim registry against paper result sentences and preserve all unsupported/null findings.
- [ ] Commit only the curated project subtree and push only the verified release artifacts.
