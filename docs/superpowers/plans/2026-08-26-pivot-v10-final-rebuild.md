# PIVOT V10 Final Rebuild Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with verification checkpoints. Preserve the frozen scientific record and do not tune confirmatory experiments.

**Goal:** Convert the frozen V9 ICLR package into a submission-quality V10 package with an auditable narrative and publication-grade main and appendix figures, including redesigned Figures 6--9.

**Architecture:** Snapshot the current package first. Add a V10 analysis/figure layer that reads only frozen V9 source tables and emits vector figures, source tables, and metadata. Update the LaTeX manuscript to use scientific terminology and the new figure system, then run unified audits and build the final PDF and supplement.

**Tech Stack:** Python 3, pandas/NumPy/SciPy/Matplotlib, PyArrow, LaTeX/latexmk, existing ICLR style, shell verification scripts.

## Global Constraints

- Confirmatory seeds, environments, operators, response parameters, budgets, proxy definitions, and frozen MPE2 null are immutable.
- No new environment or favorable-result research cycle is permitted.
- Main text contains no V7/V8/V9/V10 engineering language or E2C/E3C/E4C/E5C/E7C identifiers except artifact navigation in supplementary material.
- PIVOT-VOI is the proposed acquisition rule; PIVOT-H remains a baseline.
- Every figure has PDF, SVG, PNG, CSV/Parquet source, and metadata JSON.
- The final PDF must satisfy the ICLR main-text page limit and the supplementary package must remain sanitized.

### Task 1: Freeze and audit

**Files:**
- Create: `snapshot/v10_pre_final/`
- Create: `V10_PRE_FINAL_AUDIT.md`
- Create: `V10_METRIC_SCALE_AUDIT.md`

- [x] Record the commit, PDF/supplement hashes, V9 manifests, source tables, and all current figure inputs.
- [x] Audit proposition count, version/experiment language, numbers, references, metric definitions, figure references, page budget, and double-blind metadata.
- [x] Trace CISR/ISR/CTI implementations and document units and aggregation; add FER only as a transparent secondary cross-environment summary where identifiable.

### Task 2: Rebuild figures from frozen sources

**Files:**
- Create: `paper/figures/v10_style.py`
- Create: `experiments/v10/figures.py`
- Create: `figures/v10/fig1_improvement_reversal.*` through `figures/v10/figE_finance_boundary.*`
- Create: `V10_FIGURE_AUDIT.md`

- [x] Build Figures 1--5 to the V10 scientific questions and fixed-K/paired uncertainty rules.
- [x] Rebuild old Figure 6 as the world-response decomposition (`figA_response_footprint`) with paired paths, effect distributions, and a reversal plane.
- [x] Rebuild old Figure 7 as the powered OOD null (`figB_learned_ood_null`) with forest/effect and paired raw scatter views.
- [x] Rebuild old Figure 8 as posterior-sample/cost-misspecification robustness (`figC_posterior_robustness`).
- [x] Rebuild old Figure 9 as strategic generalization (`figD_strategic_distribution`) with opponent-family distributions, forest, and reversal plane; retain finance as `figE_finance_boundary`.
- [x] Export source rows and metadata for every plot; enforce shared visual identity and grayscale distinguishability.

### Task 3: Rewrite manuscript

**Files:**
- Modify: `paper/iclr2027/main.tex`
- Modify: `paper/iclr2027/v9_results_macros.tex` or replace with V10 macros
- Modify: `paper/iclr2027/references.bib`

- [x] Rewrite abstract and related work around Improvement Fidelity, operator-relative shift, SEAL, Self-Improvement Reversal, and AI4AI-Bench.
- [x] Correct proposition count to six and remove internal state labels from prose/captions.
- [x] Align all results, captions, terminology, limitations, and conclusion with frozen evidence and scoped PIVOT-VOI claims.
- [x] Move historical ablation detail and engineering ladder material to the supplement.

### Task 4: Unified finalization and gates

**Files:**
- Create: `experiments/v10/audit_numbers.py`
- Create: `experiments/v10/finalize.py`
- Create: `V10_CLAIM_AUDIT.md`
- Create: `V10_BIBLIOGRAPHY_AUDIT.md`
- Create: `V10_REVIEWER_ATTACK_AUDIT.md`
- Create: `V10_FINAL_REPORT.md`

- [x] Run source validation, metric-scale, figure, number, claim, bibliography, LaTeX, PDF metadata, page, grayscale, and supplement sanitization checks.
- [x] Verify figure-only reading test and all V10 acceptance gates.
- [x] Return `READY_FOR_SUBMISSION`, `READY_WITH_MINOR_MANUAL_CHECKS`, or `NOT_READY` with evidence and residual manual gates.

### Task 5: Release

- [x] Rebuild the submission PDF and supplement from the curated tree.
- [x] Review the diff and path boundary; exclude caches, credentials, raw vendor data, and expanded build trees.
- [x] Commit and push the curated `research/pivot` tree, then verify remote commit, files, hashes, and final status.

## Completion record (2026-08-26)

- Finalizer: `.venv/bin/python -m experiments.v10.finalize --root .`
- Local machine result: `READY_WITH_MINOR_MANUAL_CHECKS`; main text 7 pages, PDF 12 pages.
- PDF SHA-256: `84d3489463af42e5ea322ebdb2379a58a5ac72779be4eb7ce1e4f88cbca8252c`.
- Supplement SHA-256: `e8369baa3794ca1fce983fb9aa3f22810885a0cda90d18573b54a1f06a9bf32a`.
- Verification: 205 pytest tests passed; Ruff and mypy passed (176 files); ZIP integrity and remote hash parity passed.
- GitHub: the standalone curated release chain is verified on `origin/master`; the artifact commit is `0790025f5939d24ce8bac712d3c5681cfc868fdc` and later commits only close release documentation.
- Remaining gates are author-side OpenReview metadata/quota/conflicts/parallel-submission checks and the explicitly bounded external interactive/strategic scientific validation; no causal finance or equilibrium claim is made.
