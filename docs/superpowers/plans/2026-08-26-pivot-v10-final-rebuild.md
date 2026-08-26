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

- [ ] Record the commit, PDF/supplement hashes, V9 manifests, source tables, and all current figure inputs.
- [ ] Audit proposition count, version/experiment language, numbers, references, metric definitions, figure references, page budget, and double-blind metadata.
- [ ] Trace CISR/ISR/CTI implementations and document units and aggregation; add FER only as a transparent secondary cross-environment summary where identifiable.

### Task 2: Rebuild figures from frozen sources

**Files:**
- Create: `paper/figures/v10_style.py`
- Create: `experiments/v10/figures.py`
- Create: `figures/v10/fig1_improvement_reversal.*` through `figures/v10/figE_finance_boundary.*`
- Create: `V10_FIGURE_AUDIT.md`

- [ ] Build Figures 1--5 to the V10 scientific questions and fixed-K/paired uncertainty rules.
- [ ] Rebuild Figure 6 as the learned OOD null forest/effect figure.
- [ ] Rebuild Figure 7 as the three-panel strategic distribution, opponent-family forest, and adaptation-trajectory figure.
- [ ] Rebuild Figure 8 as posterior-sample/cost-misspecification robustness with uncertainty.
- [ ] Rebuild Figure 9 as a finance boundary/observer-to-actor-to-strategic evidence figure using only existing public evidence.
- [ ] Export source rows and metadata for every plot; enforce shared visual identity and grayscale distinguishability.

### Task 3: Rewrite manuscript

**Files:**
- Modify: `paper/iclr2027/main.tex`
- Modify: `paper/iclr2027/v9_results_macros.tex` or replace with V10 macros
- Modify: `paper/iclr2027/references.bib`

- [ ] Rewrite abstract and related work around Improvement Fidelity, operator-relative shift, SEAL, Self-Improvement Reversal, and AI4AI-Bench.
- [ ] Correct proposition count to six and remove internal state labels from prose/captions.
- [ ] Align all results, captions, terminology, limitations, and conclusion with frozen evidence and scoped PIVOT-VOI claims.
- [ ] Move historical ablation detail and engineering ladder material to the supplement.

### Task 4: Unified finalization and gates

**Files:**
- Create: `experiments/v10/audit_numbers.py`
- Create: `experiments/v10/finalize.py`
- Create: `V10_CLAIM_AUDIT.md`
- Create: `V10_BIBLIOGRAPHY_AUDIT.md`
- Create: `V10_REVIEWER_ATTACK_AUDIT.md`
- Create: `V10_FINAL_REPORT.md`

- [ ] Run source validation, metric-scale, figure, number, claim, bibliography, LaTeX, PDF metadata, page, grayscale, and supplement sanitization checks.
- [ ] Verify figure-only reading test and all V10 acceptance gates.
- [ ] Return `READY_FOR_SUBMISSION`, `READY_WITH_MINOR_MANUAL_CHECKS`, or `NOT_READY` with evidence and residual manual gates.

### Task 5: Release

- [ ] Rebuild the submission PDF and supplement from the curated tree.
- [ ] Review the diff and path boundary; exclude caches, credentials, raw vendor data, and expanded build trees.
- [ ] Commit and push the curated `research/pivot` tree, then verify remote commit, files, hashes, and final status.
