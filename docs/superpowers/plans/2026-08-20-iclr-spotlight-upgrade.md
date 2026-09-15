# ICLR Spotlight-Level Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** Upgrade the anonymous ICLR 2027 PIVOT paper to the supplied Goal v3 standard while preserving its title, central thesis, conservative claims, and reproducible release gates.

**Architecture:** Keep the existing `paper/iclr2027/main.tex` and frozen snapshot contract as the publication surface. Add a small deterministic E4 evaluator-contrast module and figure rendering path, then revise the paper source and verifier invariants around the transition estimand. Rebuild all derived PDF/supplement artifacts from scripts and push only the project subtree.

**Tech Stack:** Python 3.10, NumPy, Matplotlib/Pillow fallback renderer, existing PIVOT transfer models, LaTeX/pdfLaTeX/latexmk, pytest, Ruff, mypy.

## Global Constraints

- Preserve the title and central thesis exactly; the paper studies policy-transition fidelity, not isolated policy value.
- Do not claim universal failure, causal market reversal, simulator ground truth, general equilibrium, or PIVOT universal dominance.
- Keep controlled mechanism evidence, strategic fixture evidence, and observational finance boundary evidence separate.
- Retain the finance nulls (0/7 primary, 0/5 holdout), all limitations, and open external scientific gates.
- No credentials, private paths, raw vendor archives, live orders, or unverified data may enter the package.
- Main paper text must remain at most nine pages and anonymous.

### Task 1: Add the design and audit hooks

**Files:**
- Create: `docs/superpowers/specs/2026-08-20-iclr-spotlight-upgrade-design.md`
- Create: `docs/superpowers/plans/2026-08-20-iclr-spotlight-upgrade.md`
- Modify: `tests/unit/test_iclr_submission.py`

- [ ] Add source-token tests for the exact opening concept, four contribution labels, decision-preservation proposition, active-learning distinction, stress-test heading, and finance nulls.
- [ ] Run the focused test and record the expected failure before source changes.

### Task 2: Implement the evaluator-contrast experiment

**Files:**
- Create: `experiments/e4_value_vs_improvement.py`
- Create: `src/pivot/transfer/evaluator_contrast.py`
- Create: `tests/unit/test_evaluator_contrast.py`
- Modify: `scripts/make_paper_figures.py`
- Modify: `tests/integration/test_figure_reproduction.py`

- [ ] Define a typed deterministic evaluator-contrast result with policy-value MAE/rank and transition IDE/ISC/IRR/ISR.
- [ ] Use only frozen E4 held-out rows and explicit bias/noise parameters; never overwrite source labels.
- [ ] Emit `comparison.json`, `rows.csv`, and a central PNG/CSV figure from one command.
- [ ] Test deterministic output, disjoint source rows, and metric schema.

### Task 3: Revise the paper narrative and theory

**Files:**
- Modify: `paper/iclr2027/main.tex`
- Modify: `paper/iclr2027/references.bib` only if a needed active-learning citation already has verified local metadata

- [ ] Rewrite the abstract into the four required paragraphs.
- [ ] Sharpen the introduction around the replacement operation and “rank policies correctly while ranking improvements incorrectly.”
- [ ] Change contribution framing to four ordered contributions.
- [ ] Add explicit related-work distinctions for policy evaluation, world-model evaluation, active learning, and performative prediction.
- [ ] Add the decision-preservation proposition and proof after the existing propositions.
- [ ] Add “Why Transition Validation Differs from Active Learning” under PIVOT.
- [ ] Rename the finance section and preserve observational/non-causal wording and nulls.
- [ ] Add the new evaluator-contrast result and central figure caption without inflating claims.

### Task 4: Upgrade figure rendering and source artifact contract

**Files:**
- Modify: `scripts/make_paper_figures.py`
- Modify: `paper/snapshot/figures/fig1_when_better_gets_worse.png`
- Modify: `paper/snapshot/figures/fig1_when_better_gets_worse.csv`
- Modify: `paper/snapshot/figures/fig2_reversal_phase_diagram.png`
- Modify: `paper/snapshot/figures/fig2_reversal_phase_diagram.csv`
- Add: evaluator-contrast figure/source rows in `paper/snapshot/figures/`

- [ ] Render Figure 1 with labeled quadrants and a highlighted reversal region.
- [ ] Render Figure 2 with a continuous heatmap, colorbar, and zero-reversal contour.
- [ ] Render the evaluator-contrast panel from the experiment output.
- [ ] Update figure validation and snapshot hashes after regeneration.

### Task 5: Rebuild and verify the deliverable

**Files:**
- Modify: `paper/iclr2027/verification.json`
- Modify: `paper/iclr2027/submission_verification.json`
- Modify: `paper/iclr2027/pivot_iclr2027_submission.pdf`
- Modify: `paper/iclr2027/pivot_iclr2027_supplementary.zip`
- Modify: `docs/improve-x-v5-status.md`

- [ ] Run the new E4 experiment, figure/table/snapshot generation, paper build, full pytest, Ruff, and mypy.
- [ ] Run PDF page/anonymity/font/reference checks and the ICLR package verifier.
- [ ] Confirm the new source tokens and no forbidden claims.
- [ ] Commit with scoped messages and push the project-root subtree to GitHub without force.
- [ ] Verify the remote commit and final tree hash.
