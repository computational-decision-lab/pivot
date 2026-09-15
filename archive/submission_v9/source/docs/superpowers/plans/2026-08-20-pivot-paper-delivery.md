# PIVOT Paper Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** Produce an anonymous, reproducible PIVOT working-paper PDF and a source-data snapshot from the already frozen P0-P9 and twelve-ablation evidence.

**Architecture:** A tracked `paper/` package contains the LaTeX source, bibliography, build script, and figure manifest. A small snapshot script copies only verified JSON/CSV/PNG artifacts from the clean-room roots and records SHA-256 values; the paper reads those stable files rather than `/tmp` paths. The PDF is built with `latexmk` using the standard `article` class and a nine-page main-text budget, with supplementary limitations and reproducibility material after the main text.

**Tech Stack:** Python 3.10, existing PIVOT scripts, NumPy/Matplotlib-generated PNG/CSV figures, LaTeX/pdfLaTeX, BibTeX, `pdfinfo`, `pdftotext`, Ghostscript raster checks.

## Global Constraints

- Preserve the frozen estimand: policy transition `pi -> pi'`, not action authorization or policy ranking alone.
- Do not add Databento or Binance data unless a reproducible, non-secret source is required; current Binance snapshot is observational and virtual-fill only.
- Do not call the public depth proxy causal endogenous-response ground truth.
- Do not claim E3 overoptimization: the recorded E3 fixture has proxy and true values moving together and must be reported as a null/limitation.
- Keep LLM/EvoQuant/M3 adapters deferred.
- Main text target is at most nine pages; appendix material may follow but must not hide core claims.
- No credentials, raw vendor L2, live orders, or private logs enter the paper package.
- Every copied artifact gets a SHA-256 entry and source commit in a manifest.

## Tasks

### Task 1: Freeze the paper artifact snapshot

**Files:**
- Create: `paper/snapshot/`
- Create: `paper/manifest.json`
- Create: `scripts/freeze_paper_snapshot.py`
- Test: `tests/integration/test_paper_snapshot.py`

Copy the seven validated figures and source CSVs from the clean-room figure root, the current E4/E5/P2/E6/F/E9 summaries, the twelve-ablation aggregate, public expansion summary, and the relevant evidence markdown into `paper/snapshot/`. Refuse missing files, unexpected hashes, or a non-empty destination. Write source roots, source commits, SHA-256 values, and claim-boundary flags.

### Task 2: Generate paper figures and tables

**Files:**
- Create: `paper/figures/`
- Create: `paper/tables/`
- Create: `paper/figure_manifest.json`
- Modify: `scripts/freeze_paper_snapshot.py`

Use the existing canonical figure generator against the frozen clean-room source. Add compact tables for the controlled gates, budget frontier, strategic contrast, public audit, and all twelve ablations. Keep null results and confidence intervals visible.

### Task 3: Write the anonymous paper source

**Files:**
- Create: `paper/main.tex`
- Create: `paper/references.bib`
- Create: `paper/build.sh`
- Create: `paper/README.md`

Sections: abstract, introduction, problem/estimand, improvement fidelity metrics, PIVOT, theory propositions with proofs, controlled experiments, finance testbed, ablations, limitations/claim boundary, reproducibility, conclusion. Use anonymous author metadata and no identifying acknowledgements. Cite only sources whose URLs/metadata are present in the repository or official public-data documentation.

### Task 4: Compile and audit the PDF

**Files:**
- Create: `paper/build/`
- Create: `paper/verification.json`
- Create: `paper/preview.png`

Run `latexmk -pdf`, `pdfinfo`, `pdftotext`, font inspection, page-count checks, duplicate/undefined-reference scans, and Ghostscript rasterization. Fail if the main text exceeds nine pages, if figures are missing, if metadata reveals author identity, or if the PDF has zero/near-zero raster content.

### Task 5: Reproducibility handoff

**Files:**
- Create: `docs/paper-delivery-2026-08-20.md`
- Modify: `README.md`, `docs/implementation-status.md`

Record exact build command, source commit, artifact hashes, page count, remaining scientific limitations, and the distinction between a completed working-paper PDF and a submission-ready external claim.
