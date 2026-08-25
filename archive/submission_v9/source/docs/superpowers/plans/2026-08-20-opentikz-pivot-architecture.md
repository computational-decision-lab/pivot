# OpenTikZ PIVOT Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy pinned OpenTikZ tooling and add a verified, editable PIVOT architecture figure to the anonymous ICLR package.

**Architecture:** A lock-backed bootstrap installs OpenTikZ read-only under `.tools/`. A standalone project copy of the `system-block-diagram` template renders PDF/SVG artifacts, which are hash-bound into the paper snapshot and included as Figure 3.

**Tech Stack:** Python standard library, Git, OpenTikZ, TikZ/LaTeX, `latexmk`, `dvisvgm`, pytest.

## Global Constraints

- Keep the paper title and central thesis unchanged.
- Preserve nine main pages and all conservative finance/scientific boundaries.
- Use OpenTikZ Mode A; never edit the upstream checkout.
- Use only `otblue`, `otorange`, `otteal`, `otpurple`, and `otgray` in figure content.
- Keep the figure standalone-compilable and parameterized with semantic node names.
- Never publish `.tools/`, credentials, local paths, raw vendor data, or build caches.

---

### Task 1: Pinned OpenTikZ Bootstrap

**Files:**
- Create: `configs/tooling/opentikz.json`
- Create: `scripts/bootstrap_opentikz.py`
- Modify: `.gitignore`
- Test: `tests/unit/test_opentikz_architecture.py`

**Interfaces:**
- Consumes: repository URL, commit, file hashes from the lock.
- Produces: `bootstrap_opentikz(lock_path: Path, destination: Path) -> dict[str, str]`.

- [x] Write a local-git-fixture test that expects exact detached checkout reuse and drift refusal.
- [x] Run the focused test and confirm it fails because the module does not exist.
- [x] Implement lock parsing, exact fetch/checkout, and hash validation without resetting an existing drifted checkout.
- [x] Run the focused test and confirm it passes.
- [x] Bootstrap `.tools/opentikz` and verify its HEAD and source hashes.

### Task 2: Editable OpenTikZ Architecture

**Files:**
- Create: `paper/iclr2027/figures/fig3_pivot_architecture.tex`
- Create: `paper/iclr2027/figures/fig3_pivot_architecture.meta.json`
- Create: `paper/iclr2027/figures/fig3_pivot_architecture.pdf`
- Create: `paper/iclr2027/figures/fig3_pivot_architecture.svg`
- Test: `tests/unit/test_opentikz_architecture.py`

**Interfaces:**
- Consumes: OpenTikZ `system-block-diagram` edit contract.
- Produces: standalone TikZ source and vector render artifacts.

- [x] Add contract tests for source template provenance, semantic nodes, named colors, parameter macros, and standalone class.
- [x] Run the tests and confirm they fail because the figure is absent.
- [x] Copy and adapt the template into the project with the two-row PIVOT feedback layout.
- [x] Compile PDF with fixed `SOURCE_DATE_EPOCH` and render SVG through OpenTikZ's renderer.
- [x] Inspect the SVG/PDF and run the contract tests to green.

### Task 3: Snapshot And Paper Integration

**Files:**
- Modify: `scripts/freeze_paper_snapshot.py`
- Modify: `paper/iclr2027/main.tex`
- Modify: `paper/iclr2027/build.sh`
- Modify: `paper/iclr2027/README.md`
- Modify: `scripts/build_iclr_supplement.py`
- Modify: `tests/integration/test_paper_snapshot.py`
- Modify: `tests/unit/test_iclr_submission.py`

**Interfaces:**
- Consumes: architecture source/metadata/PDF/SVG.
- Produces: Figure 3 in the main paper and four hash-indexed snapshot artifacts.

- [x] Add failing tests for snapshot inclusion, Figure 3 placement, appendix relocation, and build verification.
- [x] Extend the freezer with an `architecture_root` input and four declared files.
- [x] Insert and reference the method architecture after Figure 2; move the observer/actor diagnostic to the appendix.
- [x] Rebuild the snapshot, paper, and supplement.
- [x] Confirm main text remains at most nine pages and local submission checks remain green.

### Task 4: Final Release Verification

**Files:**
- Modify: `docs/implementation-status.md`
- Modify: `docs/improve-x-v5-status.md`
- Modify: generated paper/snapshot/release artifacts.

**Interfaces:**
- Consumes: final source and release artifacts.
- Produces: reproducible GitHub release commit and verified remote tree.

- [x] Run `pytest`, Ruff, mypy, standalone TikZ compile, forced deterministic PDF rebuild, archive integrity, secret scan, and private-path scan.
- [x] Record final PDF, supplement, and snapshot hashes without changing `CONDITIONAL GO` gates.
- [x] Commit only `research/pivot`, create a subtree split, push non-force to the PIVOT GitHub repository, and verify remote hashes/tree.
