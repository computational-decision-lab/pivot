# ICLR 2027 Submission Checklist

Updated 2026-08-20. This checklist distinguishes local evidence from gates
that cannot be closed inside the repository.

## Machine-verified

- PASS: official `iclr2027_conference` style files and recorded hashes.
- PASS: double-blind PDF metadata and official anonymous header.
- PASS: 9 main-text pages, references and appendix after the main-text gate.
- PASS: embedded fonts, non-empty text, no undefined citations, no overfull boxes.
- PASS: mandatory AI Use Statement, Reproducibility Statement, and Ethics Statement.
- PASS: deterministic supplementary ZIP with source, configs, tests, snapshot,
  tables, and README.
- PASS: no author identity, private paths, credentials, raw vendor archives, or
  live-order artifacts in the submission package.
- PASS: repository tests, Ruff, and mypy are clean at the delivery snapshot.
- PASS (platform extension): IMPROVE-X contracts and controlled
  ImprovementBench/trajectory smoke tests are tracked separately from the
  frozen submission snapshot; they are not silently promoted into paper
  results.

## Manual author gates

- PENDING: every author has an OpenReview profile and the submission metadata is
  complete.
- PENDING: author quota and reciprocal-review obligations are satisfied.
- PENDING: conflicts, affiliations, acknowledgements, and final author order
  are checked without changing the anonymous PDF.
- PENDING: authors confirm no parallel-submission violation.

## Scientific gates

- OPEN: independently calibrated causal interactive response beyond the
  controlled fixture.
- OPEN: external strategic validation beyond the deterministic opponent fixture.
- OPEN: confirmatory update-generation rule, volatility/liquidity labels, and
  frozen holdout for any stronger external claim.
- OPEN: integrate the IMPROVE-X V5 platform artifact and rebuild all paper
  tables/figures from a fresh, hash-bound run before changing the submitted
  PDF.

## Decision

`CONDITIONAL GO` is the correct local conclusion. It means the paper can enter
final author review and rebuttal preparation, not that an OpenReview upload is
complete or that the external finance claim is causal.
