# ICLR 2027 Submission Checklist

Updated 2026-08-20. This checklist distinguishes local evidence from gates
that cannot be closed inside the repository.

## Machine-verified

- PASS: official `iclr2027_conference` style files and recorded hashes.
- PASS: double-blind PDF metadata and official anonymous header.
- PASS: 9 main-text pages, references and appendix after the main-text gate.
- PASS: embedded fonts, non-empty text, no undefined citations, no overfull boxes.
- PASS: mandatory AI Use Statement, Reproducibility Statement, and Ethics Statement.
- PASS: deterministic supplementary ZIP with source, configs, tests, the
  controlled ImprovementBench release, snapshot, tables, and README.
- PASS: no author identity, private paths, credentials, raw vendor archives, or
  live-order artifacts in the submission package.
- PASS: repository tests, Ruff, and mypy are clean at the delivery snapshot.
- PASS (platform extension): IMPROVE-X contracts, ImprovementBench v1/v2, and
  the held-out controlled comparison are included in the anonymous supplement;
  they are not silently promoted into main-text results.

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
- OPEN: repeat the new held-out comparison over additional frozen seeds and
  rebuild paper tables/figures from a fresh, hash-bound confirmatory run before
  changing the submitted PDF.

## Decision

`CONDITIONAL GO` is the correct local conclusion. It means the paper can enter
final author review and rebuttal preparation, not that an OpenReview upload is
complete or that the external finance claim is causal.
