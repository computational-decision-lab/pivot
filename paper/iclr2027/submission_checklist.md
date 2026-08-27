# ICLR 2027 Submission Checklist

Updated 2026-08-26. This checklist distinguishes local evidence from gates
that cannot be closed inside the repository.

## Machine-verified

- PASS: official `iclr2027_conference` style files and recorded hashes.
- PASS: double-blind PDF metadata and official anonymous header.
- PASS: the final verifier reports at most 9 main-text pages; references and
  appendix begin after the main-text gate.
- PASS: embedded fonts, non-empty text, no undefined citations, no overfull boxes.
- PASS: mandatory AI Use Statement, Reproducibility Statement, and Ethics Statement.
  The AI disclosure names language editing, code and figure-formatting assistance,
  limited experimental-infrastructure assistance, and literature discovery; it
  also assigns verification and responsibility to the authors.
  This follows the ICLR 2027 policy checked on 2026-08-27, which requires
  disclosure both in the manuscript and in the submission form.
- PASS: deterministic supplementary ZIP with source, configs, tests, the
  controlled ImprovementBench release, snapshot, tables, and README.
- PASS: no author identity, private paths, credentials, raw vendor archives, or
  live-order artifacts in the submission package.
- PASS: repository tests (`205 passed`), Ruff, and mypy are clean at the
  delivery snapshot.
- PASS (platform extension): IMPROVE-X contracts, ImprovementBench v1/v2, and
  the held-out controlled comparison are included in the anonymous supplement;
  they are not silently promoted into main-text results.
- PASS (V6 theory artifact): Global Fidelity Blindness and
  Response-Footprint Sensitivity checks include registered configuration,
  row-level data, figures, provenance, and a SHA-256 manifest in the
  supplementary archive.
- PASS (V9 frozen evidence): E2C, E3C, E4C, E5C, and E7C have isolated
  configurations, terminal scientific states, independent-seed accounting,
  paired transition streams or explicitly typed OOD/strategic artifacts, and
  regenerated source tables/figures. The powered E4C null is retained.
- PASS (frozen baseline): tracked V7 blobs verify against the recorded pre-upgrade
  commit; ignored LaTeX/supplement derivatives are explicitly classified as
  derived in `artifacts/v9/baseline_verification.json`.

## Manual author gates

- PENDING: every author has an OpenReview profile and the submission metadata is
  complete.
- PENDING: author quota and reciprocal-review obligations are satisfied.
- PENDING: conflicts, affiliations, acknowledgements, and final author order
  are checked without changing the anonymous PDF.
- PENDING: authors confirm no parallel-submission violation.
- PENDING: the same AI-use disclosure is copied accurately into the submission
  form, as required by the ICLR policy.

## Scientific evidence boundaries

- FROZEN NULL: E3b passes all five construct gates but its powered 772-paired-
  trajectory comparison does not support a PIVOT-VOI CTI advantage; this is the
  external MPE2 panel carried into E3C, not a relabeled V9 rerun.
- FROZEN NULL: E4b's powered trajectory-disjoint comparison does not support
  a transition-over-global advantage.
- FROZEN SCOPED PASS: E7b has 140 held-out opponent-seed clusters (required
  135) and supports strategic reversal for the registered best-response
  family; this is not a general equilibrium or market-validity claim.
- OPEN BOUNDARY: independently calibrated causal market response, realistic
  ecology, and any stronger finance claim remain outside this package.

## Decision

`READY_FOR_SUBMISSION` is the package-level conclusion after the finalizer
passes: the anonymous PDF and supplement are ready for upload within their
stated scientific scope. The separate platform audit remains `CONDITIONAL GO`
until OpenReview profile, author metadata, quota, conflict, AI-use form,
parallel-submission, and upload gates are completed manually. This does not
make the finance audit causal.
