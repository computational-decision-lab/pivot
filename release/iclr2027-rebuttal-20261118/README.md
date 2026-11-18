# ICLR 2027 rebuttal release — 2026-11-18

The current manuscript fits the 10-page discussion-version main-text limit,
with the Conclusion ending on page 10, references on page 11, and the appendix
on page 14 (20 pages in total). Improvement Fidelity is the evaluation
contribution; PIVOT is the validation framework; PIVOT-KG applies established
one-step correlated-Gaussian KG.

The HighwayEnv 0.0435-to-0.0055 result is explicitly a saved-truth replay.
Both fresh-simulator IVR comparison intervals include zero. The five-round
study is a controlled extension selected after exploration, with calibration/
test shift and a six-contrast Holm family that does not cover the preceding
variant search. Active Offline Policy Selection is cited in Related Work.

## Build the release

Run `bash paper/build.sh` in the full checkout. This compiles frozen audited
assets, builds the anonymous supplementary ZIP, and verifies the release.
For an extracted supplementary ZIP, run `bash paper/build_frozen.sh`; Python,
TeX Live including texlive-science, latexmk, and Poppler are required.

The supplemental source closure includes every referenced TeX, bibliography,
and figure asset. The verifier rejects missing or stale manuscript files.
The embedded release scripts are preserved byte for byte so anonymity
sanitization cannot alter their own checking rules. The new baseline and
calibration runners, normalized replay inputs, sealed poker decisions, and
fresh Highway aggregate ledgers are included. Complete per-shard physical
runtime journals remain outside this compact archive; their recorded audit
and aggregate cost/decision evidence are included.

## Validation

`validation.json` records artifact hashes and the checks: 42 relevant unit
tests, selected release-script lint, 29 submission machine checks, a clean
build from the extracted supplement, 75,000 sealed/scored poker decisions,
four fresh Highway contrasts and intervals, and the full five-round replay
(10,800 queries, 5,400 state transitions, and 1,080 trajectories).

The automated status is still CONDITIONAL GO. Author/OpenReview form gates
remain pending; external evidence retains its bounded-null/scoped-pass
classification. This release prepares files and does not submit to OpenReview.
