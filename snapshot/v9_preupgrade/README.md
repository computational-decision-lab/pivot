# V9 Pre-upgrade Baseline

This directory records the V7 PIVOT publication and experiment surface before
the V9 upgrade. `manifest.json` stores the pre-upgrade git commit and SHA-256
for every existing result, figure, PDF, supplement, source, and registered
configuration. The files remain in their canonical paths so the public package
does not duplicate large raw tables.

Do not overwrite or re-label these artifacts. V9 runs use separate IDs under
`results/v9/`, `figures/v9/`, and `tables/v9`; powered nulls are preserved as
results.

Verification uses the recorded baseline commit for tracked blobs. Ignored
LaTeX build and expanded supplement derivatives are classified as derived and
reported separately by `scripts/freeze_v9_baseline.py --verify`. The parent
checkout status inventory is omitted so unrelated workspace names are not
published.
