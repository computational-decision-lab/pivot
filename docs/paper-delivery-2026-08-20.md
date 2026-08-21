# PIVOT Paper Delivery Record

Date: 2026-08-21 (Asia/Shanghai)

## Current ICLR 2027 package

The current anonymous ICLR package is the `paper/iclr2027/` tree, not the
historical working-paper build below. It is locally verified as **CONDITIONAL
GO** with 9 main-text pages and 14 total pages. The bibliography starts on
page 10 and is excluded from the ICLR main-text limit through the explicit
`refs:start` boundary; the appendix follows the bibliography:

- PDF: `paper/iclr2027/pivot_iclr2027_submission.pdf`
- Supplement: `paper/iclr2027/pivot_iclr2027_supplementary.zip`
- Snapshot: `paper/snapshot/manifest.json` (33 hash-indexed files)
- Architecture: `paper/snapshot/figures/fig3_pivot_architecture.pdf`
- PDF SHA-256: `617a7ac552b120f78e4da198b39f81157c0b83da5e1320afc7b460ffe8227778`
- Supplement SHA-256: `7f9c55f861d40f7630a18bdea3baec3104c8797057904977d8f51191ea80e70f`
- Snapshot manifest SHA-256: `a79d65be03b8719aaa3b06fac4c7181f02147a15a455baa2267a1942214931fa`

OpenTikZ is installed through the lock-bound bootstrap at commit
`359befbf8e8af7ce08e7e387b2c2a198e0ca735d`; the adapted source is included
in the snapshot with its editable metadata and SVG preview.
The current architecture PDF SHA-256 is
`32fcc9dedbe13d2f72f9ac01483f09cadd7f15fb2be7352f053d429b29636f1b`.
Its feedback loop stays inside the PIVOT panel, W0 stays inside the adaptive
world panel, and visual review confirmed that the `execution fidelity` label,
panel titles, and report containers are clear of connector and boundary overlap.

The 50-entry bibliography prioritizes verified 2025--2026 work while keeping
the most relevant ICLR 2024 and NeurIPS 2022 benchmark papers. All 46 arXiv
identifiers resolve through the official API; the title/author audit and the
two corrected stale URLs are recorded in
`docs/reference-audit-2026-08-21.md`.

Figures are generated with the local PIVOT style adapter, whose two external
style references are pinned in `configs/tooling/figure_tools.json`; the
checkouts remain outside the tracked/public tree. The paper's AI Use
Statement is limited to language editing and code/figure formatting support.

The supplement also contains the registered V6 theory artifact: 285
row-level checks for Global Fidelity Blindness and the exact response-footprint
bound. These checks are analytic constructions and do not close the separate
causal interactive-response or strategic-validation gates.

The remaining blockers are author-side OpenReview gates and independent
interactive/strategic scientific validation. They are intentionally not
represented as passed by the local build.

## Historical working-paper record

## Delivered

The anonymous working paper is built from the tracked snapshot at
`paper/snapshot/`; the build never reads `/tmp`, network data, credentials,
raw vendor L2, or live execution endpoints.

- Source: `paper/main.tex`
- Bibliography: `paper/references.bib`
- Build command: `(cd paper && ./build.sh)`
- PDF: `paper/pivot_working_paper.pdf`
- Verification report: `paper/verification.json`
- First-page raster preview: `paper/preview.png`
- Snapshot manifest: `paper/snapshot/manifest.json`

The PDF is 6 pages in total, with 5 pages before the labelled appendix. The
verification report records: main text within the 9-page gate, embedded fonts,
anonymous author metadata, non-empty text and preview, no undefined references,
and zero overfull boxes.

## Evidence Boundary

The paper reports registered controlled P2/E4/E5/E6/E7/E8/E9 evidence and the
twelve-ablation aggregate. It reports the Binance multi-asset public audit as
observational: 0/7 primary and 0/5 holdout depth-proxy reversals, with a
negative pooled depth effect. It does not claim causal endogenous response,
realistic strategic equilibrium, performative overoptimization in E3, or
LLM/EvoQuant/M3 results.

## Remaining Before External Submission

1. Re-run the controlled and public registries from a fresh environment and
   compare hashes against the tracked snapshot.
2. Add an independently calibrated causal interactive response source and an
   external strategic validation if the paper is to make a finance/market claim.
3. Freeze a confirmatory typed update-generation rule and volatility/liquidity
   labels before treating the public audit as hypothesis testing.
4. Recheck the official ICLR 2027 call, template, anonymity rules, and deadline
   before submission; the current PDF is a working paper, not an uploaded paper.

Databento was not queried because no Databento credential is configured and the
current scientific boundary does not require private or paid L2 data. Binance
public data already provide the declared observational audit.
