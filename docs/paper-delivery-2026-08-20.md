# PIVOT Paper Delivery Record

Date: 2026-08-20 (Asia/Shanghai)

## Current ICLR 2027 package

The current anonymous ICLR package is the `paper/iclr2027/` tree, not the
historical working-paper build below. It is locally verified as **CONDITIONAL
GO** with 9 main pages and 11 total pages:

- PDF: `paper/iclr2027/pivot_iclr2027_submission.pdf`
- Supplement: `paper/iclr2027/pivot_iclr2027_supplementary.zip`
- Snapshot: `paper/snapshot/manifest.json` (33 hash-indexed files)
- Architecture: `paper/snapshot/figures/fig3_pivot_architecture.pdf`
- PDF SHA-256: `1125b8896754450b77466f7f3d381e0b888ef66b7cae1cb22eef7bca84bdeb21`
- Supplement SHA-256: `db391ee57d9970488f14f477af6ea7a1e561439578bc0917056aa5496362c39b`
- Snapshot manifest SHA-256: `4dddd65f289c0bf9ccbb3dc3ca9ca2bd32e10cd848567c51b90c93ca1108d37c`

OpenTikZ is installed through the lock-bound bootstrap at commit
`359befbf8e8af7ce08e7e387b2c2a198e0ca735d`; the adapted source is included
in the snapshot with its editable metadata and SVG preview.
The current architecture PDF SHA-256 is
`b5fdb18241229b68e4fa48816afd11ca2c573206e37eb6924c418ca6c7eb4725`.
Its feedback loop uses a dedicated outer routing channel, and visual review
confirmed that the `execution fidelity` label and all report containers are
clear of connector and container overlap.

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
