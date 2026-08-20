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
- PDF SHA-256: `b60c4a255d7329a3d0dcd241bdc5bb635df623387c62910b2c9672cf9b6c24ec`
- Supplement SHA-256: `8a1f73857b2f46ee0218745c9335ee0759d84b41d9b82fdbd8a39f080c409767`
- Snapshot manifest SHA-256: `e485dba1ef28fd8eefa9f5a217086bde4a416bb9646d6212306b516020b6d15e`

OpenTikZ is installed through the lock-bound bootstrap at commit
`359befbf8e8af7ce08e7e387b2c2a198e0ca735d`; the adapted source is included
in the snapshot with its editable metadata and SVG preview.
The current architecture PDF SHA-256 is
`83d7d965b33a644f734ff746cefb6e08ce64c6b172f9041701cced65d98f1034`.
Its feedback loop stays inside the PIVOT panel, W0 stays inside the adaptive
world panel, and visual review confirmed that the `execution fidelity` label,
panel titles, and report containers are clear of connector and boundary overlap.

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
