# V10 Pre-Final Audit

Status: **PASS**

## Frozen baseline

- Source commit: `8541638d9115fb71c1b4d780e702762a030e3f59`
- Snapshot: `snapshot/v10_pre_final`
- Timestamp: `2026-08-26T20:16:54+08:00`
- PDF SHA-256: `928e4776de7b78814c78343e0a85933fa4dc08b33f379969750e4cd42791fc32`
- Supplement SHA-256: `60e5f13cc57b9007ea1b69d3326eadc884bc6d14fa5455f4a564e0b1ed4678df`
- Manifest SHA-256: `b70341fce9c4a8f7b5f8046fc4a265483456ed1856f6f9fededb2e6dc7724807`
- The snapshot is immutable. Its manifest is externally bound because a file cannot contain its own final SHA-256.

## Findings and resolutions

| Location | Problem | Severity | Required fix | Source evidence | Resolution |
| --- | --- | --- | --- | --- | --- |
| snapshot manuscript, Theory introduction | Text stated four results while six propositions were present. | **blocking** | State six results and automatically count proposition environments. | snapshot main.tex:283 and six proposition environments | fixed and checked by claim/PDF verifier |
| snapshot abstract, results, captions, and appendix | Internal version and run identifiers read like a development log. | **high** | Use scientific environment/operator names in the main paper; retain IDs only in artifact metadata. | snapshot main.tex contains V7/V9 and E2C/E3C/E4C/E5C/E7C | fixed in the scientific body; automated forbidden-token scan added |
| snapshot Figures 6--9 | Sparse points/bars/lines did not expose mechanism, raw variation, null evidence, or strategic reversal. | **high** | Rebuild as paired layers, effect distributions/forests, robustness diagnostics, and reversal planes. | frozen pre-final figure bundles | replaced by Figures A--D with source tables and cluster-level uncertainty |
| snapshot architecture figure | The earlier dense world/reporting layout produced container and connector collisions at paper scale. | **high** | Use a compact decision-critical two-band OpenTikZ architecture and keep world taxonomy in the appendix. | frozen OpenTikZ source and visual review | rebuilt with 11 semantic nodes, routed connectors, raster/vector QA, and no overflow |
| metrics and cross-environment figures | CISR was used for both one-set and cumulative settings without a single scale contract; FER was absent. | **high** | Define ISR, CTI, CISR_T, and matched-cell FER; prohibit raw cross-environment comparisons. | statistics.py, closed-loop runner, efficiency runner | formal definitions and V10 metric-scale audit added |
| related work and references | Novelty boundaries omitted key self-improvement reversal/benchmark work; one off-policy URL was wrong. | **high** | Add explicit boundary paragraphs and verify primary metadata. | pre-final bibliography and recent primary metadata | expanded recent/top-conference coverage; Thomas--Brunskill URL corrected |
| finance paragraph and public table | The seven-session single-asset sweep and the 12-session three-asset expansion were conflated. | **high** | Describe the datasets separately and source 0/7, 0/5, and the pooled effect from the frozen expansion JSON. | public calibration summary versus paper snapshot public-expansion-summary.json | main text, caption, table, and number audit now keep the two observational layers distinct |
| main Figure 3 bundle | Editable architecture assets existed but the required method-oriented figure bundle was incomplete. | **medium** | Export PDF/SVG/PNG/TEX plus semantic-node CSV/Parquet and metadata under fig3_pivot_voi. | release output contract | implemented in deterministic architecture/figure build |
| page budget | The pre-final manuscript occupied the full nine-page main-text budget. | **medium** | Rebuild and verify the reference boundary rather than total PDF page count. | pre-final verification main_pages=9 | final verifier enforces at most nine main pages; current build leaves margin |
| pre-final SHA256SUMS | The manifest contains an unavoidable invalid self-hash placeholder for SHA256SUMS itself. | **low** | Do not mutate the frozen snapshot; bind the manifest externally by its own SHA-256. | first manifest line is the empty-file digest while the external digest is nonempty | external freeze record stores the manifest hash and excludes self-verification |
| double-blind package | No identity leak was found in the pre-final PDF/package. | **none** | Preserve and rerun source/PDF/archive checks. | pre-final verification anonymous_author=true | retained as a release gate |
