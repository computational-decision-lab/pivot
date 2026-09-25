# Current manuscript figures

The ten entries below follow the figure order in `reproduction/manuscript/main.tex`
at final readback `f7afee89` (2026-09-26). This package freezes the current
manuscript visuals. It does not rerun simulations or create new estimates.

Run from any working directory with Python 3 and PyMuPDF (`fitz`), NumPy, and
Matplotlib installed:

```sh
python3 /path/to/repo/reproduction/figures/run.py --output /tmp/pivot-figures
```

The command creates `/tmp/pivot-figures/figures/` with all ten manuscript
paths, plus `figure-manifest.json` and plot audits in `/tmp/pivot-figures/`.
To build the manuscript against these results, place or copy the generated
`figures/` directory beside `main.tex`. The launcher accepts an absolute or
relative output path and resolves all inputs relative to its own location.

| Manuscript no. | PDF under `figures/` | Provenance | Frozen source and boundary |
| --- | --- | --- | --- |
| 1 | `fig1_improvement_reversal.pdf` | Supplied asset | Final 2026-09-26 readback. Original plotting data and code are not present in this package. |
| 2 | `fig2_operator_shift.pdf` | PDF replay | `base/fig2_operator_shift.pdf`; frozen vector-label patch in `source/replay_legacy_labels.py`. Four labels changed; 122 chart paths preserved. |
| 3 | `fig1_decision_relevance.pdf` | Supplied asset | Final 2026-09-26 readback. Earlier label-patch output differs from the final PDF, so this uses the final verified asset. |
| 4 | `fig4_evidence_efficiency.pdf` | Supplied asset | User-fixed PDF, SHA-256 `66a796b6deb61dd20f953dd14bebe41179ae9199019d559dbcc23de5007849ea`. Treated as immutable; no local redraw or alteration. |
| 5 | `figA_response_footprint.pdf` | PDF replay | `base/figA_response_footprint.pdf`; header/annotation edits in `source/refine_leduc_and_response.py`. Pixels outside the authorized label areas remain unchanged. |
| 6 | `additional_leduc/01_proxy_deployment_reversal.pdf` | PDF replay | `base/01_proxy_deployment_reversal.pdf`; vertical-axis label edit in `source/refine_leduc_and_response.py`. Pixels outside the authorized area remain unchanged. |
| 7 | `additional_leduc/05_v4_component_contrasts.pdf` | Saved-data plot | `data/leduc/summary.json`; 11 displayed means and 9 nonzero intervals checked against `base/05_v4_component_contrasts.pdf` before plotting. |
| 8 | `fig5_closed_loop.pdf` | PDF replay | `base/fig5_closed_loop.pdf`; frozen label patch in `source/replay_legacy_labels.py`. Two labels changed; 291 chart paths preserved. |
| 9 | `figC_posterior_robustness.pdf` | Supplied asset | Final 2026-09-26 readback. Earlier label-patch output differs from the final PDF, so this uses the final verified asset. |
| 10 | `additional_metadrive/matched_posterior_selected_audit_gain.pdf` | Saved-data plot | `data/metadrive/seed_results.csv` and `summary.json`; 30 roots, 24 method/budget rows, and paired contrasts checked by `source/replot_metadrive.py`. |

`run.py` pins the SHA-256 of every packaged data table, base PDF, and final
reference PDF. Supplied assets must be byte-identical to the final readback.
Replayed and saved-data plots must match its page geometry and differ by at
most 0.1% of rendered pixel channels at 144 dpi; the current environment
reproduces all six with zero rendered-pixel difference, and both saved-data
plots are also byte-identical. The manifest records output hashes and each
comparison. Differences in PDF metadata can make a replay's binary hash differ
even when its rendered page is identical.

The source classes are deliberately distinct. A saved-data plot recomputes
only the displayed summaries and bootstrap intervals from frozen evidence; a
PDF replay changes labels on a saved vector plot without recomputing results;
a supplied asset is an exact frozen copy where the final statistical plotting
source is absent or the final artwork had a later edit. None is a claim of
full experimental regeneration.
