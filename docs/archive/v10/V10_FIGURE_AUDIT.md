# V10 Figure Audit

Status: **PASS**

Each required bundle is checked for PDF/SVG/PNG/CSV/Parquet, one-page rendering, metadata, source hashes, and explicit uncertainty/display policy.

| Figure | CSV rows | Parquet rows | PNG | PDF pages | Alias | Valid |
| --- | ---: | ---: | --- | ---: | --- | --- |
| `fig1_improvement_reversal` | 28800 | 28800 | (2196, 708) | 1 | `` | True |
| `fig2_operator_shift` | 20 | 20 | (2205, 675) | 1 | `` | True |
| `fig3_pivot_voi` | 11 | 11 | (2065, 945) | 1 | `fig3_pivot_architecture` | True |
| `fig4_evidence_efficiency` | 78 | 78 | (2515, 1299) | 1 | `` | True |
| `fig5_closed_loop` | 324 | 324 | (2227, 1218) | 1 | `` | True |
| `figA_response_footprint` | 150 | 150 | (2239, 690) | 1 | `fig6_world_response_decomposition` | True |
| `figB_learned_ood_null` | 16 | 16 | (2265, 654) | 1 | `fig7_learned_ood_null` | True |
| `figC_posterior_robustness` | 9 | 9 | (2225, 637) | 1 | `fig8_voi_robustness` | True |
| `figD_strategic_distribution` | 155 | 155 | (2221, 690) | 1 | `fig9_strategic_generalization` | True |
| `figE_finance_boundary` | 5 | 5 | (1459, 665) | 1 | `` | True |

## Reading checks

- `raw_observations_visible`: True
- `uncertainty_or_descriptive_spans_labeled`: True
- `incomparable_conditions_declared`: True
- `oracle_reference_separate`: True
- `grayscale_encoding_redundant`: True
- `no_interpolation_of_unobserved_cells`: True
- `source_traceability`: True
