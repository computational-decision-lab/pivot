# Ablation Analysis

Registered pairing, footprint, and acquisition ablation diagnostics.

```json
{
  "all_rows": {
    "ci_high": 0.9333333333333332,
    "ci_low": 0.34761904761904755,
    "estimate": 0.6714285714285714,
    "independent_unit": "trajectory_or_task_cluster",
    "n_clusters": 5,
    "n_rows": 19
  },
  "assessment_accessed": false,
  "confirmatory": false,
  "criterion": "paired and unpaired evidence are compared on the frozen archive; no post-outcome feature selection",
  "design_valid": true,
  "implementation_failures": 0,
  "independent_n": 3,
  "independent_unit": "trajectory_or_task_cluster",
  "leakage_detected": false,
  "no_pairing_diagnostic": {
    "ci_high": 0.0,
    "ci_low": 0.0,
    "estimate": 0.0,
    "independent_unit": "trajectory_or_task_cluster",
    "n_clusters": 2,
    "n_rows": 4
  },
  "note": "DEV ablation rows are diagnostic; a null or overlap does not justify redesigning the protocol.",
  "outcome_chasing": false,
  "phase": "DEV",
  "primary_hypothesis": "H4",
  "query_overlap_diagnostic": {
    "ci_high": 0.9333333333333332,
    "ci_low": 0.34761904761904755,
    "estimate": 0.6714285714285714,
    "independent_unit": "trajectory_or_task_cluster",
    "n_clusters": 5,
    "n_rows": 19
  },
  "registered_ablation_families": [
    "no_footprint",
    "no_pairing",
    "no_voi",
    "promotion_baseline_diagnostic"
  ],
  "rows_read": 23,
  "schema_version": "pivot-v15-scientific-analysis-1",
  "source": "results/v15/dev-external-ablations/ablation_results.jsonl",
  "source_hash": "95e6a4cc16a2280ebfd9f94f4d4df42ab046ad167f7a946bae80f36a7e346782",
  "source_manifest_hash": "5a9620e6435ead52bf6bc53312af28dc80fd67e56409e767172b9ede712dc567",
  "status": "DEV_ONLY",
  "terminal_state": "UNDERPOWERED"
}
```
