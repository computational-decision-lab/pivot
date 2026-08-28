# Closed-Loop Analysis

Paired terminal deployment transfer and cumulative selection-regret analysis.

```json
{
  "assessment_audit": {
    "all_rows_queried_once": true,
    "all_rows_terminal_role": true,
    "assessment_rows": 10,
    "outcomes_returned_to_operator": false,
    "sealed": true,
    "unique_terminal_pairs": 2
  },
  "cisr_effect": {
    "ci_high": 0.0,
    "ci_low": 0.0,
    "direction": "proxy_minus_pivot; positive_favors_pivot",
    "estimate": 0.0,
    "n_clusters": 2,
    "n_rows": 2
  },
  "confirmatory": false,
  "criterion": "PIVOT-VOI minus Proxy Only terminal assessment score has a positive 95% paired cluster-bootstrap lower bound",
  "design_valid": true,
  "endpoint_by_method": {
    "All-HF Oracle": {
      "ci_high": 0.0,
      "ci_low": 0.0,
      "estimate": 0.0,
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 2
    },
    "Global-VOI": {
      "ci_high": 0.0,
      "ci_low": 0.0,
      "estimate": 0.0,
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 2
    },
    "PIVOT-VOI": {
      "ci_high": 0.0,
      "ci_low": 0.0,
      "estimate": 0.0,
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 2
    },
    "Paired LUCB": {
      "ci_high": 0.0,
      "ci_low": 0.0,
      "estimate": 0.0,
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 2
    },
    "Proxy Only": {
      "ci_high": 0.0,
      "ci_low": 0.0,
      "estimate": 0.0,
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 2
    }
  },
  "endpoint_effect": {
    "ci_high": 0.0,
    "ci_low": 0.0,
    "estimate": 0.0,
    "n_clusters": 2,
    "n_rows": 2
  },
  "implementation_failures": 0,
  "independent_n": 2,
  "independent_unit": "trajectory_cluster",
  "leakage_detected": false,
  "note": "Terminal assessment is summarized from existing rows and is never re-queried by analysis.",
  "outcome_chasing": false,
  "phase": "DEV",
  "primary_hypothesis": "H3_closed_loop_transfer",
  "schema_version": "pivot-v15-scientific-analysis-1",
  "source": "results/v15/dev-external-closed-loop/closed_loop_results.jsonl",
  "source_hash": "302110a272af13ffa4267868e4611e13e9fd5cb44593d54cac556dfb0de3b2fd",
  "source_manifest_hash": "05b2b74467ae0ec6d18a75ea3e62a53af6253e85eeaf490fa1ae0bf24ed6cd7d",
  "status": "DEV_ONLY",
  "terminal_state": "UNDERPOWERED"
}
```
