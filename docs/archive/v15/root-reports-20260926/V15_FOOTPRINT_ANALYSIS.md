# Footprint Analysis

Registered pre-gate footprint features are evaluated against transition error with trajectory-cluster bootstrap intervals.

```json
{
  "analysis_decision_artifact": "artifacts/v15/transition_scientific_decision.json",
  "feature_associations": [
    {
      "feature": "prompt_token_delta",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": [],
      "pearson_error": null
    },
    {
      "feature": "prompt_semantic_distance",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": [],
      "pearson_error": null
    },
    {
      "feature": "skill_diff_size",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": [],
      "pearson_error": null
    },
    {
      "feature": "skills_added",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": [],
      "pearson_error": null
    },
    {
      "feature": "skills_removed",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": [],
      "pearson_error": null
    },
    {
      "feature": "tool_schema_change",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": [],
      "pearson_error": null
    },
    {
      "feature": "tool_count_delta",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": [],
      "pearson_error": null
    },
    {
      "feature": "loop_parameter_delta",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": [],
      "pearson_error": null
    },
    {
      "feature": "context_policy_change",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": [],
      "pearson_error": null
    },
    {
      "feature": "test_policy_change",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": [],
      "pearson_error": null
    },
    {
      "feature": "search_policy_change",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": [],
      "pearson_error": null
    },
    {
      "feature": "tool_call_distribution_shift",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": [],
      "pearson_error": null
    },
    {
      "feature": "shell_command_distribution_shift",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": [],
      "pearson_error": null
    },
    {
      "feature": "test_execution_shift",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": [],
      "pearson_error": null
    },
    {
      "feature": "files_read_shift",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": [],
      "pearson_error": null
    },
    {
      "feature": "files_written_shift",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": [],
      "pearson_error": null
    },
    {
      "feature": "dependency_operation_shift",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": [],
      "pearson_error": null
    },
    {
      "feature": "token_usage_shift",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": [],
      "pearson_error": null
    },
    {
      "feature": "context_peak_shift",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": [],
      "pearson_error": null
    },
    {
      "feature": "wall_clock_shift",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": [],
      "pearson_error": null
    },
    {
      "feature": "action_sequence_distance",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": [],
      "pearson_error": null
    }
  ],
  "feature_contract": [
    "prompt_token_delta",
    "prompt_semantic_distance",
    "skill_diff_size",
    "skills_added",
    "skills_removed",
    "tool_schema_change",
    "tool_count_delta",
    "loop_parameter_delta",
    "context_policy_change",
    "test_policy_change",
    "search_policy_change",
    "tool_call_distribution_shift",
    "shell_command_distribution_shift",
    "test_execution_shift",
    "files_read_shift",
    "files_written_shift",
    "dependency_operation_shift",
    "token_usage_shift",
    "context_peak_shift",
    "wall_clock_shift",
    "action_sequence_distance"
  ],
  "improvement_reversal_rate": {
    "ci_high": 0.0,
    "ci_low": 0.0,
    "clusters": 2,
    "estimate": 0.0,
    "rows": 4
  },
  "independent_trajectory_units": 2,
  "independent_unit": "trajectory_or_task_cluster",
  "leakage_detected": false,
  "note": "DEV_ONLY descriptive diagnostic; transition rows are nested within trajectory units and are not confirmatory evidence.",
  "outcome_chasing": false,
  "outcome_fields_excluded_from_features": [
    "actor_reversal",
    "assessment_score",
    "delta_actor",
    "delta_strategic",
    "deployment_score",
    "strategic_reversal"
  ],
  "phase": "DEV",
  "rows_read": 4,
  "rows_with_proxy_and_actor": 4,
  "schema_version": "pivot-v15-footprint-analysis-1",
  "scientific_claim_allowed": false,
  "source": "results/v15/canonical/autonomous_transitions.csv",
  "source_hash": "8f076188dac0ce522865de86aee68b7603b709d566d366b9145d144d15b59b82",
  "source_manifest": "results/v15/canonical/manifest.json",
  "source_manifest_sha256": "ff98ac32af7f70de29a843fc5db5460df23bd2d2835065f7742d1fd093d59e32",
  "status": "DEV_ONLY",
  "terminal_state": "UNDERPOWERED",
  "transition_error": {
    "ci_high": 0.0,
    "ci_low": 0.0,
    "clusters": 2,
    "estimate": 0.0,
    "rows": 4
  }
}
```
