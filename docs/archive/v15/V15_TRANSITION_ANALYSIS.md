# Transition Analysis

Cluster-bootstrap analysis of observer versus deployment improvement transitions.

```json
{
  "by_operator": {
    "harness_skill_evolution": {
      "IDE": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 1,
        "n_rows": 2
      },
      "IRR": {
        "ci_high": null,
        "ci_low": null,
        "estimate": null,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 0,
        "n_rows": 0
      },
      "ISC": {
        "ci_high": null,
        "ci_low": null,
        "estimate": null,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 0,
        "n_rows": 0
      },
      "SIRR": {
        "ci_high": null,
        "ci_low": null,
        "estimate": null,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 0,
        "n_rows": 0
      },
      "strategic_effect": {
        "ci_high": null,
        "ci_low": null,
        "estimate": null,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 0,
        "n_rows": 0
      }
    },
    "mutation_self_edit": {
      "IDE": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 1,
        "n_rows": 2
      },
      "IRR": {
        "ci_high": null,
        "ci_low": null,
        "estimate": null,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 0,
        "n_rows": 0
      },
      "ISC": {
        "ci_high": null,
        "ci_low": null,
        "estimate": null,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 0,
        "n_rows": 0
      },
      "SIRR": {
        "ci_high": null,
        "ci_low": null,
        "estimate": null,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 0,
        "n_rows": 0
      },
      "strategic_effect": {
        "ci_high": null,
        "ci_low": null,
        "estimate": null,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 0,
        "n_rows": 0
      }
    }
  },
  "by_task_family": {
    "mixed": {
      "IDE": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 4
      },
      "IRR": {
        "ci_high": null,
        "ci_low": null,
        "estimate": null,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 0,
        "n_rows": 0
      },
      "ISC": {
        "ci_high": null,
        "ci_low": null,
        "estimate": null,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 0,
        "n_rows": 0
      },
      "SIRR": {
        "ci_high": null,
        "ci_low": null,
        "estimate": null,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 0,
        "n_rows": 0
      },
      "strategic_effect": {
        "ci_high": null,
        "ci_low": null,
        "estimate": null,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 0,
        "n_rows": 0
      }
    }
  },
  "confirmatory": false,
  "criterion": "95% cluster-bootstrap interval excludes zero after the frozen independent-N rule",
  "design_valid": true,
  "footprint": {
    "l1_error_association": {
      "ci_high": 0.0,
      "ci_low": 0.0,
      "estimate": 0.0,
      "feature": "registered_footprint_l1_norm",
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4,
      "outcome_fields_used": []
    },
    "outcome_fields_used": [],
    "registered_features": [
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
    ]
  },
  "global_policy_value_fidelity_available": false,
  "hypotheses": {
    "H1": {
      "criterion": "95% cluster-bootstrap interval excludes zero after the frozen independent-N rule",
      "metric": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 4
      },
      "primary_metric": "IDE",
      "statement": "autonomously generated updates have measurable observer/deployment error",
      "terminal_state": "UNDERPOWERED"
    },
    "H2": {
      "criterion": "predeclared operator strata are compared with global policy rank fidelity; no delta reconstruction is used",
      "metric": {
        "ci_high": null,
        "ci_low": null,
        "estimate": null,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 0,
        "n_rows": 0
      },
      "primary_metric": "operator_relative_rank_fidelity",
      "statement": "operator-conditioned transition fidelity differs from global fidelity",
      "terminal_state": "UNDERPOWERED"
    },
    "H5": {
      "criterion": "association is descriptive unless a frozen confirmatory split and interval are available",
      "metric": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "feature": "registered_footprint_l1_norm",
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 4,
        "outcome_fields_used": []
      },
      "primary_metric": "registered_footprint_l1_norm_association",
      "statement": "registered update footprint predicts transition error",
      "terminal_state": "UNDERPOWERED"
    }
  },
  "implementation_failures": 0,
  "independent_n": 2,
  "independent_unit": "trajectory_or_task_cluster",
  "leakage_detected": false,
  "metrics": {
    "IDE": {
      "ci_high": 0.0,
      "ci_low": 0.0,
      "estimate": 0.0,
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 2,
      "n_rows": 4
    },
    "IRR": {
      "ci_high": null,
      "ci_low": null,
      "estimate": null,
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 0,
      "n_rows": 0
    },
    "ISC": {
      "ci_high": null,
      "ci_low": null,
      "estimate": null,
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 0,
      "n_rows": 0
    },
    "SIRR": {
      "ci_high": null,
      "ci_low": null,
      "estimate": null,
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 0,
      "n_rows": 0
    },
    "strategic_effect": {
      "ci_high": null,
      "ci_low": null,
      "estimate": null,
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 0,
      "n_rows": 0
    }
  },
  "note": "DEV-only descriptive result; no scientific claim is allowed until the frozen confirmatory archive is opened.",
  "outcome_chasing": false,
  "phase": "DEV",
  "policy_level": {
    "available": false,
    "global_rank_fidelity": {
      "ci_high": null,
      "ci_low": null,
      "estimate": null,
      "independent_unit": "trajectory_or_task_cluster",
      "n_clusters": 0,
      "n_rows": 0
    },
    "note": "Policy-level rank fidelity requires recorded level scores for all candidates in a batch.",
    "rank_fidelity_by_operator": {},
    "rank_observation_count": 0
  },
  "primary_hypothesis": "H1",
  "rows_analyzed": 4,
  "rows_read": 4,
  "schema_version": "pivot-v15-scientific-analysis-1",
  "source": "results/v15/dev-external-transition-audit/autonomous_transitions.jsonl",
  "source_hash": "84c6b7a837a8cbf4f93dee7c822664a4de6d6cc958723609cbf860c966ae0cfd",
  "source_manifest_hash": "6f499b8132534822b2fcdf10d79804f6559b59d25b4497234817247ad369aa78",
  "status": "DEV_ONLY",
  "terminal_state": "UNDERPOWERED"
}
```
