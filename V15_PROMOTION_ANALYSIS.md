# Promotion Analysis

Cluster-bootstrap promotion regret and paired high-fidelity query accounting.

```json
{
  "budgets": [
    1,
    2,
    4
  ],
  "confirmatory": false,
  "criterion": "Proxy Only ISR minus PIVOT-VOI ISR has a positive 95% cluster-bootstrap lower bound at the frozen target budget",
  "design_valid": true,
  "fairness": {
    "all_hf_oracle_is_reference_only": true,
    "candidate_batch_hashes_by_method": {
      "All-HF Oracle": [
        "74b3822effebf872c6ebf2cda4adb1087ec51b9a67d67aa04003aef28c08b37f",
        "7614bbddec3fd5611d671facbc0c8d6056ffdd97360cc482c9b9483b2ed2843b"
      ],
      "Global-VOI": [
        "74b3822effebf872c6ebf2cda4adb1087ec51b9a67d67aa04003aef28c08b37f",
        "7614bbddec3fd5611d671facbc0c8d6056ffdd97360cc482c9b9483b2ed2843b"
      ],
      "PIVOT-H": [
        "74b3822effebf872c6ebf2cda4adb1087ec51b9a67d67aa04003aef28c08b37f",
        "7614bbddec3fd5611d671facbc0c8d6056ffdd97360cc482c9b9483b2ed2843b"
      ],
      "PIVOT-VOI": [
        "74b3822effebf872c6ebf2cda4adb1087ec51b9a67d67aa04003aef28c08b37f",
        "7614bbddec3fd5611d671facbc0c8d6056ffdd97360cc482c9b9483b2ed2843b"
      ],
      "Paired LUCB": [
        "74b3822effebf872c6ebf2cda4adb1087ec51b9a67d67aa04003aef28c08b37f",
        "7614bbddec3fd5611d671facbc0c8d6056ffdd97360cc482c9b9483b2ed2843b"
      ],
      "Proxy Only": [
        "74b3822effebf872c6ebf2cda4adb1087ec51b9a67d67aa04003aef28c08b37f",
        "7614bbddec3fd5611d671facbc0c8d6056ffdd97360cc482c9b9483b2ed2843b"
      ],
      "Random HF": [
        "74b3822effebf872c6ebf2cda4adb1087ec51b9a67d67aa04003aef28c08b37f",
        "7614bbddec3fd5611d671facbc0c8d6056ffdd97360cc482c9b9483b2ed2843b"
      ]
    },
    "same_candidate_batches_observed": true
  },
  "implementation_failures": 0,
  "independent_n": 2,
  "independent_unit": "candidate_batch_cluster",
  "leakage_detected": false,
  "method_metrics": {
    "All-HF Oracle": {
      "1": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 2
      },
      "2": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 2
      },
      "4": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 2
      }
    },
    "Global-VOI": {
      "1": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 2
      },
      "2": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 2
      },
      "4": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 2
      }
    },
    "PIVOT-H": {
      "1": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 2
      },
      "2": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 2
      },
      "4": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 2
      }
    },
    "PIVOT-VOI": {
      "1": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 2
      },
      "2": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 2
      },
      "4": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 2
      }
    },
    "Paired LUCB": {
      "1": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 2
      },
      "2": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 2
      },
      "4": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 2
      }
    },
    "Proxy Only": {
      "1": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 2
      },
      "2": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 2
      },
      "4": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 2
      }
    },
    "Random HF": {
      "1": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 2
      },
      "2": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 2
      },
      "4": {
        "ci_high": 0.0,
        "ci_low": 0.0,
        "estimate": 0.0,
        "independent_unit": "trajectory_or_task_cluster",
        "n_clusters": 2,
        "n_rows": 2
      }
    }
  },
  "note": "ISR orientation is fixed as Proxy Only minus PIVOT-VOI; positive values always favor PIVOT.",
  "outcome_chasing": false,
  "paired_effect": {
    "ci_high": 0.0,
    "ci_low": 0.0,
    "direction": "proxy_minus_pivot; positive_favors_pivot",
    "estimate": 0.0,
    "n_clusters": 2,
    "n_rows": 2
  },
  "phase": "DEV",
  "primary_hypothesis": "H3",
  "query_accounting": {
    "cache_hit_rows": 0,
    "legacy_fields_missing": true,
    "logical_hf_queries": 62,
    "physical_pair_evaluations": 62,
    "post_decision_truth_excluded": true,
    "query_rows": 62
  },
  "schema_version": "pivot-v15-scientific-analysis-1",
  "source": "results/v15/dev-external-promotion/promotion_results.jsonl",
  "source_hash": "a727b984071c0bc2ec1d76c7df25537ca633af773aea2206d886b49d3a1655c1",
  "source_manifest_hash": "7ca02c2a6c9d188357a560f7eedf6b89d26c695c6cdcde70e8fbf20c1b6a4b37",
  "status": "DEV_ONLY",
  "target_budget": 4,
  "terminal_state": "UNDERPOWERED"
}
```
