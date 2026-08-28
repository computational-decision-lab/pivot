# Finalization Audit

Local package status: **BLOCKED**. External modern-agent evidence remains separate from local paper compliance.

```json
{
  "audit_failures": [],
  "audits": {
    "anonymity": {
      "email_hits": [],
      "manual_platform_gate": "OpenReview profile, conflicts, affiliations, and upload metadata",
      "nonempty_author_commands": [],
      "pdf_exists": true,
      "private_path_hits": [],
      "supplement_exists": true,
      "valid": true
    },
    "claims": {
      "body_version_tokens": [],
      "claim_ids": [
        "C1",
        "C2",
        "C3",
        "C4",
        "C5"
      ],
      "claims_have_statements": true,
      "forbidden_inference_count": 5,
      "invalid_terminal_states": {},
      "missing_fields": {},
      "registry": "research/claims_v15.yaml",
      "required_fields": [
        "claim_id",
        "statement",
        "required_experiment",
        "required_terminal_state",
        "allowed_scope",
        "forbidden_scope",
        "paper_location"
      ],
      "required_ids_present": true,
      "statuses": [
        "boundary",
        "conditional",
        "registered"
      ],
      "unique_claim_ids": true,
      "valid": true
    },
    "figures": {
      "all_final": true,
      "figure_count": 14,
      "records_well_formed": true,
      "review_coverage_match": true,
      "review_manifest_error": null,
      "review_manifest_valid": true,
      "selection": "all_manifest_figures",
      "status": "PASS",
      "valid": true,
      "visual_context_pass": true
    },
    "language": {
      "body_source_files": [
        "main.tex"
      ],
      "body_source_version_hits": {},
      "body_version_tokens": [],
      "paper_facing_codex_tokens": 0,
      "pdf_exists": true,
      "pdf_version_tokens": [],
      "reviewer_artifact_codex_hits": [],
      "reviewer_artifact_codex_tokens": 0,
      "source_exists": true,
      "valid": true
    },
    "numbers": {
      "checks": {
        "StrategicAdaptiveFamilies": true,
        "StrategicAllFamilySeedTraces": true,
        "StrategicClusters": true,
        "StrategicFamilySeedTraces": true,
        "StrategicSIRR": true
      },
      "expected": {
        "StrategicAdaptiveFamilies": "3",
        "StrategicAllFamilySeedTraces": "150",
        "StrategicClusters": "30",
        "StrategicFamilySeedTraces": "90",
        "StrategicSIRR": "0.9495"
      },
      "valid": true
    },
    "references": {
      "bibliography_exists": true,
      "cited_entry_count": 42,
      "duplicate_entry_keys": [],
      "entry_count": 67,
      "missing_citation_keys": [],
      "unique_entry_count": 67,
      "valid": true
    },
    "reproducibility": {
      "dev_only_artifacts_explicit": true,
      "external_execution_status": "PINNED_DEV_RUNTIME_VERIFIED; CONFIRMATORY_NOT_OPENED",
      "external_phase_statuses": {
        "dev-external-ablations": "COMPLETED",
        "dev-external-closed-loop": "COMPLETED",
        "dev-external-promotion": "COMPLETED",
        "dev-external-strategic-response": "PARTIAL",
        "dev-external-transition-audit": "COMPLETED",
        "dev-pi-replication": "COMPLETED"
      },
      "missing": [],
      "required_artifacts": {
        "V15_SCIENTIFIC_SUMMARY.md": true,
        "artifacts/v15/figure_status.json": true,
        "artifacts/v15/scientific_summary.json": true,
        "artifacts/v15/terminal_state_audit.json": true,
        "configs/v15/confirmatory.yaml": true,
        "configs/v15/external_versions.json": true,
        "configs/v15/task_manifest.json": true,
        "experiments/v15/confirmatory_lock.json": true,
        "figures/v15/visual_review_manifest.json": true,
        "results/v15/candidate-archive/manifest.json": true,
        "results/v15/canonical/manifest.json": true,
        "results/v15/dev-smoke/manifest.json": true,
        "results/v15/promotion-replay/manifest.json": true,
        "snapshot/v15_pre_modern_agent/PROVENANCE.txt": true
      },
      "terminal_state_audit": {
        "present": true,
        "sha256": "9a7b134fa7f2444ffd1eee848badfdca6df055e2ac37a2555de58e3194f9c3da",
        "valid": true
      },
      "valid": true,
      "visual_review": {
        "error": null,
        "figure_count": 14,
        "valid": true
      }
    },
    "terminal_states": {
      "attempted_phase_count": 6,
      "candidate_archives": {
        "candidate-archive": {
          "archive_sha256": "d735daf23b55af4aa5b0b342d810e25fd78841c0c8c481ce4f4c613d9569d4c3",
          "confirmatory": false,
          "manifest_sha256": "1de876291ef9454f9038d009b473bccd8d607017810c291c2ac2f4fe964a8304",
          "row_count": 16,
          "valid": true
        },
        "dev-external-candidate-archive": {
          "archive_sha256": "f9ab8b189b0c71295ee5b4c1e83b09cf9d5ad7058951d9be5f02bf64f088983f",
          "confirmatory": false,
          "manifest_sha256": "13710c9185d0a4222727ed3b1d45226fd7b7b75da963c83eb9a52aeda171477a",
          "row_count": 4,
          "valid": true
        }
      },
      "issues": [],
      "lock": {
        "confirmatory_open": false,
        "hash_valid": true,
        "outcome_chasing": false,
        "present": true
      },
      "no_outcome_chasing": true,
      "note": "This audit validates closure and provenance; it does not upgrade DEV or underpowered artifacts into confirmatory evidence.",
      "phase_count": 6,
      "phases": {
        "dev-external-ablations": {
          "access_issues": [],
          "attempted": true,
          "confirmatory": false,
          "decision_path": "/opt/projects/research/pivot/results/v15/dev-external-ablations/scientific_decision.json",
          "leakage_detected": false,
          "manifest_hash_valid": true,
          "manifest_terminal_state": "UNDERPOWERED",
          "outcome_chasing": false,
          "status": "COMPLETED",
          "terminal_state": "UNDERPOWERED",
          "valid": true
        },
        "dev-external-closed-loop": {
          "access_issues": [],
          "attempted": true,
          "confirmatory": false,
          "decision_path": "/opt/projects/research/pivot/results/v15/dev-external-closed-loop/scientific_decision.json",
          "leakage_detected": false,
          "manifest_hash_valid": true,
          "manifest_terminal_state": "UNDERPOWERED",
          "outcome_chasing": false,
          "status": "COMPLETED",
          "terminal_state": "UNDERPOWERED",
          "valid": true
        },
        "dev-external-promotion": {
          "access_issues": [],
          "attempted": true,
          "confirmatory": false,
          "decision_path": "/opt/projects/research/pivot/results/v15/dev-external-promotion/scientific_decision.json",
          "leakage_detected": false,
          "manifest_hash_valid": true,
          "manifest_terminal_state": "UNDERPOWERED",
          "outcome_chasing": false,
          "status": "COMPLETED",
          "terminal_state": "UNDERPOWERED",
          "valid": true
        },
        "dev-external-strategic-response": {
          "access_issues": [],
          "attempted": true,
          "confirmatory": false,
          "decision_path": "/opt/projects/research/pivot/results/v15/dev-external-strategic-response/scientific_decision.json",
          "leakage_detected": false,
          "manifest_hash_valid": true,
          "manifest_terminal_state": "UNDERPOWERED",
          "outcome_chasing": false,
          "status": "PARTIAL",
          "terminal_state": "UNDERPOWERED",
          "valid": true
        },
        "dev-external-transition-audit": {
          "access_issues": [],
          "attempted": true,
          "confirmatory": false,
          "decision_path": "/opt/projects/research/pivot/results/v15/dev-external-transition-audit/scientific_decision.json",
          "leakage_detected": false,
          "manifest_hash_valid": true,
          "manifest_terminal_state": "UNDERPOWERED",
          "outcome_chasing": false,
          "status": "COMPLETED",
          "terminal_state": "UNDERPOWERED",
          "valid": true
        },
        "dev-pi-replication": {
          "access_issues": [],
          "attempted": true,
          "confirmatory": false,
          "decision_path": "/opt/projects/research/pivot/results/v15/dev-pi-replication/scientific_decision.json",
          "leakage_detected": false,
          "manifest_hash_valid": true,
          "manifest_terminal_state": "UNDERPOWERED",
          "outcome_chasing": false,
          "status": "COMPLETED",
          "terminal_state": "UNDERPOWERED",
          "valid": true
        }
      },
      "schema_version": "pivot-v15-terminal-state-audit-1",
      "status": "PASS",
      "task_planes": {
        "disjoint": true,
        "manifest_sha256": "33a0546b95e8765cdd677b1ee27aaee896bb1c3e2933dce050693f3e27d404bc",
        "plane_counts": {
          "assessment": 4,
          "gate": 4,
          "proxy": 4
        },
        "present": true,
        "public_membership_match": true,
        "valid": true
      },
      "terminal_states": [
        "UNDERPOWERED"
      ],
      "valid": true
    }
  },
  "figure_count": 14,
  "scientific_report": {
    "blockers": [
      "confirmatory mini-SWE transition audit",
      "confirmatory promotion replay and closed loop",
      "confirmatory Pi replication",
      "confirmatory strategic response and registered ablations"
    ],
    "canonical_rows": {
      "autonomous_transitions": 4,
      "closed_loop_results": 10,
      "hf_queries": 62,
      "promotion_candidates": 4,
      "promotion_results": 42
    },
    "dev_transition_count": 4,
    "evidence_profile": "LEVEL_E",
    "language": {
      "body_source_files": [
        "main.tex"
      ],
      "body_source_version_hits": {},
      "body_version_tokens": [],
      "paper_facing_codex_tokens": 0,
      "pdf_exists": true,
      "pdf_version_tokens": [],
      "reviewer_artifact_codex_hits": [],
      "reviewer_artifact_codex_tokens": 0,
      "source_exists": true,
      "valid": true
    },
    "lock_sha256": "5ce1ca3e0aa2ddeb4ef818d2cddbab21315b345830eeebf593f4a6750de7680f",
    "manifest_migration": {
      "changed": [],
      "manifest_hashes": {
        "dev-external-ablations": "5a9620e6435ead52bf6bc53312af28dc80fd67e56409e767172b9ede712dc567",
        "dev-external-closed-loop": "05b2b74467ae0ec6d18a75ea3e62a53af6253e85eeaf490fa1ae0bf24ed6cd7d",
        "dev-external-promotion": "7ca02c2a6c9d188357a560f7eedf6b89d26c695c6cdcde70e8fbf20c1b6a4b37",
        "dev-external-strategic-response": "d05842bed862d1df2478c75b6fd1c3fc8f61869cd20445857ec7429088d4c31a",
        "dev-external-transition-audit": "6f499b8132534822b2fcdf10d79804f6559b59d25b4497234817247ad369aa78",
        "dev-pi-replication": "b8c5d490435a19da7ce54cb14e77727c1c895b0d82e04e7f900156c2376235a0"
      },
      "missing": [],
      "schema_version": "pivot-v15-manifest-contract-1",
      "skipped_confirmatory": []
    },
    "numbers": {
      "checks": {
        "StrategicAdaptiveFamilies": true,
        "StrategicAllFamilySeedTraces": true,
        "StrategicClusters": true,
        "StrategicFamilySeedTraces": true,
        "StrategicSIRR": true
      },
      "expected": {
        "StrategicAdaptiveFamilies": "3",
        "StrategicAllFamilySeedTraces": "150",
        "StrategicClusters": "30",
        "StrategicFamilySeedTraces": "90",
        "StrategicSIRR": "0.9495"
      },
      "valid": true
    },
    "references": {
      "bibliography_exists": true,
      "cited_entry_count": 42,
      "duplicate_entry_keys": [],
      "entry_count": 67,
      "missing_citation_keys": [],
      "unique_entry_count": 67,
      "valid": true
    },
    "required_reports": [
      "V15_BASELINE_SNAPSHOT.md",
      "V15_REPO_AUDIT.md",
      "V15_RESOURCE_PLAN.md",
      "V15_CONSTRUCT_VALIDITY.md",
      "V15_CONFIRMATORY_PREREGISTRATION.md",
      "V15_TRANSITION_AUDIT.md",
      "V15_SCIENTIFIC_SUMMARY.md",
      "V15_OPERATOR_RELATIVE_ANALYSIS.md",
      "V15_FOOTPRINT_ANALYSIS.md",
      "V15_PROMOTION_RESULTS.md",
      "V15_PAIRED_ABLATION.md",
      "V15_PIVOT_ABLATIONS.md",
      "V15_CLOSED_LOOP_RESULTS.md",
      "V15_PI_REPLICATION.md",
      "V15_STRATEGIC_RESULTS.md",
      "V15_FALSIFICATION_REPORT.md",
      "V15_FIGURE_STATUS.md",
      "V15_PAPER_CONTEXT_AUDIT.md",
      "V15_NUMBER_AUDIT.md",
      "V15_CLAIM_AUDIT.md",
      "V15_REFERENCE_AUDIT.md",
      "V15_ANONYMITY_AUDIT.md",
      "V15_LANGUAGE_AUDIT.md",
      "V15_REPRODUCIBILITY_AUDIT.md",
      "V15_REVIEWER_ATTACK_AUDIT.md",
      "V15_OUTSTANDING_PROFILE.md",
      "V15_FINAL_REPORT.md"
    ],
    "scientific_analysis": {
      "ablations": {
        "independent_n": 3,
        "status": "DEV_ONLY",
        "terminal_state": "UNDERPOWERED"
      },
      "closed_loop": {
        "independent_n": 2,
        "status": "DEV_ONLY",
        "terminal_state": "UNDERPOWERED"
      },
      "pi": {
        "independent_n": 1,
        "status": "DEV_ONLY",
        "terminal_state": "UNDERPOWERED"
      },
      "promotion": {
        "independent_n": 2,
        "status": "DEV_ONLY",
        "terminal_state": "UNDERPOWERED"
      },
      "strategic": {
        "independent_n": 0,
        "status": "DEV_ONLY",
        "terminal_state": "UNDERPOWERED"
      },
      "transition": {
        "independent_n": 2,
        "status": "DEV_ONLY",
        "terminal_state": "UNDERPOWERED"
      }
    },
    "scientific_summary": {
      "analysis_artifacts": {
        "ablations": "artifacts/v15/ablation_analysis.json",
        "closed_loop": "artifacts/v15/closed_loop_analysis.json",
        "pi": "artifacts/v15/pi_analysis.json",
        "promotion": "artifacts/v15/promotion_analysis.json",
        "strategic": "artifacts/v15/strategic_analysis.json",
        "transition": "artifacts/v15/transition_analysis.json"
      },
      "confirmatory": false,
      "hypotheses": {
        "H1": "UNDERPOWERED",
        "H2": "UNDERPOWERED",
        "H3": "UNDERPOWERED",
        "H4": "UNDERPOWERED",
        "H5": "UNDERPOWERED",
        "H6": "UNDERPOWERED"
      },
      "independent_units": {
        "ablations": 3,
        "closed_loop": 2,
        "pi": 1,
        "promotion": 2,
        "strategic": 0,
        "transition": 2
      },
      "levels": {
        "modern_agent_evidence": "LEVEL_E",
        "nulls_preserved": true
      },
      "multiple_testing": {
        "adjusted_p_values": {},
        "method": "Holm",
        "raw_p_values": {},
        "status": "NO_CONFIRMATORY_P_VALUES"
      },
      "outcome_chasing": false,
      "schema_version": "pivot-v15-scientific-summary-1",
      "status": "DEV_ONLY"
    },
    "status": "BLOCKED",
    "terminal_state_audit": {
      "issues": [],
      "status": "PASS",
      "valid": true
    }
  },
  "status": "BLOCKED",
  "submission_verification": {
    "returncode": 0,
    "status": "PASS",
    "stderr": "",
    "stdout": "{\"decision\": \"CONDITIONAL GO\", \"blocking_gates\": [\"manual_gates.openreview_profile\", \"manual_gates.author_quota_and_reciprocal_review\", \"manual_gates.final_author_metadata_and_conflicts\", \"manual_gates.ai_use_disclosure_in_submission_form\", \"manual_gates.no_parallel_submission_confirmation\", \"scientific_gates.external_interactive_response\", \"scientific_gates.external_strategic_validation\"]}\n"
  }
}
```
