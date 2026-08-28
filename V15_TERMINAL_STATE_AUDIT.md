# Terminal-State and Leakage Audit

Every attempted V15 phase must resolve to one closed terminal state without role leakage or outcome chasing.

```json
{
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
```
