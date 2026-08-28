# Reproducibility Audit

Protocol inputs, lock, canonical schemas, candidate provenance, figure passports, and the preserved baseline are checked for presence.

```json
{
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
}
```
