from __future__ import annotations

import json
from pathlib import Path

import pytest

from pivot.theory.empirical import run_theory_experiment
from pivot.validation import validate_run_artifacts


def test_theory_experiment_writes_deterministic_audited_artifact(tmp_path: Path) -> None:
    config = {
        "policy_counts": [16, 32],
        "epsilons": [0.1],
        "response_strengths": [0.0, 0.5],
        "footprints": [0.1, 0.2],
        "seeds": [1, 2],
        "operator_samples": 12,
        "value_lipschitz": 1.5,
    }
    output = tmp_path / "theory"
    result = run_theory_experiment(output, config)

    assert result["global_fidelity_pass"] is True
    assert result["response_footprint_bound_pass"] is True
    assert validate_run_artifacts(
        output,
        [
            "global_fidelity_rows.jsonl",
            "response_footprint_rows.jsonl",
            "metrics.json",
            "summary.csv",
            "provenance.json",
            "config_snapshot.json",
            "README.md",
        ],
    )["valid"] is True
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    config_snapshot = json.loads((output / "config_snapshot.json").read_text(encoding="utf-8"))
    assert metrics["global_fidelity"]["n_rows"] == 4
    assert metrics["global_fidelity"]["all_epsilon_targets_covered"] is True
    assert metrics["response_footprint"]["n_rows"] == 8
    assert config_snapshot["q_a_definition"].startswith("empirical law")
    assert config_snapshot["improvement_fidelity_losses"] == [
        "absolute_delta_error",
        "sign_error",
    ]
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["git_commit"]


def test_theory_experiment_refuses_to_mix_runs(tmp_path: Path) -> None:
    output = tmp_path / "theory"
    output.mkdir()
    (output / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_theory_experiment(output, {"policy_counts": [16]})
