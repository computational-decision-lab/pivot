from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def _run(experiment: str, output: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.v9.run",
            "--experiment",
            experiment,
            "--profile",
            "smoke",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )


def _rows(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_e2c_smoke_writes_schema_complete_hash_bound_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "e2c"
    _run("e2c", output)
    state = json.loads((output / "scientific_decision.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    rows = _rows(output / "transition_rows.jsonl.gz")
    assert state["status"] == "UNDERPOWERED"
    assert manifest["status"] == "UNDERPOWERED"
    assert len(rows) > 0
    required = {"experiment_id", "environment_id", "operator_family", "delta_proxy", "delta_actor", "delta_true", "chi_square_shift", "hf_queried"}
    assert required <= rows[0].keys()
    assert {row["environment_id"] for row in rows} == {"performative_control", "congestion_resource", "mpe2_frozen"}


def test_e3c_smoke_keeps_frozen_null_and_shared_candidate_templates(tmp_path: Path) -> None:
    output = tmp_path / "e3c"
    _run("e3c", output)
    frozen = json.loads((output / "mpe2_frozen_reference.json").read_text(encoding="utf-8"))
    rows = _rows(output / "transition_rows.jsonl.gz")
    assert frozen["status"] == "HYPOTHESIS_NOT_SUPPORTED"
    first_round = [
        row
        for row in rows
        if row["environment_id"] == "performative_control" and row["seed"] == 9101 and row["round_id"] == 0
    ]
    templates = {}
    for row in first_round:
        templates.setdefault(row["method"], set()).add(row["candidate_template_id"])
    assert len(templates) >= 4
    assert len({tuple(sorted(value)) for value in templates.values()}) == 1


def test_e5c_smoke_emits_fixed_budget_methods_and_real_calibration(tmp_path: Path) -> None:
    output = tmp_path / "e5c"
    _run("e5c", output)
    decision = json.loads((output / "scientific_decision.json").read_text(encoding="utf-8"))
    groups = _rows(output / "group_metrics.jsonl.gz")
    calibration = json.loads((output / "calibration_robustness.json").read_text(encoding="utf-8"))
    assert decision["status"] == "UNDERPOWERED"
    assert {row["method"] for row in groups} >= {"proxy_only", "paired_lucb", "pivot_voi", "all_hf"}
    assert {int(row["candidate_count"]) for row in groups} == {4, 8, 16}
    assert calibration["posterior_sample_stability"]
    assert "selected_set_jaccard" in calibration["posterior_sample_stability"][0]
    assert "selected_query_jaccard" in calibration["cost_misspecification"][0]


def test_e7c_smoke_emits_all_opponent_mechanisms(tmp_path: Path) -> None:
    output = tmp_path / "e7c"
    _run("e7c", output)
    decision = json.loads((output / "scientific_decision.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "strategic_summary.json").read_text(encoding="utf-8"))
    assert decision["status"] == "UNDERPOWERED"
    assert {row["opponent_mode"] for row in summary["by_mode"]} == {
        "fixed",
        "reactive",
        "best_response",
        "gradient_adaptive",
        "rl_evolutionary",
    }
