from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from improve_x.benchmark.dataset import ImprovementBenchDataset
from improve_x.benchmark.tasks import evaluate_ranking_task

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_script(script: str, config: str, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script, "--config", config, "--output", str(output)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_build_improvementbench_is_deterministic_and_has_all_world_levels(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    config = "configs/improve_x/benchmark.yaml"
    run_script("scripts/build_improvementbench.py", config, first)
    run_script("scripts/build_improvementbench.py", config, second)
    rows = ImprovementBenchDataset.read(first).rows
    assert len(rows) == 12
    assert {row.world_level for row in rows} == {"observer", "actor", "strategic"}
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert ImprovementBenchDataset.read(first).validate()["valid"] is True


def test_build_improvementbench_preserves_world_specific_values_and_candidate_groups(tmp_path: Path) -> None:
    output = tmp_path / "benchmark"
    run_script("scripts/build_improvementbench.py", "configs/improve_x/benchmark.yaml", output)
    rows = ImprovementBenchDataset.read(output).rows

    for round_id in {row.round_id for row in rows}:
        for world_level in ("observer", "actor", "strategic"):
            candidates = [row for row in rows if row.round_id == round_id and row.world_level == world_level]
            assert [row.candidate_index for row in candidates] == [0, 1]

    first_candidate = [row for row in rows if row.round_id == 0 and row.candidate_index == 0]
    values = {row.world_level: row for row in first_candidate}
    assert values["observer"].deployment_value == values["observer"].proxy_value
    assert values["actor"].deployment_value != values["strategic"].deployment_value


def test_improvementbench_ranking_task_groups_candidates_by_world_and_round(tmp_path: Path) -> None:
    output = tmp_path / "benchmark"
    run_script("scripts/build_improvementbench.py", "configs/improve_x/benchmark.yaml", output)
    rows = ImprovementBenchDataset.read(output).rows
    scores = {row.transition_id: row.deployment_delta for row in rows if row.deployment_delta is not None}

    ranking = evaluate_ranking_task(rows, scores)

    assert ranking["n_groups"] == 6
    assert ranking["accuracy"] == 1.0


def test_improvementbench_evaluator_emits_task_and_layer_metrics(tmp_path: Path) -> None:
    dataset_output = tmp_path / "benchmark"
    metrics_output = tmp_path / "metrics"
    run_script("scripts/build_improvementbench.py", "configs/improve_x/benchmark.yaml", dataset_output)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_improvementbench.py",
            "--input",
            str(dataset_output),
            "--output",
            str(metrics_output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "rows" in result.stdout
    metrics = json.loads((metrics_output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["row_count"] == 12
    assert metrics["dataset_valid"] is True
    assert metrics["sign_by_world"]["observer"]["accuracy"] == 1.0
    assert metrics["layer_fidelity"]["observer_fidelity"] == 0.0


def test_trajectory_runner_retains_candidates_and_multiple_rounds(tmp_path: Path) -> None:
    output = tmp_path / "trajectory"
    run_script("scripts/run_improvement_trajectory.py", "configs/improve_x/trajectory.yaml", output)
    trajectory = json.loads((output / "trajectory.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (output / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert trajectory["rounds"] == 4
    assert len(rows) == 12
    assert sum(bool(row["selected"]) for row in rows) == 4
    assert {row["failure_type"] for row in rows}
    assert trajectory["true_curve"][-1] < 0
    assert trajectory["actor_curve"] == trajectory["true_curve"]
    assert trajectory["strategic_curve"][-1] < trajectory["actor_curve"][-1]
    first_round = [row for row in rows if row["round_id"] == 0]
    footprints = [float(row["update_footprint"]) for row in first_round]
    assert footprints == sorted(footprints)
    assert footprints[-1] > footprints[0] * 3
    assert (output / "provenance.json").is_file()
