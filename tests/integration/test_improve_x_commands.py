from __future__ import annotations

import hashlib
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


def write_v2_config(path: Path) -> Path:
    config = path / "benchmark_v2.yaml"
    config.write_text(
        """
mode: multiround_multioperator
world:
  response_strength: 0.7
  competition_strength: 0.5
  noise_scale: 0.0
  horizon: 12
  reward_bound: 10.0
  config_id: improve-x-benchmark-v2
splits:
  train: [31]
  validation: [41]
  test: [51]
rounds: 3
candidate_scales: [0.04, 0.10, 0.20]
initial_policy:
  intensity: 0.15
contexts_per_transition: 2
collection_promotion_world: actor
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config


def test_build_improvementbench_is_deterministic_and_has_all_world_levels(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    config = "configs/improve_x/benchmark.yaml"
    run_script("scripts/build_improvementbench.py", config, first)
    run_script("scripts/build_improvementbench.py", config, second)
    rows = ImprovementBenchDataset.read(first).rows
    assert len(rows) == 12
    assert {row.world_level for row in rows} == {"observer", "actor", "strategic"}
    metadata = json.loads((first / "metadata.json").read_text(encoding="utf-8"))
    assert "release_version" not in metadata
    frozen_manifest = json.loads(
        (PROJECT_ROOT / "benchmarks/improvementbench/v1/manifest.json").read_text(encoding="utf-8")
    )
    assert hashlib.sha256((first / "transitions.jsonl").read_bytes()).hexdigest() == frozen_manifest["files"]["transitions.jsonl"]
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


def test_build_v2_improvementbench_has_frozen_splits_and_cross_operator_pools(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    config = write_v2_config(tmp_path)
    run_script("scripts/build_improvementbench.py", str(config), first)
    run_script("scripts/build_improvementbench.py", str(config), second)
    dataset = ImprovementBenchDataset.read(first)

    assert len(dataset.rows) == 243
    assert dataset.split_names == ("test", "train", "validation")
    assert {row.improvement_operator for row in dataset.rows} == {
        "synthetic",
        "rl-update",
        "evolutionary-mutation",
    }
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    for split in dataset.split_names:
        split_rows = dataset.rows_for_split(split)
        assert len(split_rows) == 81
        assert sum(bool(row.metadata["collection_selected"]) for row in split_rows) == 9
        for round_id in range(3):
            for world_level in ("observer", "actor", "strategic"):
                candidates = [
                    row
                    for row in split_rows
                    if row.round_id == round_id and row.world_level == world_level
                ]
                assert len(candidates) == 9
                assert {row.candidate_index for row in candidates} == set(range(9))
                assert {row.improvement_operator for row in candidates} == {
                    "synthetic",
                    "rl-update",
                    "evolutionary-mutation",
                }

        scores = {
            row.transition_id: row.deployment_delta
            for row in split_rows
            if row.deployment_delta is not None
        }
        ranking = evaluate_ranking_task(split_rows, scores)
        assert ranking["n_groups"] == 9
        assert ranking["accuracy"] == 1.0


def test_improvementbench_evaluator_filters_a_frozen_split(tmp_path: Path) -> None:
    dataset_output = tmp_path / "benchmark"
    metrics_output = tmp_path / "metrics"
    config = write_v2_config(tmp_path)
    run_script("scripts/build_improvementbench.py", str(config), dataset_output)
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_improvementbench.py",
            "--input",
            str(dataset_output),
            "--output",
            str(metrics_output),
            "--split",
            "test",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads((metrics_output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["requested_split"] == "test"
    assert metrics["row_count"] == 81


def test_multiround_comparison_reports_matched_hf_budget_and_true_curves(tmp_path: Path) -> None:
    output = tmp_path / "comparison"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_improve_x_comparison.py",
            "--config",
            "configs/improve_x/comparison.yaml",
            "--benchmark",
            "benchmarks/improvementbench/v2",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "methods" in result.stdout
    summary = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    rounds = [
        json.loads(line)
        for line in (output / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    methods = set(summary["methods"])
    assert methods == {"proxy_only", "random_hf", "top_proxy_hf", "pivot_x"}
    assert summary["evaluation_split"] == "test"
    assert summary["calibration_split"] == "train"
    assert summary["rounds"] == 3
    assert summary["hf_queries_per_round"] == 2
    assert (output / "manifest.json").is_file()
    assert {row["method"] for row in rounds} == methods
    assert len(rounds) == 4 * 3 * 9
    for method in methods:
        method_rows = [row for row in rounds if row["method"] == method]
        assert sum(bool(row["selected"]) for row in method_rows) == 3
        queried = {row["transition_id"] for row in method_rows if row["hf_queried"]}
        expected_queries = 0 if method == "proxy_only" else 6
        assert len(queried) == expected_queries
        assert len({row["round_id"] for row in method_rows}) == 3
        assert len(summary["results"][method]["true_curve"]) == 4
        assert "final_true_performance" in summary["results"][method]


def test_multiround_comparison_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    command = [
        sys.executable,
        "scripts/run_improve_x_comparison.py",
        "--config",
        "configs/improve_x/comparison.yaml",
        "--benchmark",
        "benchmarks/improvementbench/v2",
    ]
    subprocess.run([*command, "--output", str(first)], cwd=PROJECT_ROOT, check=True)
    subprocess.run([*command, "--output", str(second)], cwd=PROJECT_ROOT, check=True)

    assert (first / "comparison.json").read_bytes() == (second / "comparison.json").read_bytes()
    assert (first / "rounds.jsonl").read_bytes() == (second / "rounds.jsonl").read_bytes()
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()


def test_multiround_comparison_manifest_covers_every_result_file(tmp_path: Path) -> None:
    output = tmp_path / "comparison"
    command = [
        sys.executable,
        "scripts/run_improve_x_comparison.py",
        "--config",
        "configs/improve_x/comparison.yaml",
        "--benchmark",
        "benchmarks/improvementbench/v2",
        "--output",
        str(output),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["row_count"] == 108
    assert manifest["benchmark_manifest_sha256"]
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


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
