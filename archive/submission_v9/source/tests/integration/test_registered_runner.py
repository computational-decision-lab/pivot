from __future__ import annotations

import json
from pathlib import Path

import pytest

from pivot.analysis.registry import run_registered


def test_registered_runner_materializes_isolated_runs_and_manifests(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        """
world:
  response_strength: 0.7
  noise_scale: 0.05
  horizon: 4
  reward_bound: 10.0
  config_id: registered-test
seeds: [1]
response_strengths: [0.0, 0.7]
candidate_scales: [0.2]
optimization_strengths: [1.0]
""",
        encoding="utf-8",
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        f"""
experiment: p2
base_config: {base}
seed_sets:
  - run_id: r1
    seeds: [101]
  - run_id: r2
    seeds: [202]
""",
        encoding="utf-8",
    )
    output = tmp_path / "runs"
    result = run_registered(registry, output, project_root=Path.cwd())
    assert result["n_ok"] == 2
    for run_id in ("r1", "r2"):
        run = output / run_id
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "ok"
        assert (run / "config.yaml").exists()
        assert (run / "stdout.log").exists()
        assert (run / "transitions.jsonl").exists()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_registered(registry, output, project_root=Path.cwd())


def test_registered_runner_accepts_ablation_suite(tmp_path: Path) -> None:
    registry = tmp_path / "ablations.yaml"
    registry.write_text(
        """experiment: ablations
base_config: configs/sweeps/ablations.yaml
seed_sets:
  - run_id: a01
    seeds: [41, 42, 43, 44]
  - run_id: a02
    seeds: [51, 52, 53, 54]
""",
        encoding="utf-8",
    )
    output = tmp_path / "runs"
    result = run_registered(registry, output, project_root=Path.cwd())
    assert result["experiment"] == "ablations"
    assert result["n_ok"] == 2
    assert (output / "a01" / "ablation_summary.json").exists()
