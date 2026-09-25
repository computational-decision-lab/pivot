from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_e4b_reports_unseen_trajectory_estimands(tmp_path: Path) -> None:
    output = tmp_path / "e4b"
    subprocess.run(
        [
            sys.executable,
            "experiments/e4b_global_vs_transition.py",
            "--config",
            "configs/confirmatory/e4b.yaml",
            "--output",
            str(output),
            "--trajectory-count",
            "4",
            "--held-out-trajectories",
            "2",
            "--rounds",
            "1",
        ],
        check=True,
    )
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["split_contract"] == "trajectory-disjoint"
    assert metrics["global_value"]["policy_value_mae"] >= 0.0
    assert metrics["transition"]["ide"] >= 0.0
    assert metrics["held_out_trajectory_count"] == 2
    assert (output / "manifest.json").exists()
