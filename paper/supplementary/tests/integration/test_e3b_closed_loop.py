from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_e3b_development_run_emits_clustered_metrics_and_state(tmp_path: Path) -> None:
    output = tmp_path / "e3b"
    subprocess.run(
        [
            sys.executable,
            "experiments/e3b_closed_loop.py",
            "--config",
            "configs/confirmatory/e3b.yaml",
            "--output",
            str(output),
            "--phase",
            "development",
            "--max-trajectories",
            "1",
            "--rounds",
            "1",
            "--methods",
            "Proxy Only,PIVOT-VOI,All-HF Oracle",
        ],
        check=True,
    )
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    state = json.loads((output / "state.json").read_text(encoding="utf-8"))
    assert metrics["uncertainty_unit"] == "trajectory"
    assert "PIVOT-VOI" in metrics["methods"]
    assert state["state"] in {"UNDERPOWERED", "DESIGN_INVALID", "HYPOTHESIS_NOT_SUPPORTED", "HYPOTHESIS_SUPPORTED"}
    assert (output / "transition_rows.jsonl").exists()
    trajectories = json.loads((output / "trajectory_metrics.json").read_text(encoding="utf-8"))
    by_method = {str(item["trajectory_id"]).split("|method=", 1)[1]: item for item in trajectories}
    assert by_method["Proxy Only"]["hf_cost"] == 0.0
    assert by_method["All-HF Oracle"]["hf_cost"] == 8.0
    assert by_method["PIVOT-VOI"]["hf_cost"] <= 2.0
