from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_aggregate_results_preserves_failed_runs(tmp_path: Path) -> None:
    first = tmp_path / "first"
    first.mkdir()
    (first / "metrics.json").write_text(json.dumps({"irr": 0.5, "ide": 1.0}), encoding="utf-8")
    output = tmp_path / "aggregate.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    subprocess.run(
        [sys.executable, "scripts/aggregate_results.py", "--inputs", str(first), str(tmp_path / "missing"), "--output", str(output)],
        check=True,
        env=environment,
    )
    payload = json.loads(output.read_text())
    assert payload["n_valid"] == 1
    assert payload["n_inputs"] == 2
    assert payload["runs"][1]["status"] == "missing_metrics"
