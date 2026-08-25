from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_e2_operator_shift_writes_registered_manifest_and_q_a_metrics(tmp_path: Path) -> None:
    output = tmp_path / "e2"
    subprocess.run(
        [
            sys.executable,
            "experiments/e2_operator_shift.py",
            "--config",
            "configs/registered/e2_operator_shift.yaml",
            "--output",
            str(output),
        ],
        check=True,
    )
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    summaries = json.loads((output / "temperature_summary.json").read_text(encoding="utf-8"))
    assert metrics["q_a_definition"].startswith("softmax")
    assert metrics["population_size"] >= 4
    assert len(summaries) == 5
    assert manifest["file_count"] == 6
    for name, digest in manifest["files"].items():
        import hashlib

        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
