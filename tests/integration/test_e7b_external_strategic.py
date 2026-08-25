from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_e7b_reports_actor_and_strategic_effects_with_holdout_metadata(tmp_path: Path) -> None:
    output = tmp_path / "e7b"
    subprocess.run(
        [
            sys.executable,
            "experiments/e7b_external_strategic.py",
            "--config",
            "configs/confirmatory/e7b.yaml",
            "--output",
            str(output),
            "--seeds",
            "4",
            "--candidates",
            "3",
        ],
        check=True,
    )
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["environment_source"] == "Farama MPE2"
    assert set(metrics["families"]) == {"family_a", "family_b"}
    assert "sir" in metrics["families"]["family_a"]
    assert metrics["cross_family"]["split_contract"] == "opponent-family-disjoint"
    assert (output / "transition_rows.jsonl").exists()
