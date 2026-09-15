from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_closed_loop_persists_rounds_and_query_ledger(tmp_path: Path) -> None:
    output = tmp_path / "e9"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    subprocess.run(
        [
            sys.executable,
            "experiments/e9_closed_loop.py",
            "--config",
            "configs/sweeps/e9.yaml",
            "--output",
            str(output),
        ],
        check=True,
        env=environment,
    )
    rounds = json.loads((output / "closed_loop.json").read_text())
    assert len(rounds) == 8
    assert all(row["hf_budget"] == 1 for row in rounds)
    assert (output / "transition_rows.jsonl").exists()
    assert len((output / "transition_rows.jsonl").read_text().splitlines()) == 8 * 3
    assert len((output / "hf_query_rows.jsonl").read_text().splitlines()) == 8
