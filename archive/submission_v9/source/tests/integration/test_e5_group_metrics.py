from __future__ import annotations

import json
from pathlib import Path

from experiments.e5_budget_frontier import main as e5_main


def test_e5_writes_group_level_metrics_for_paired_aggregation(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "e5"
    monkeypatch.setattr(
        "sys.argv",
        [
            "e5_budget_frontier.py",
            "--config",
            "configs/sweeps/e5.yaml",
            "--output",
            str(output),
        ],
    )
    e5_main()
    rows = [json.loads(line) for line in (output / "group_metrics.jsonl").read_text().splitlines()]
    assert rows
    assert {row["method"] for row in rows} >= {"random_hf", "top_proxy_hf", "pivot"}
    assert {"group_id", "cti", "isr", "budget"} <= set(rows[0])
