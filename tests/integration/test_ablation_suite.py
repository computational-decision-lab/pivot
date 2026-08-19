from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.ablation_suite import REQUIRED_ABLATIONS, run_suite


def test_ablation_suite_writes_all_required_ids_and_raw_rows(tmp_path: Path) -> None:
    config = Path("configs/sweeps/ablations.yaml")
    output = tmp_path / "ablations"
    summary = run_suite(config, output)
    assert summary["ablation_count"] == len(REQUIRED_ABLATIONS) == 12
    assert set(summary["ablations"]) == set(REQUIRED_ABLATIONS)
    rows = [json.loads(line) for line in (output / "ablation_rows.jsonl").read_text().splitlines()]
    assert rows
    assert {row["ablation_id"] for row in rows} == set(REQUIRED_ABLATIONS)
    assert (output / "failed_runs.jsonl").exists()
    assert json.loads((output / "provenance.json").read_text())["hf_budget_matched"] is True


def test_ablation_suite_refuses_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "ablations"
    output.mkdir()
    (output / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_suite(Path("configs/sweeps/ablations.yaml"), output)
