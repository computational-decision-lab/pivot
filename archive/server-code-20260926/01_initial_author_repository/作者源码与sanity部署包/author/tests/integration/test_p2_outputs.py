from __future__ import annotations

import json
from pathlib import Path

from pivot.environments.performative.config import PerformativeConfig
from pivot.environments.performative.proxy import run_first_milestone


def test_registered_grid_has_transition_schema_and_artifacts(tmp_path: Path) -> None:
    manifest = run_first_milestone(
        tmp_path,
        PerformativeConfig(response_strength=0.7, config_id="test-p2"),
        seeds=[1, 2, 3],
        candidate_scales=[0.1, 0.3],
        response_strengths=[0.0, 0.7],
        optimization_strengths=[0.5, 1.0],
    )
    assert manifest.row_count == 24
    assert set(manifest.required_columns) >= {
        "delta_proxy",
        "delta_true",
        "improvement_reversal",
        "update_footprint",
        "response_strength",
        "optimization_strength",
        "selected",
    }
    assert (tmp_path / "transitions.jsonl").exists()
    assert (tmp_path / "transitions.parquet").exists() or manifest.storage_format == "jsonl"
    assert json.loads((tmp_path / "schema.json").read_text()) ["row_count"] == 24
    assert json.loads((tmp_path / "provenance.json").read_text())["paired"] is True
    for stem in (
        "proxy_vs_true_scatter",
        "irr_vs_response",
        "irr_vs_footprint",
        "response_footprint_heatmap",
    ):
        assert (tmp_path / f"{stem}.json").exists()
        assert (tmp_path / f"{stem}.csv").exists()
        assert (tmp_path / f"{stem}.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "confidence_intervals.csv").exists()
    assert (tmp_path / "failed_runs.jsonl").read_text() == ""
