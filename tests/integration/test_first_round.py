from pathlib import Path

from pivot.environments.performative.config import PerformativeConfig
from pivot.environments.performative.proxy import run_first_milestone


def test_first_milestone_creates_transition_data_and_five_outputs(tmp_path: Path) -> None:
    result = run_first_milestone(
        output_dir=tmp_path,
        config=PerformativeConfig(response_strength=0.7, competition_strength=0.0),
        seeds=[1, 2],
        candidate_scales=[0.2, 0.5],
    )
    assert result.row_count == 4
    assert (tmp_path / "transitions.jsonl").exists()
    for name in (
        "proxy_vs_true_scatter.json",
        "irr_vs_response.json",
        "irr_vs_footprint.json",
        "response_footprint_heatmap.json",
        "confidence_intervals.json",
    ):
        assert (tmp_path / name).exists()
