from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from pivot.environments.performative.config import PerformativeConfig
from pivot.environments.performative.proxy import run_first_milestone
from pivot.validation import validate_figure_bundle


def test_canonical_figures_regenerate_from_source_tables(tmp_path: Path) -> None:
    source = tmp_path / "source"
    figures = tmp_path / "figures"
    run_first_milestone(
        source,
        PerformativeConfig(response_strength=0.4),
        seeds=[1],
        candidate_scales=[0.1, 0.2],
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    subprocess.run(
        [sys.executable, "scripts/make_paper_figures.py", "--input", str(source), "--output", str(figures)],
        check=True,
        env=environment,
    )
    result = validate_figure_bundle(
        figures,
        (
            "fig1_when_better_gets_worse",
            "fig2_reversal_phase_diagram",
            "fig3_optimizing_wrong_world",
            "fig4_policy_vs_improvement_fidelity",
            "fig5_pivot_budget_frontier",
            "fig6_observer_actor_strategic",
            "fig7_strategic_reversal",
        ),
    )
    assert result["valid"] is True
    assert (figures / "fig1_when_better_gets_worse.csv").exists()
