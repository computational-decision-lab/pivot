from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from pivot.core.policy import Policy
from pivot.core.result import RolloutContext
from pivot.core.transition import PolicyTransition
from pivot.environments.interactive_market.config import InteractiveMarketConfig
from pivot.environments.interactive_market.world import InteractiveMarketWorld
from pivot.environments.strategic_market.config import StrategicMarketConfig
from pivot.environments.strategic_market.world import StrategicMarketWorld
from pivot.evaluation.paired import PairedEvaluator


def _transition() -> PolicyTransition:
    return PolicyTransition(
        incumbent=Policy.from_mapping({"intensity": 0.2, "position_size": 0.2}),
        candidate=Policy.from_mapping({"intensity": 0.6, "position_size": 0.6}),
        round_id=0,
        candidate_index=0,
        improvement_operator="strategic_fixture",
        edit_type="intensity",
    )


def test_same_transition_exhibits_incremental_strategic_reversal() -> None:
    transition = _transition()
    context = [RolloutContext(seed=1, scenario_id="strategic-1")]
    actor_world = InteractiveMarketWorld(InteractiveMarketConfig(participation_rate=0.01))
    strategic_world = StrategicMarketWorld(
        StrategicMarketConfig(
            interactive=actor_world.config,
            opponent_mode="adaptive",
            adaptation_steps=5,
            learning_rate=0.2,
            market_share_sensitivity=0.04,
        )
    )

    proxy_delta = PairedEvaluator(actor_world, mode="observer").evaluate(transition, context).delta
    actor_delta = PairedEvaluator(actor_world, mode="actor").evaluate(transition, context).delta
    strategic_delta = PairedEvaluator(strategic_world, mode="strategic").evaluate(
        transition, context
    ).delta

    assert proxy_delta > 0
    assert actor_delta > 0
    assert strategic_delta < 0


def test_competition_knobs_change_strategic_value_systematically() -> None:
    policy = Policy.from_mapping({"intensity": 0.7, "position_size": 0.7})
    interactive = InteractiveMarketConfig(participation_rate=0.05)
    weak = StrategicMarketWorld(
        StrategicMarketConfig(
            interactive=interactive,
            opponent_mode="adaptive",
            opponent_count=1,
            adaptation_steps=2,
            learning_rate=0.05,
            market_share_sensitivity=0.01,
        )
    ).evaluate(policy, seed=1)
    strong = StrategicMarketWorld(
        StrategicMarketConfig(
            interactive=interactive,
            opponent_mode="adaptive",
            opponent_count=2,
            adaptation_steps=8,
            learning_rate=0.2,
            market_share_sensitivity=0.04,
        )
    ).evaluate(policy, seed=1)

    assert strong.value < weak.value
    assert strong.metadata["opponent_response"] > weak.metadata["opponent_response"]


def test_e8_registered_sweep_persists_all_competition_axes(tmp_path: Path) -> None:
    output = tmp_path / "e8"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    subprocess.run(
        [
            sys.executable,
            "experiments/e8_competition.py",
            "--config",
            "configs/sweeps/e8.yaml",
            "--output",
            str(output),
        ],
        check=True,
        env=environment,
    )
    rows = json.loads((output / "competition.json").read_text(encoding="utf-8"))
    assert {row["mode"] for row in rows} == {"fixed", "reactive", "adaptive"}
    assert {row["opponent_count"] for row in rows} == {1, 2}
    assert {row["adaptation_steps"] for row in rows} >= {0, 1, 5, 10}
    assert {row["learning_rate"] for row in rows} >= {0.05, 0.2}
    assert {row["market_share_sensitivity"] for row in rows} >= {0.0, 0.01, 0.04}
    assert (output / "provenance.json").exists()
