#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any, cast

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.core.policy import Policy
from pivot.core.result import RolloutContext
from pivot.core.transition import PolicyTransition
from pivot.environments.interactive_market.config import InteractiveMarketConfig
from pivot.environments.interactive_market.world import InteractiveMarketWorld
from pivot.environments.strategic_market.config import OpponentMode, StrategicMarketConfig
from pivot.environments.strategic_market.world import StrategicMarketWorld
from pivot.evaluation.paired import PairedEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="PIVOT E8 competition sensitivity")
    parser.add_argument("--config", type=Path, default=Path("configs/sweeps/e8.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/raw/e8-competition"))
    args = parser.parse_args()
    raw_config = args.config.read_bytes()
    payload = yaml.safe_load(raw_config)
    if not isinstance(payload, dict):
        raise TypeError("E8 config must be a mapping")

    interactive_config = InteractiveMarketConfig(**payload.get("interactive", {}))
    actor_world = InteractiveMarketWorld(interactive_config)
    transition = _transition(payload)
    seeds = [int(seed) for seed in payload.get("seeds", [])]
    if not seeds:
        raise ValueError("E8 config must include at least one seed")
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        context = [RolloutContext(seed=seed, scenario_id=f"e8-{seed}")]
        actor = PairedEvaluator(actor_world, mode="actor").evaluate(transition, context)
        for config in _strategic_grid(payload, interactive_config):
            world = StrategicMarketWorld(config)
            strategic = PairedEvaluator(world, mode="strategic").evaluate(transition, context)
            rows.append(
                {
                    "transition_id": transition.transition_id,
                    "seed": seed,
                    "mode": config.opponent_mode,
                    "opponent_count": config.opponent_count,
                    "adaptation_steps": config.adaptation_steps,
                    "learning_rate": config.learning_rate,
                    "market_share_sensitivity": config.market_share_sensitivity,
                    "strategic_sensitivity": world.strategic_sensitivity(
                        transition.incumbent, transition.candidate
                    ),
                    "delta_actor": actor.delta,
                    "delta_strategic": strategic.delta,
                    "competition_effect": strategic.delta - actor.delta,
                    "strategic_improvement_reversal": (
                        actor.delta > 0 and strategic.delta < 0
                    ),
                    "value": strategic.candidate_value,
                    "paired": True,
                }
            )

    if not rows:
        raise ValueError("E8 config produced no rows")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "competition.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    provenance = {
        "config": payload,
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "seeds": seeds,
        "transition_id": transition.transition_id,
        "paired": True,
        "virtual_fills": True,
        "environment_version": "strategic-fixture-v1",
        "row_count": len(rows),
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"E8 rows={len(rows)} output={args.output}")


def _transition(payload: dict[str, Any]) -> PolicyTransition:
    return PolicyTransition(
        incumbent=Policy.from_mapping(payload.get("incumbent", {})),
        candidate=Policy.from_mapping(payload.get("candidate", {})),
        round_id=0,
        candidate_index=0,
        improvement_operator="typed_finance_fixture",
        edit_type="intensity_and_position_size",
        proxy_world_id="F2-interactive-actor",
        high_fidelity_world_id="F4-strategic-market",
        config_id="strategic-e8-v1",
    )


def _strategic_grid(
    payload: dict[str, Any], interactive: InteractiveMarketConfig
) -> list[StrategicMarketConfig]:
    modes = [str(value) for value in payload.get("opponent_modes", [])]
    counts = [int(value) for value in payload.get("opponent_counts", [])]
    steps = [int(value) for value in payload.get("adaptation_steps", [])]
    rates = [float(value) for value in payload.get("learning_rates", [])]
    sensitivities = [float(value) for value in payload.get("market_share_sensitivities", [])]
    if not modes or not counts or not steps or not rates or not sensitivities:
        raise ValueError("E8 sweep axes must not be empty")
    configs: list[StrategicMarketConfig] = []
    for mode_value in modes:
        if mode_value not in {"fixed", "reactive", "adaptive"}:
            raise ValueError(f"unknown opponent mode: {mode_value}")
        mode = cast(OpponentMode, mode_value)
        if mode == "fixed":
            for count in counts:
                configs.append(
                    StrategicMarketConfig(
                        interactive=interactive,
                        opponent_mode=mode,
                        opponent_count=count,
                    )
                )
        elif mode == "reactive":
            for count, sensitivity in product(counts, sensitivities):
                configs.append(
                    StrategicMarketConfig(
                        interactive=interactive,
                        opponent_mode=mode,
                        opponent_count=count,
                        market_share_sensitivity=sensitivity,
                    )
                )
        else:
            for count, step, rate, sensitivity in product(counts, steps, rates, sensitivities):
                configs.append(
                    StrategicMarketConfig(
                        interactive=interactive,
                        opponent_mode=mode,
                        opponent_count=count,
                        adaptation_steps=step,
                        learning_rate=rate,
                        market_share_sensitivity=sensitivity,
                    )
                )
    return configs


if __name__ == "__main__":
    main()
