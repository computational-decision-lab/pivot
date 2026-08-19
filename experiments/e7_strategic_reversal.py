#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.core.policy import Policy
from pivot.core.result import RolloutContext
from pivot.core.transition import PolicyTransition
from pivot.environments.interactive_market.config import InteractiveMarketConfig
from pivot.environments.interactive_market.world import InteractiveMarketWorld
from pivot.environments.strategic_market.config import StrategicMarketConfig
from pivot.environments.strategic_market.world import StrategicMarketWorld
from pivot.evaluation.paired import PairedEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="PIVOT E7 strategic improvement reversal")
    parser.add_argument("--config", type=Path, default=Path("configs/sweeps/e7.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/raw/e7-strategic"))
    args = parser.parse_args()
    raw_config = args.config.read_bytes()
    payload = yaml.safe_load(raw_config)
    if not isinstance(payload, dict):
        raise TypeError("E7 config must be a mapping")

    interactive_config = InteractiveMarketConfig(**payload.get("interactive", {}))
    strategic_config = StrategicMarketConfig(
        interactive=interactive_config,
        **payload.get("strategic", {}),
    )
    actor_world = InteractiveMarketWorld(interactive_config)
    strategic_world = StrategicMarketWorld(strategic_config)
    transition = _transition(payload)
    seeds = [int(seed) for seed in payload.get("seeds", [])]
    if not seeds:
        raise ValueError("E7 config must include at least one seed")

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        contexts = [RolloutContext(seed=seed, scenario_id=f"e7-{seed}")]
        proxy = PairedEvaluator(actor_world, mode="observer").evaluate(transition, contexts)
        actor = PairedEvaluator(actor_world, mode="actor").evaluate(transition, contexts)
        strategic = PairedEvaluator(strategic_world, mode="strategic").evaluate(
            transition, contexts
        )
        rows.append(
            {
                "transition_id": transition.transition_id,
                "seed": seed,
                "delta_proxy": proxy.delta,
                "delta_actor": actor.delta,
                "delta_strategic": strategic.delta,
                "mechanical_effect": actor.delta - proxy.delta,
                "competition_effect": strategic.delta - actor.delta,
                "improvement_reversal": proxy.delta > 0 and actor.delta < 0,
                "strategic_improvement_reversal": actor.delta > 0 and strategic.delta < 0,
                "paired": True,
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "strategic_reversal.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_provenance(args.output, payload, raw_config, seeds, transition.transition_id)
    print(f"E7 rows={len(rows)} output={args.output}")


def _transition(payload: dict[str, Any]) -> PolicyTransition:
    return PolicyTransition(
        incumbent=Policy.from_mapping(payload.get("incumbent", {})),
        candidate=Policy.from_mapping(payload.get("candidate", {})),
        round_id=0,
        candidate_index=0,
        improvement_operator="typed_finance_fixture",
        edit_type="intensity_and_position_size",
        proxy_world_id="F1-execution-replay",
        high_fidelity_world_id="F4-strategic-market",
        config_id="strategic-e7-v1",
    )


def _write_provenance(
    output: Path,
    payload: dict[str, Any],
    raw_config: bytes,
    seeds: list[int],
    transition_id: str,
) -> None:
    provenance = {
        "config": payload,
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "seeds": seeds,
        "transition_id": transition_id,
        "paired": True,
        "virtual_fills": True,
        "environment_version": "strategic-fixture-v1",
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
