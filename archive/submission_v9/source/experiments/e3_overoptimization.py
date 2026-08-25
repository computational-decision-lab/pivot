#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.core.policy import Policy
from pivot.core.result import RolloutContext
from pivot.environments.performative.config import PerformativeConfig
from pivot.environments.performative.world import PerformativeWorld
from pivot.improvers.rl_update import RLUpdateOperator


def main() -> None:
    parser = argparse.ArgumentParser(description="PIVOT E3 Performative Overoptimization")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/raw/e3-overoptimization"))
    parser.add_argument("--rounds", type=int, default=12)
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = PerformativeConfig(**payload.get("world", {}))
    response = float((payload.get("response_strengths") or [config.response_strength])[0])
    world = PerformativeWorld(replace(config, response_strength=response))
    policy = Policy.from_mapping({"intensity": 0.2})
    operator = RLUpdateOperator()
    rows: list[dict[str, float | int | str]] = []
    context = RolloutContext(seed=int((payload.get("seeds") or [1])[0]), scenario_id="e3")
    for round_id in range(args.rounds):
        proxy = world.evaluate(policy, context, mode="observer").value
        true = world.evaluate(policy, context, mode="actor").value
        rows.append({"round_id": round_id, "policy_id": policy.policy_id, "proxy_value": proxy, "true_value": true})
        transition = operator.propose(
            policy,
            context,
            num_candidates=1,
            optimization_strength=config.optimization_strength,
            seed=context.seed + round_id,
            round_id=round_id,
            config_id=config.config_id,
        )[0]
        policy = transition.candidate
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "overoptimization.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"E3 rounds={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
