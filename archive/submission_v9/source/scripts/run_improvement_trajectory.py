#!/usr/bin/env python3
"""Run a controlled multi-round IMPROVE-X trajectory without hiding candidates."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import yaml

from improve_x.controlled import evaluate_transition, make_contexts, policy_from_config
from improve_x.core.operator import CandidateBatch, adapt_legacy_operator
from improve_x.core.trajectory import ImprovementTrajectory
from improve_x.failures.taxonomy import classify_failure
from pivot.environments.performative.config import PerformativeConfig
from pivot.environments.performative.world import PerformativeWorld
from pivot.improvers.perturbation import SyntheticPerturbation


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _required_float(mapping: dict[str, object], field: str) -> float:
    value = mapping.get(field)
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled IMPROVE-X multi-round trajectory")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("trajectory config must be a mapping")
    config = PerformativeConfig(**dict(payload.get("world", {})))
    world = PerformativeWorld(config)
    seed = int(payload.get("seed", 0))
    rounds = int(payload.get("rounds", 1))
    scales = tuple(float(value) for value in payload.get("candidate_scales", (0.1,)))
    if not scales or any(value < 0 for value in scales):
        raise ValueError("candidate_scales must contain at least one non-negative value")
    contexts_per_transition = int(payload.get("contexts_per_transition", 1))
    selection_world = str(payload.get("selection_world", "actor"))
    field = {"observer": "delta_proxy", "actor": "delta_actor", "strategic": "delta_strategic"}.get(selection_world)
    if field is None:
        raise ValueError("selection_world must be observer, actor, or strategic")
    trajectory = ImprovementTrajectory(policy_from_config(payload))
    operator = adapt_legacy_operator(SyntheticPerturbation())
    all_records: list[dict[str, object]] = []
    for round_id in range(rounds):
        transitions = tuple(
            replace(
                operator.propose(
                    trajectory.current_policy,
                    round_id,
                    seed + round_id,
                    scale=scale,
                    num_candidates=1,
                    config_id=config.config_id,
                ).candidates[0],
                candidate_index=index,
            )
            for index, scale in enumerate(scales)
        )
        batch = CandidateBatch(
            incumbent=trajectory.current_policy,
            candidates=transitions,
            operator=operator.operator_name,
            round_id=round_id,
            seed=seed + round_id,
            metadata={"candidate_scales": scales},
        )
        evaluations = tuple(
            evaluate_transition(world, transition, make_contexts(seed + round_id, contexts_per_transition, f"trajectory-{round_id}"))
            for transition in batch.candidates
        )
        selected_index = max(
            range(len(evaluations)), key=lambda index: _required_float(evaluations[index], field)
        )
        trajectory.append_round(
            batch,
            selected_index,
            evaluations,
            query_cost=sum(_required_float(value, "hf_query_cost") for value in evaluations),
        )
    for record in trajectory.to_records():
        record["failure_type"] = classify_failure(
            delta_proxy=_required_float(record, "delta_proxy"),
            delta_actor=_required_float(record, "delta_actor"),
            delta_strategic=_required_float(record, "delta_strategic"),
        ).value
        all_records.append(record)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rounds.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in all_records), encoding="utf-8"
    )
    summary = {
        "rounds": rounds,
        "selection_world": selection_world,
        "cumulative_true_improvement": trajectory.cumulative_true_improvement,
        "cumulative_actor_improvement": trajectory.cumulative_actor_improvement,
        "cumulative_strategic_improvement": trajectory.cumulative_strategic_improvement,
        "proxy_curve": trajectory.proxy_curve,
        "true_curve": trajectory.true_curve,
        "actor_curve": trajectory.actor_curve,
        "strategic_curve": trajectory.strategic_curve,
        "initial_policy_id": trajectory.initial_policy.policy_id,
        "final_policy_id": trajectory.current_policy.policy_id,
    }
    (args.output / "trajectory.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = {"config": payload, "git_commit": _commit(), "seed": seed, "world_config_id": config.config_id}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rounds": rounds, "rows": len(all_records), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
