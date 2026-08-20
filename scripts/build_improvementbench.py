#!/usr/bin/env python3
"""Build deterministic controlled ImprovementBench releases."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from improve_x.benchmark.dataset import ImprovementBenchDataset, ImprovementBenchRow
from improve_x.controlled import evaluate_transition, make_contexts, policy_from_config
from improve_x.operators.evolutionary import EvolutionaryMutation
from pivot.core.policy import Policy
from pivot.core.result import RolloutContext
from pivot.core.transition import PolicyTransition
from pivot.environments.performative.config import PerformativeConfig
from pivot.environments.performative.world import PerformativeWorld
from pivot.improvers.perturbation import SyntheticPerturbation
from pivot.improvers.rl_update import RLUpdateOperator

WORLD_LEVELS = ("observer", "actor", "strategic")
V2_OPERATORS = ("synthetic", "rl-update", "evolutionary-mutation")


@dataclass(frozen=True)
class _GeneratedCandidate:
    transition: PolicyTransition
    scale: float
    scale_index: int
    proposal_seed: int
    evaluated: Mapping[str, object]


def _world_row_record(evaluated: Mapping[str, object], world_level: str) -> dict[str, object]:
    row_record = dict(evaluated)
    if world_level == "observer":
        row_record["delta_true"] = row_record["delta_proxy"]
        row_record["true_incumbent_value"] = row_record["proxy_incumbent_value"]
        row_record["true_candidate_value"] = row_record["proxy_candidate_value"]
        row_record["delta_actor"] = None
        row_record["delta_strategic"] = None
    elif world_level == "actor":
        row_record["delta_true"] = row_record["delta_actor"]
        row_record["true_incumbent_value"] = row_record["actor_incumbent_value"]
        row_record["true_candidate_value"] = row_record["actor_candidate_value"]
        row_record["delta_strategic"] = None
    elif world_level == "strategic":
        row_record["delta_true"] = row_record["delta_strategic"]
        row_record["true_incumbent_value"] = row_record["strategic_incumbent_value"]
        row_record["true_candidate_value"] = row_record["strategic_candidate_value"]
    else:
        raise ValueError(f"unsupported world level: {world_level}")
    return row_record


def _append_transition_rows(
    dataset: ImprovementBenchDataset,
    transition: PolicyTransition,
    evaluated: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    base = {**transition.to_record(), **dict(evaluated)}
    for world_level in WORLD_LEVELS:
        adjusted = ImprovementBenchRow.from_transition(
            _world_row_record(base, world_level),
            world_level=world_level,
        )
        dataset.append(
            ImprovementBenchRow(
                **{
                    **adjusted.__dict__,
                    "transition_id": f"{adjusted.transition_id}-{world_level}",
                    "metadata": {**dict(adjusted.metadata), **dict(metadata)},
                }
            )
        )


def _build_v1(payload: Mapping[str, Any]) -> ImprovementBenchDataset:
    world_config = PerformativeConfig(**dict(payload.get("world", {})))
    world = PerformativeWorld(world_config)
    incumbent = policy_from_config(payload)
    scales = tuple(float(value) for value in payload.get("candidate_scales", (0.1,)))
    seeds = tuple(int(value) for value in payload.get("seeds", (0,)))
    contexts_per_transition = int(payload.get("contexts_per_transition", 1))
    operator = SyntheticPerturbation()
    dataset = ImprovementBenchDataset(
        metadata={
            "benchmark": "ImprovementBench",
            "world_config": dict(payload.get("world", {})),
            "seeds": list(seeds),
            "candidate_scales": list(scales),
        }
    )
    for round_id, seed in enumerate(seeds):
        for scale_index, scale in enumerate(scales):
            transition = operator.propose(
                incumbent,
                scale=scale,
                num_candidates=1,
                seed=seed,
                round_id=round_id,
                config_id=world_config.config_id,
            )[0]
            # Stable indexes define one within-round candidate pool.
            transition = replace(transition, candidate_index=scale_index)
            evaluated = evaluate_transition(
                world,
                transition,
                make_contexts(seed, contexts_per_transition, "bench"),
            )
            _append_transition_rows(
                dataset,
                transition,
                evaluated,
                {
                    "source_transition_id": transition.transition_id,
                    "scale_index": scale_index,
                    "scale": scale,
                },
            )
    return dataset


def _parse_splits(payload: Mapping[str, Any]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, Mapping) or not raw_splits:
        raise ValueError("multiround_multioperator mode requires non-empty splits")
    parsed: list[tuple[str, tuple[int, ...]]] = []
    for name, values in raw_splits.items():
        split_name = str(name)
        if not split_name:
            raise ValueError("split names must not be empty")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
            raise ValueError(f"split {split_name} must contain at least one seed")
        parsed.append((split_name, tuple(int(value) for value in values)))
    return tuple(parsed)


def _candidate_seed(seed: int, round_id: int, operator_index: int, scale_index: int) -> int:
    return seed * 10_000 + round_id * 100 + operator_index * 10 + scale_index


def _propose_v2_candidates(
    incumbent: Policy,
    *,
    round_id: int,
    seed: int,
    scales: Sequence[float],
    parents: Sequence[Policy],
    config_id: str,
) -> tuple[tuple[PolicyTransition, float, int, int], ...]:
    candidates: list[tuple[PolicyTransition, float, int, int]] = []
    for operator_index, operator_name in enumerate(V2_OPERATORS):
        for scale_index, scale in enumerate(scales):
            proposal_seed = _candidate_seed(seed, round_id, operator_index, scale_index)
            if operator_name == "synthetic":
                transition = SyntheticPerturbation().propose(
                    incumbent,
                    scale=scale,
                    num_candidates=1,
                    seed=proposal_seed,
                    round_id=round_id,
                    config_id=config_id,
                )[0]
            elif operator_name == "rl-update":
                context = RolloutContext(
                    seed=proposal_seed,
                    scenario_id=f"improvementbench-v2-update-{round_id}-{proposal_seed}",
                )
                transition = RLUpdateOperator(step_size=scale).propose(
                    incumbent,
                    context,
                    num_candidates=1,
                    seed=proposal_seed,
                    round_id=round_id,
                    config_id=config_id,
                )[0]
            else:
                transition = EvolutionaryMutation(mutation_scale=scale).propose(
                    incumbent,
                    round_id,
                    proposal_seed,
                    num_candidates=1,
                    parents=parents,
                    config_id=config_id,
                ).candidates[0]
            candidates.append((transition, scale, scale_index, proposal_seed))
    return tuple(
        (replace(transition, candidate_index=index), scale, scale_index, proposal_seed)
        for index, (transition, scale, scale_index, proposal_seed) in enumerate(candidates)
    )


def _required_float(record: Mapping[str, object], field: str) -> float:
    value = record.get(field)
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    return float(value)


def _build_v2(payload: Mapping[str, Any]) -> ImprovementBenchDataset:
    world_config = PerformativeConfig(**dict(payload.get("world", {})))
    world = PerformativeWorld(world_config)
    splits = _parse_splits(payload)
    rounds = int(payload.get("rounds", 1))
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    scales = tuple(float(value) for value in payload.get("candidate_scales", (0.1,)))
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("candidate_scales must be finite positive values")
    contexts_per_transition = int(payload.get("contexts_per_transition", 1))
    if contexts_per_transition <= 0:
        raise ValueError("contexts_per_transition must be positive")
    promotion_world = str(payload.get("collection_promotion_world", "actor"))
    if promotion_world != "actor":
        raise ValueError("controlled v2 collection promotion must be actor")
    dataset = ImprovementBenchDataset(
        metadata={
            "benchmark": "ImprovementBench",
            "release_version": "v2",
            "mode": "multiround_multioperator",
            "world_config": dict(payload.get("world", {})),
            "splits": {name: list(seeds) for name, seeds in splits},
            "rounds": rounds,
            "candidate_scales": list(scales),
            "operators": list(V2_OPERATORS),
            "collection_promotion_world": promotion_world,
        }
    )
    for split_name, seeds in splits:
        for seed in seeds:
            trajectory_id = f"{split_name}-{seed}"
            incumbent = policy_from_config(payload)
            parents: tuple[Policy, ...] = ()
            for round_id in range(rounds):
                generated: list[_GeneratedCandidate] = []
                for transition, scale, scale_index, proposal_seed in _propose_v2_candidates(
                    incumbent,
                    round_id=round_id,
                    seed=seed,
                    scales=scales,
                    parents=parents,
                    config_id=world_config.config_id,
                ):
                    evaluated = evaluate_transition(
                        world,
                        transition,
                        make_contexts(seed + round_id, contexts_per_transition, trajectory_id),
                    )
                    generated.append(
                        _GeneratedCandidate(
                            transition=transition,
                            scale=scale,
                            scale_index=scale_index,
                            proposal_seed=proposal_seed,
                            evaluated=evaluated,
                        )
                    )
                winner = max(
                    generated,
                    key=lambda item: (
                        _required_float(item.evaluated, "delta_actor"),
                        -item.transition.candidate_index,
                    ),
                )
                selected_source_id = winner.transition.transition_id
                for item in generated:
                    _append_transition_rows(
                        dataset,
                        item.transition,
                        item.evaluated,
                        {
                            "source_transition_id": item.transition.transition_id,
                            "trajectory_id": trajectory_id,
                            "split": split_name,
                            "scale_index": item.scale_index,
                            "scale": item.scale,
                            "proposal_seed": item.proposal_seed,
                            "collection_promotion_world": promotion_world,
                            "collection_selected": item.transition.transition_id == selected_source_id,
                            "collection_selected_source_transition_id": selected_source_id,
                        },
                    )
                parents = (incumbent, *parents)[:2]
                incumbent = winner.transition.candidate
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a controlled ImprovementBench release")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("benchmark config must be a mapping")
    mode = str(payload.get("mode", "single_round"))
    if mode == "single_round":
        dataset = _build_v1(payload)
    elif mode == "multiround_multioperator":
        dataset = _build_v2(payload)
    else:
        raise ValueError(f"unsupported benchmark mode: {mode}")
    manifest = dataset.write(args.output, created_at="2026-08-20T00:00:00+00:00")
    print(
        json.dumps(
            {"rows": len(dataset.rows), "output": str(args.output), "manifest": manifest},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
