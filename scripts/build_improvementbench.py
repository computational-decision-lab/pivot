#!/usr/bin/env python3
"""Build a deterministic controlled ImprovementBench dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import yaml

from improve_x.benchmark.dataset import ImprovementBenchDataset, ImprovementBenchRow
from improve_x.controlled import evaluate_transition, make_contexts, policy_from_config
from pivot.environments.performative.config import PerformativeConfig
from pivot.environments.performative.world import PerformativeWorld
from pivot.improvers.perturbation import SyntheticPerturbation


def main() -> None:
    parser = argparse.ArgumentParser(description="Build controlled ImprovementBench v1")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("benchmark config must be a mapping")
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
            # The configured scale list defines the candidate group. Re-key the
            # single-candidate legacy proposal so ranking tasks see stable
            # within-round indices rather than three copies of index zero.
            transition = replace(transition, candidate_index=scale_index)
            evaluated = {**transition.to_record(), **evaluate_transition(world, transition, make_contexts(seed, contexts_per_transition, "bench"))}
            for world_level in ("observer", "actor", "strategic"):
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
                else:
                    row_record["delta_true"] = row_record["delta_strategic"]
                    row_record["true_incumbent_value"] = row_record["strategic_incumbent_value"]
                    row_record["true_candidate_value"] = row_record["strategic_candidate_value"]
                adjusted = ImprovementBenchRow.from_transition(row_record, world_level=world_level)
                dataset.append(
                    ImprovementBenchRow(
                        **{
                            **adjusted.__dict__,
                            "transition_id": f"{adjusted.transition_id}-{world_level}",
                            "metadata": {**dict(adjusted.metadata), "scale_index": scale_index, "scale": scale},
                        }
                    )
                )
    manifest = dataset.write(args.output, created_at="2026-08-20T00:00:00+00:00")
    print(json.dumps({"rows": len(dataset.rows), "output": str(args.output), "manifest": manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
