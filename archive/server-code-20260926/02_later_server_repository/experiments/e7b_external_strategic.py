#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.environments.external_adaptive.mpe2_world import (
    MPE2Config,
    MPE2Policy,
    MPE2World,
    generate_mpe2_candidates,
)
from pivot.evaluation.uncertainty import bootstrap_mean_ci
from pivot.research.state import classify_experiment
from pivot.transfer.differential import DifferentialModel


def main() -> None:
    parser = argparse.ArgumentParser(description="V7 E7b external adaptive strategic response")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--phase", choices=["development", "validation", "confirmatory"], default=None)
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("E7b config must be a mapping")
    seed_count = int(args.seeds or len(payload.get("seeds", [])))
    candidate_count = int(args.candidates or payload.get("candidates_per_seed", 8))
    phase = str(args.phase or payload.get("phase", "development"))
    if phase == "confirmatory" and payload.get("confirmatory_seed_count") is not None:
        start = int(payload.get("confirmatory_seed_start", payload.get("seeds", [0])[0]))
        count = int(payload["confirmatory_seed_count"])
        if count <= 0:
            raise ValueError("confirmatory_seed_count must be positive")
        seeds = list(range(start, start + count))
    else:
        seeds = [int(seed) for seed in payload.get("seeds", [])][:seed_count]
    strengths = [float(value) for value in payload.get("adaptation_strengths", [])]
    if len(seeds) < 2 or candidate_count < 2 or not strengths:
        raise ValueError("E7b needs at least two seeds, candidates, and adaptation strengths")
    config = MPE2Config(
        max_cycles=int(payload.get("max_cycles", 12)),
        scenario=str(payload.get("scenario", "simple_adversary_v3")),
        observation_dim=int(payload.get("observation_dim", 10)),
    )
    world = MPE2World(config)
    rows: list[dict[str, Any]] = []
    for family_index, (family_name, opponent_family) in enumerate(
        (("family_a", "reactive-observation"), ("family_b", "explicit-best-response"))
    ):
        for seed in seeds:
            incumbent = MPE2Policy.random(seed=seed + family_index * 1000, observation_dim=config.observation_dim, action_dim=5)
            candidates = generate_mpe2_candidates(
                incumbent,
                count=candidate_count,
                seed=seed + family_index * 1000,
                scale=float(payload.get("candidate_scale", 0.15)),
            )
            for strength in strengths:
                for index, candidate in enumerate(candidates):
                    actor_incumbent = world.evaluate_actor(incumbent, seed=seed, opponent_family=opponent_family).value
                    actor_candidate = world.evaluate_actor(candidate, seed=seed, opponent_family=opponent_family).value
                    strategic_incumbent = world.evaluate_actor(incumbent, seed=seed, opponent_bias=strength, opponent_family=opponent_family).value
                    strategic_candidate = world.evaluate_actor(candidate, seed=seed, opponent_bias=strength, opponent_family=opponent_family).value
                    rows.append(
                        {
                            "transition_id": f"{family_name}-{seed}-{strength:g}-{index}-{candidate.policy_id}",
                            "trajectory_id": f"{family_name}|seed={seed}|strength={strength:g}",
                            "operator_id": "typed-external-strategic-candidate",
                            "opponent_family": family_name,
                            "opponent_mechanism": opponent_family,
                            "opponent_seed": seed,
                            "adaptation_strength": strength,
                            "candidate_index": index,
                            "delta_actor": actor_candidate - actor_incumbent,
                            "delta_strategic": strategic_candidate - strategic_incumbent,
                            "strategic_effect": (strategic_candidate - strategic_incumbent) - (actor_candidate - actor_incumbent),
                            "delta_proxy": actor_candidate - actor_incumbent,
                            "response_strength": strength,
                            "competition_strength": strength,
                            "update_footprint": incumbent.distance(candidate),
                            "hf_queried": True,
                            "paired_rollout_ids": [f"{family_name}:{seed}:{strength:g}"],
                        }
                    )
    family_rows = {
        family: [row for row in rows if row["opponent_family"] == family]
        for family in ("family_a", "family_b")
    }
    family_metrics = {family: _metrics(items) for family, items in family_rows.items()}
    train = family_rows["family_a"]
    test = family_rows["family_b"]
    correction_model = DifferentialModel()
    correction_model.fit(train, [float(row["delta_strategic"] - row["delta_actor"]) for row in train])
    predictions = [
        correction_model.predict_correction(row).predicted_delta for row in test
    ]
    true = [float(row["delta_strategic"]) for row in test]
    cross_mae = float(np.mean(np.abs(np.asarray(predictions) - np.asarray(true))))
    actor_positive = [row for row in rows if float(row["delta_actor"]) > 1e-9]
    all_strengths = len(rows)
    cluster_effects = _cluster_effects(family_rows["family_b"])
    minimum = float(payload.get("minimum_effect_strategic", 0.5))
    cluster_sigma = float(np.std(cluster_effects, ddof=1)) if len(cluster_effects) > 1 else 0.0
    required_clusters = _required_cluster_mean_samples(
        sigma=max(cluster_sigma, 1e-9),
        margin=minimum,
        alpha=float(payload.get("alpha", 0.05)),
        power=float(payload.get("power", 0.8)),
    )
    if not actor_positive:
        state = classify_experiment(design_invalid=True, reason="no actor-positive focal updates in external strategic pool")
    elif phase != "confirmatory" or len(cluster_effects) < required_clusters:
        state = classify_experiment(
            underpowered=True,
            reason=(
                f"opponent-seed clusters {len(cluster_effects)} are below the registered power rule "
                f"({required_clusters})"
            ),
        )
    else:
        effect = float(np.mean(cluster_effects))
        state = classify_experiment(
            hypothesis_supported=effect <= -minimum,
            confirmatory=True,
            reason=(f"strategic effect {effect:.6g} is below the registered threshold" if effect <= -minimum else f"strategic effect {effect:.6g} does not pass the registered threshold"),
        )
    metrics = {
        "experiment": "e7b_external_strategic",
        "environment_source": "Farama MPE2",
        "phase": phase,
        "families": family_metrics,
        "cross_family": {
            "split_contract": "opponent-family-disjoint",
            "train_family": "family_a",
            "test_family": "family_b",
            "transition_correction_mae": cross_mae,
            "test_rows": len(test),
        },
        "actor_positive_rows": len(actor_positive),
        "family_b_cluster_effects": cluster_effects,
        "family_b_cluster_count": len(cluster_effects),
        "power_required_opponent_seed_clusters": required_clusters,
        "minimum_effect_strategic": minimum,
        "total_rows": all_strengths,
        "state": state.state.value,
        "state_reason": state.reason,
        "paired": True,
        "cluster_units": ["opponent_family", "opponent_seed"],
    }
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "transition_rows.jsonl", rows, jsonl=True)
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "state.json", {"state": state.state.value, "reason": state.reason})
    _write_json(
        output / "provenance.json",
        {
            "config": payload,
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "environment_source": "Farama MPE2",
            "opponent_families": {
                "family_a": "reactive-observation",
                "family_b": "explicit-best-response",
            },
            "held_out_adaptation_strengths": strengths,
            "paired": True,
        },
    )
    _write_json(output / "manifest.json", _manifest(output))
    print(json.dumps({"experiment": "e7b_external_strategic", "rows": len(rows), "state": state.state.value}, sort_keys=True))


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actor = [float(row["delta_actor"]) for row in rows]
    strategic = [float(row["delta_strategic"]) for row in rows]
    positive = [row for row in rows if float(row["delta_actor"]) > 1e-9]
    sir = None if not positive else sum(float(row["delta_strategic"]) < -1e-9 for row in positive) / len(positive)
    effects = [float(row["strategic_effect"]) for row in positive]
    ci = list(bootstrap_mean_ci(effects, seed=20260825)) if effects else [None, None]
    return {
        "rows": len(rows),
        "actor_mean": float(np.mean(actor)),
        "strategic_mean": float(np.mean(strategic)),
        "strategic_effect_mean": float(np.mean(effects)) if effects else None,
        "strategic_effect_ci": ci,
        "sir": sir,
        "actor_positive_rows": len(positive),
    }


def _cluster_effects(rows: list[dict[str, Any]]) -> list[float]:
    """Average conditional strategic effects within each held-out opponent seed."""

    grouped: dict[int, list[float]] = {}
    for row in rows:
        if float(row["delta_actor"]) > 1e-9:
            grouped.setdefault(int(row["opponent_seed"]), []).append(float(row["strategic_effect"]))
    return [float(np.mean(values)) for _, values in sorted(grouped.items()) if values]


def _required_cluster_mean_samples(*, sigma: float, margin: float, alpha: float, power: float) -> int:
    from statistics import NormalDist

    critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    target = NormalDist().inv_cdf(power)
    return max(1, int(np.ceil(((critical + target) * sigma / margin) ** 2)))


def _manifest(directory: Path) -> dict[str, Any]:
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.iterdir()) if path.is_file() and path.name != "manifest.json"}
    return {"schema_version": "v7-e7b-1", "file_count": len(files), "files": files}


def _write_json(path: Path, payload: Any, *, jsonl: bool = False) -> None:
    if jsonl:
        path.write_text("".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in payload), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
