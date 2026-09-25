#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.core.policy import Policy
from pivot.core.result import RolloutContext
from pivot.environments.performative.config import PerformativeConfig
from pivot.environments.performative.world import PerformativeWorld
from pivot.evaluation.paired import PairedEvaluator
from pivot.research.state import classify_experiment
from pivot.theory.operator_shift import operator_shift_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="V7 E2 operator-distribution shift diagnostic")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("E2 config must be a mapping")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    rows = _make_population(payload)
    temperatures = [float(value) for value in payload.get("operator_temperatures", [])]
    if not temperatures or any(value <= 0.0 or not math.isfinite(value) for value in temperatures):
        raise ValueError("operator_temperatures must be finite and positive")
    summaries = [_summarize_temperature(rows, temperature) for temperature in temperatures]
    summaries.sort(key=lambda item: float(item["temperature"]))
    minimum_effect = float(payload.get("minimum_shift_effect", 0.001))
    temperature_contrast = float(payload.get("minimum_temperature_contrast", 0.1))
    shift_effect = float(summaries[-1]["operator_if"] - summaries[0]["operator_if"])
    chi_contrast = float(summaries[-1]["chi_square_divergence"] - summaries[0]["chi_square_divergence"])
    supported = shift_effect >= minimum_effect and chi_contrast >= temperature_contrast
    classification = classify_experiment(
        hypothesis_supported=supported,
        confirmatory=True,
        reason=(
            "concentrated operator increases weighted IF and chi-square shift"
            if supported
            else "registered temperature sweep does not establish the shift contrast"
        ),
    )
    metrics = {
        "experiment": "e2_operator_shift",
        "population_size": len(rows),
        "temperature_count": len(summaries),
        "minimum_temperature_if": float(summaries[0]["operator_if"]),
        "maximum_temperature_if": float(summaries[-1]["operator_if"]),
        "operator_if_effect": shift_effect,
        "chi_square_effect": chi_contrast,
        "state": classification.state.value,
        "state_reason": classification.reason,
        "q_a_definition": "softmax(beta * centered proxy improvement) over the frozen population",
    }
    _write_json(output / "population_rows.jsonl", rows, jsonl=True)
    _write_json(output / "temperature_summary.json", summaries)
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "state.json", {"state": classification.state.value, "reason": classification.reason})
    (output / "README.md").write_text(
        "# E2 Operator Shift\n\n"
        "This registered controlled diagnostic uses a frozen transition population and "
        "a temperature-softmax operator law. It is mechanism evidence only; it is not "
        "external causal or strategic validation.\n",
        encoding="utf-8",
    )
    _write_json(
        output / "provenance.json",
        {
            "config": payload,
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "git_commit": _git_commit(),
            "paired": True,
            "population_law": "uniform over frozen transition rows",
            "operator_law": "temperature-softmax over proxy improvement",
            "environment_version": "pivot-controlled-v1",
        },
    )
    manifest = _manifest(output)
    _write_json(output / "manifest.json", manifest)
    print(json.dumps({"experiment": "e2_operator_shift", "rows": len(rows), "state": classification.state.value}, sort_keys=True))


def _make_population(payload: dict[str, Any]) -> list[dict[str, Any]]:
    world_config = PerformativeConfig(**dict(payload.get("world", {})))
    seeds = [int(seed) for seed in payload.get("seeds", [])]
    scales = [float(scale) for scale in payload.get("candidate_scales", [])]
    responses = [float(value) for value in payload.get("response_strengths", [world_config.response_strength])]
    if not seeds or not scales or not responses:
        raise ValueError("E2 population grids must not be empty")
    incumbent = Policy.from_mapping({"intensity": 0.2})
    rows: list[dict[str, Any]] = []
    for response in responses:
        world = PerformativeWorld(
            PerformativeConfig(
                response_strength=response,
                competition_strength=world_config.competition_strength,
                noise_scale=world_config.noise_scale,
                horizon=world_config.horizon,
                reward_bound=world_config.reward_bound,
                decay=world_config.decay,
                optimization_strength=world_config.optimization_strength,
                config_id=f"{world_config.config_id}:response={response:g}",
            )
        )
        for seed in seeds:
            context = RolloutContext(seed=seed, scenario_id=f"e2-{seed}")
            for index, scale in enumerate(scales):
                candidate = Policy.from_mapping({"intensity": min(0.95, max(-0.95, 0.2 + scale))})
                from pivot.core.transition import PolicyTransition

                transition = PolicyTransition(
                    incumbent=incumbent,
                    candidate=candidate,
                    round_id=0,
                    candidate_index=index,
                    improvement_operator="synthetic-temperature-law",
                    edit_type="intensity",
                    seed=seed,
                    config_id=world.config.config_id,
                )
                observer = PairedEvaluator(world, mode="observer").evaluate(transition, [context])
                actor = PairedEvaluator(world, mode="actor").evaluate(transition, [context])
                rows.append(
                    {
                        "transition_id": transition.transition_id,
                        "seed": seed,
                        "response_strength": response,
                        "candidate_index": index,
                        "delta_proxy": observer.delta,
                        "delta_actor": actor.delta,
                        "loss": abs(observer.delta - actor.delta),
                        "sign_error": float(_sign(observer.delta) != _sign(actor.delta)),
                    }
                )
    if len(rows) < 4:
        raise ValueError("E2 population must contain at least four transitions")
    uniform = 1.0 / len(rows)
    for row in rows:
        row["population_weight"] = uniform
    return rows


def _summarize_temperature(rows: list[dict[str, Any]], temperature: float) -> dict[str, Any]:
    scores = np.asarray([float(row["delta_proxy"]) for row in rows], dtype=float)
    centered = scores - float(scores.mean())
    logits = temperature * centered / max(float(scores.std()), 1e-12)
    logits -= float(logits.max())
    weights = np.exp(logits)
    weights /= float(weights.sum())
    weighted_rows = [
        {**row, "operator_weight": float(weight), "loss": float(row["loss"])}
        for row, weight in zip(rows, weights)
    ]
    absolute = operator_shift_summary(weighted_rows)
    sign = operator_shift_summary(
        [{**row, "loss": float(row["sign_error"])} for row in weighted_rows]
    )
    positive_mass = sum(
        float(row["operator_weight"])
        for row in weighted_rows
        if float(row["delta_proxy"]) > 1e-9
    )
    reversal_mass = sum(
        float(row["operator_weight"])
        for row in weighted_rows
        if float(row["delta_proxy"]) > 1e-9 and float(row["delta_actor"]) < -1e-9
    )
    return {
        "temperature": temperature,
        "operator_if": absolute["operator_if"],
        "absolute_ide": absolute["operator_if"],
        "sign_error_if": sign["operator_if"],
        "isc": 1.0 - float(sign["operator_if"]),
        "irr": None if positive_mass == 0.0 else reversal_mass / positive_mass,
        "chi_square_divergence": absolute["chi_square_divergence"],
        "effective_sample_size": absolute["effective_sample_size"],
        "operator_shift_bound": absolute["operator_shift_bound"],
        "bound_holds": bool(absolute["bound_holds"]),
        "n_transitions": len(rows),
    }


def _sign(value: float) -> int:
    if abs(value) <= 1e-9:
        return 0
    return 1 if value > 0 else -1


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _manifest(directory: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"schema_version": "v7-e2-1", "file_count": len(files), "files": files}


def _write_json(path: Path, payload: Any, *, jsonl: bool = False) -> None:
    if jsonl:
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in payload), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
