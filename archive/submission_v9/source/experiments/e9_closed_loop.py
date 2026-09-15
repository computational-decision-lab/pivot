#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.acquisition.pivot import select_pivot
from pivot.algorithms.pivot import run_pivot_round
from pivot.core.policy import Policy
from pivot.core.result import RolloutContext
from pivot.environments.performative.config import PerformativeConfig
from pivot.environments.performative.world import PerformativeWorld
from pivot.evaluation.paired import PairedEvaluator
from pivot.footprint.generic import compute_update_footprint
from pivot.improvers.perturbation import SyntheticPerturbation
from pivot.transfer.differential import DifferentialModel


class _ColdStartModel:
    """Conservative model for the first round before correction data exists."""

    def predict_correction(self, row):
        class Prediction:
            correction = 0.0
            standard_deviation = max(0.1, float(row.get("update_footprint", 0.0)))
            predicted_delta = float(row.get("delta_proxy", 0.0))
            sign_change_probability = 0.5

        return Prediction()

    def uncertainty(self, row):
        return self.predict_correction(row).standard_deviation


def main() -> None:
    parser = argparse.ArgumentParser(description="PIVOT E9 closed-loop self-improvement")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/raw/e9-closed-loop"))
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = PerformativeConfig(**payload.get("world", {}))
    world = PerformativeWorld(config)
    rounds = int(payload.get("rounds", 8))
    budget = int(payload.get("hf_budget", 1))
    scales = [float(value) for value in payload.get("candidate_scales", [0.05, 0.1, 0.2])]
    seed = int((payload.get("seeds") or [1])[0])
    context = RolloutContext(seed=seed, scenario_id="e9")
    incumbent = Policy.from_mapping({"intensity": 0.2})
    history: list[dict[str, object]] = []
    all_transition_rows: list[dict[str, object]] = []
    queried_rows: list[dict[str, object]] = []
    model: object = _ColdStartModel()
    operator = SyntheticPerturbation()
    for round_id in range(rounds):
        transitions = operator.propose(incumbent, scales[0], len(scales), seed=seed + round_id, round_id=round_id, config_id=config.config_id)
        candidates: list[dict[str, object]] = []
        for transition in transitions:
            proxy = PairedEvaluator(world, mode="observer").evaluate(transition, [context])
            footprint = compute_update_footprint(transition.incumbent, transition.candidate, [-1.0, 0.0, 1.0])
            candidates.append(
                {
                    **transition.to_record(),
                    "delta_proxy": proxy.delta,
                    "proxy_incumbent_value": proxy.incumbent_value,
                    "proxy_candidate_value": proxy.candidate_value,
                    "update_footprint": footprint.distance,
                    "footprint_components": footprint.components,
                    "hf_query_cost": 1.0,
                }
            )

        transition_by_id = {transition.transition_id: transition for transition in transitions}

        def hf(row, lookup=transition_by_id):
            transition = lookup[str(row["transition_id"])]
            actor = PairedEvaluator(world, mode="actor").evaluate(transition, [context])
            return {
                "delta_true": actor.delta,
                "true_incumbent_value": actor.incumbent_value,
                "true_candidate_value": actor.candidate_value,
                "hf_query_cost": float(actor.environment_steps),
            }

        result = run_pivot_round(None, candidates, None, hf, select_pivot, budget, model=model)
        all_transition_rows.extend(dict(row) for row in result.rows)
        selected = next(row for row in result.rows if row["transition_id"] == result.selected_candidate_id)
        incumbent = Policy.from_mapping(selected["candidate_parameters"])
        history.append(
            {
                "round_id": round_id,
                "incumbent_policy_id": selected["incumbent_policy_id"],
                "selected_candidate_id": result.selected_candidate_id,
                "selected_delta_true": result.selected_delta_true,
                "selected_delta_estimate": result.selected_delta_estimate,
                "hf_budget": result.hf_budget,
                "hf_cost": result.hf_cost,
                "update_selection_regret": result.update_selection_regret,
            }
        )
        queried_rows.extend(dict(row) for row in result.rows if row.get("hf_queried"))
        if queried_rows:
            model = DifferentialModel()
            model.fit(queried_rows, [float(row["delta_true"]) - float(row["delta_proxy"]) for row in queried_rows])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "closed_loop.json").write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")
    (args.output / "transition_rows.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in all_transition_rows), encoding="utf-8")
    (args.output / "hf_query_rows.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in queried_rows), encoding="utf-8")
    (args.output / "provenance.json").write_text(json.dumps({"config": config.__dict__, "rounds": rounds, "hf_budget_per_round": budget, "seed": seed}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"E9 rounds={len(history)} queried={len(queried_rows)} output={args.output}")


if __name__ == "__main__":
    main()
