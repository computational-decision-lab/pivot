"""Explicit DEV acquisition controls with observable-only inputs."""

from __future__ import annotations

import copy
import math
from typing import Any

from pivot.acquisition.pivot_voi import score_pivot_voi
from pivot.algorithms.pivot import RoundResult


def run_batch(
    *,
    incumbent: Any,
    candidates: list[dict[str, Any]],
    posterior: Any,
    hf: Any,
    seed: int,
    budget: int,
) -> RoundResult:
    """Score/acquire once, observe the entire chosen batch, then condition."""
    rows = copy.deepcopy(sorted(candidates, key=lambda row: row["transition_id"]))
    trace = []
    model = posterior
    ledger = []

    def record(event: str, **details: Any) -> None:
        trace.append(
            {
                "event": event,
                "query_count": len(ledger),
                "remaining_budget": budget - len(ledger),
                "posterior_observations": model.n_observations,
                "posterior_version": f"calibrated-linear:n={model.n_observations}",
                "selection_probability": None,
                "selection_probability_lower": None,
                "incumbent_probability": None,
                "max_acquisition": None,
                "cumulative_hf_cost": sum(item["cost"] for item in ledger),
                **details,
            }
        )

    scores = score_pivot_voi(rows, posterior, seed=seed, fantasies=8, posterior_samples=32)
    identifiers = [str(score["transition_id"]) for score in scores[:budget]]
    record(
        event="batch_acquire",
        transition_ids=identifiers,
        acquisition_scores=scores,
        posterior_observations=posterior.n_observations,
    )
    for identifier in identifiers:
        row = next(row for row in rows if row["transition_id"] == identifier)
        record("query", transition_id=identifier)
        observation = hf(copy.deepcopy(row))
        delta = float(observation["delta_true"])
        cost = float(observation["hf_query_cost"])
        if not math.isfinite(delta) or not math.isfinite(cost) or cost < 0.0:
            raise ValueError("HF delta and cost must be finite, and cost nonnegative")
        row.update(observation)
        row["observed_delta"] = delta
        row["hf_queried"] = True
        ledger.append(
            {
                "transition_id": identifier,
                "cost": float(observation["hf_query_cost"]),
                "query_index": len(ledger) + 1,
                "paired": True,
            }
        )
        record(
            event="observe",
            transition_id=identifier,
            observed_delta=observation["delta_true"],
            cost=observation["hf_query_cost"],
        )
    for identifier in identifiers:
        row = next(row for row in rows if row["transition_id"] == identifier)
        before = model.n_observations
        options = (
            {"observation_variance": float(row["observation_variance"])}
            if row.get("observation_variance") is not None
            else {}
        )
        model = model.condition(row["features"], row["delta_true"] - row["delta_proxy"], **options)
        record(
            event="condition",
            transition_id=identifier,
            previous_posterior_observations=before,
            posterior_observations=model.n_observations,
        )
    incumbent_row = {
        "transition_id": "incumbent",
        "is_incumbent": True,
        "predicted_delta": 0.0,
        "delta_true": 0.0,
        "hf_queried": False,
    }
    selected = incumbent_row
    for row in rows:
        row["predicted_delta"] = (
            float(row["delta_true"])
            if row.get("hf_queried")
            else float(model.predict_correction(row).predicted_delta)
        )
        if row["predicted_delta"] > selected["predicted_delta"]:
            selected = row
    record(
        event="rescore",
        posterior_observations=model.n_observations,
        predictions={row["transition_id"]: row["predicted_delta"] for row in rows},
    )
    record("stop", stop_reason="fixed_batch_complete")
    for row in [incumbent_row, *rows]:
        row["selected"] = row is selected
    record(
        event="select",
        transition_id=selected["transition_id"],
        stop_reason="fixed_batch_complete",
        posterior_observations=model.n_observations,
    )
    return RoundResult(
        selected_candidate_id=selected["transition_id"],
        selected_delta_true=selected.get("delta_true"),
        selected_delta_estimate=selected["predicted_delta"],
        queried_ids=tuple(identifiers),
        query_ledger=tuple(ledger),
        rows=(incumbent_row, *rows),
        update_selection_regret=None,
        cti_delta=selected.get("delta_true"),
        hf_budget=budget,
        hf_cost=sum(item["cost"] for item in ledger),
        query_count=len(ledger),
        acquisition_method="pivot_voi_batch",
        stop_reason="fixed_batch_complete",
        audit_trace=tuple(trace),
        posterior=model,
    )
