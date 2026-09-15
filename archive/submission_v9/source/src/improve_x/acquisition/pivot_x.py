from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pivot.acquisition.common import candidate_id, candidate_value, validate_budget


class DecisionModel(Protocol):
    def predict_correction(self, candidate: Any) -> Any:
        """Return an object with predicted_delta and optional uncertainty fields."""


def score_decision_preservation(
    candidates: Sequence[Mapping[str, Any] | Any],
    model: DecisionModel,
    *,
    cost_key: str = "hf_query_cost",
) -> list[dict[str, float | str]]:
    """Score high-fidelity queries by estimated update-decision change.

    This is a transparent acquisition heuristic. It treats a candidate's
    standard deviation as uncertainty about its relative ordering, adds any
    model-provided sign-change probability, and normalizes by query cost. It
    does not claim calibrated Bayesian value of information.
    """

    if not candidates:
        return []
    predictions: dict[str, tuple[float, float, float]] = {}
    for candidate in candidates:
        identifier = candidate_id(candidate)
        prediction = model.predict_correction(candidate)
        predicted = float(getattr(prediction, "predicted_delta", candidate_value(candidate, "delta_proxy")))
        uncertainty = max(0.0, float(getattr(prediction, "standard_deviation", 0.0)))
        sign_risk = min(1.0, max(0.0, float(getattr(prediction, "sign_change_probability", 0.0))))
        if not math.isfinite(predicted) or not math.isfinite(uncertainty):
            raise ValueError("decision model predictions must be finite")
        predictions[identifier] = (predicted, uncertainty, sign_risk)
    ordered_predictions = sorted(predictions.items(), key=lambda item: (-item[1][0], item[0]))
    best_id = ordered_predictions[0][0]
    best_value = ordered_predictions[0][1][0]
    runner_up = ordered_predictions[1][1][0] if len(ordered_predictions) > 1 else best_value
    scores: list[dict[str, float | str]] = []
    for candidate in candidates:
        identifier = candidate_id(candidate)
        predicted, uncertainty, sign_risk = predictions[identifier]
        comparison_margin = (best_value - runner_up) if identifier == best_id else (best_value - predicted)
        decision_probability = _ordering_change_probability(comparison_margin, uncertainty)
        decision_probability = min(1.0, max(0.0, decision_probability + 0.5 * sign_risk))
        cost = max(candidate_value(candidate, cost_key, 1.0), 1e-12)
        score = decision_probability / cost
        scores.append(
            {
                "transition_id": identifier,
                "predicted_delta": predicted,
                "uncertainty": uncertainty,
                "sign_change_probability": sign_risk,
                "decision_change_probability": decision_probability,
                "margin_to_decision": comparison_margin,
                "cost": cost,
                "score": score,
                "best_predicted_id": best_id,
            }
        )
    scores.sort(key=lambda item: (-float(item["score"]), str(item["transition_id"])))
    return scores


def select_pivot_x(
    candidates: Sequence[Mapping[str, Any] | Any],
    model: DecisionModel,
    budget: int,
    *,
    cost_key: str = "hf_query_cost",
) -> list[str]:
    """Return exactly `budget` transition IDs for PIVOT-X queries."""

    validate_budget(candidates, budget)
    scores = score_decision_preservation(candidates, model, cost_key=cost_key)
    return [str(item["transition_id"]) for item in scores[:budget]]


def _ordering_change_probability(margin: float, uncertainty: float) -> float:
    if uncertainty <= 1e-12:
        return 1.0 if abs(margin) <= 1e-12 else 0.0
    normalized = abs(margin) / (uncertainty * math.sqrt(2.0))
    return math.exp(-(normalized * normalized))
