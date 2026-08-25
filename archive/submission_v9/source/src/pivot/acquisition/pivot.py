from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from .common import candidate_id, candidate_value, validate_budget


class CorrectionModel(Protocol):
    def predict_correction(self, candidate: Any) -> Any:
        ...


def select_pivot(
    candidates: Sequence[Any],
    model: CorrectionModel,
    budget: int,
    cost_key: str = "hf_query_cost",
) -> list[str]:
    """Approximate decision-change VOI per high-fidelity cost.

    The score is intentionally transparent: uncertainty and predicted sign
    instability are multiplied by the candidate's opportunity value and
    normalized by query cost.  It is an acquisition heuristic, not a policy
    authorization rule or a claim of Bayesian optimality.
    """

    validate_budget(candidates, budget)
    predicted: dict[str, float] = {}
    uncertainty: dict[str, float] = {}
    sign_risk: dict[str, float] = {}
    for candidate in candidates:
        identifier = candidate_id(candidate)
        prediction = model.predict_correction(candidate)
        predicted[identifier] = float(getattr(prediction, "predicted_delta", candidate_value(candidate, "delta_proxy")))
        uncertainty[identifier] = max(0.0, float(getattr(prediction, "standard_deviation", 0.0)))
        sign_risk[identifier] = max(0.0, min(1.0, float(getattr(prediction, "sign_change_probability", 0.0))))
    best = max(predicted.values(), default=0.0)
    ordered: list[tuple[float, str]] = []
    for candidate in candidates:
        identifier = candidate_id(candidate)
        cost = max(candidate_value(candidate, cost_key, 1.0), 1e-12)
        opportunity = max(0.0, best - predicted[identifier]) + abs(predicted[identifier])
        score = (uncertainty[identifier] * (0.5 + sign_risk[identifier]) * (1.0 + opportunity)) / cost
        ordered.append((score, identifier))
    ordered.sort(key=lambda item: (-item[0], item[1]))
    return [identifier for _, identifier in ordered[:budget]]
