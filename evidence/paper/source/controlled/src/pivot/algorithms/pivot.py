from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pivot.acquisition.common import candidate_id, validate_budget
from pivot.acquisition.pivot_voi import (
    BayesianLinearDeltaPosterior,
    score_pivot_voi,
    select_pivot_voi,
    should_stop,
)
from pivot.core.policy import Policy
from pivot.core.transition import PolicyTransition


@dataclass(frozen=True)
class RoundResult:
    selected_candidate_id: str | None
    selected_delta_true: float | None
    selected_delta_estimate: float | None
    queried_ids: tuple[str, ...]
    query_ledger: tuple[Mapping[str, Any], ...]
    rows: tuple[Mapping[str, Any], ...]
    update_selection_regret: float | None
    cti_delta: float | None
    hf_budget: int
    hf_cost: float
    acquisition_method: str = "unspecified"
    acquisition_scores: tuple[Mapping[str, Any], ...] = ()
    stop_reason: str | None = None
    posterior_version: str | None = None


def run_pivot_round(
    incumbent: Policy | None,
    candidates: Sequence[Mapping[str, Any] | PolicyTransition],
    proxy: Callable[[Any], Mapping[str, Any]] | None,
    hf: Callable[[Any], Mapping[str, Any] | float],
    acquisition: Callable[..., list[str]],
    budget: int,
    *,
    model: Any | None = None,
    acquisition_kwargs: Mapping[str, Any] | None = None,
    acquisition_method: str = "unspecified",
    acquisition_scores: Sequence[Mapping[str, Any]] = (),
    stop_reason: str | None = None,
    posterior_version: str | None = None,
) -> RoundResult:
    """Execute one budgeted transition-selection round.

    `proxy` and `hf` are explicit callables so the same orchestration can be
    used with controlled worlds, replay evaluators, or later strategic worlds.
    Every HF query is recorded; unqueried candidates remain proxy/model
    estimates and are never silently relabeled as ground truth.
    """

    _ = incumbent
    validate_budget(candidates, budget)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        base = candidate.to_record() if isinstance(candidate, PolicyTransition) else dict(candidate)
        if proxy is not None:
            base.update(dict(proxy(candidate)))
        base.setdefault("hf_queried", False)
        rows.append(base)
    extra = dict(acquisition_kwargs or {})
    if model is None:
        try:
            selected_ids = acquisition(rows, budget, **extra)
        except TypeError:
            selected_ids = acquisition(rows, budget=budget, **extra)
    else:
        try:
            selected_ids = acquisition(rows, model, budget, **extra)
        except TypeError:
            selected_ids = acquisition(rows, model=model, budget=budget, **extra)
    selected_set = set(selected_ids)
    if len(selected_set) != budget or not selected_set <= {candidate_id(row) for row in rows}:
        raise ValueError("acquisition must return exactly valid unique candidate IDs")
    ledger: list[Mapping[str, Any]] = []
    for row in rows:
        identifier = candidate_id(row)
        if identifier not in selected_set:
            if model is not None and hasattr(model, "predict_correction"):
                prediction = model.predict_correction(row)
                row["predicted_delta"] = float(prediction.predicted_delta)
            else:
                row["predicted_delta"] = float(row.get("delta_proxy", 0.0))
            continue
        result = hf(row)
        if isinstance(result, Mapping):
            row.update(dict(result))
            true_delta = result.get("delta_true", result.get("delta_actor"))
            cost = float(result.get("hf_query_cost", result.get("cost", 1.0)))
        else:
            true_delta = float(result)
            cost = 1.0
            row["delta_true"] = true_delta
        if true_delta is None:
            raise ValueError("HF result must contain delta_true or delta_actor")
        row["delta_true"] = float(true_delta)
        row["predicted_delta"] = float(true_delta)
        row["hf_queried"] = True
        row["hf_query_cost"] = cost
        ledger.append({"transition_id": identifier, "cost": cost, "paired": True})
    estimates = {candidate_id(row): float(row.get("predicted_delta", row.get("delta_proxy", 0.0))) for row in rows}
    selected_id = max(estimates, key=lambda identifier: estimates[identifier]) if estimates else None
    selected_row = next((row for row in rows if candidate_id(row) == selected_id), None)
    selected_true = None if selected_row is None else selected_row.get("delta_true")
    for row in rows:
        row["selected"] = candidate_id(row) == selected_id
    true_values = [row.get("delta_true") for row in rows]
    regret = None
    if all(value is not None for value in true_values) and selected_true is not None:
        numeric_true_values = [float(value) for value in true_values if value is not None]
        regret = float(max(numeric_true_values) - float(selected_true))
    return RoundResult(
        selected_candidate_id=selected_id,
        selected_delta_true=None if selected_true is None else float(selected_true),
        selected_delta_estimate=None if selected_id is None else float(estimates[selected_id]),
        queried_ids=tuple(selected_ids),
        query_ledger=tuple(ledger),
        rows=tuple(rows),
        update_selection_regret=regret,
        cti_delta=None if selected_true is None else float(selected_true),
        hf_budget=len(ledger),
        hf_cost=sum(float(item["cost"]) for item in ledger),
        acquisition_method=acquisition_method,
        acquisition_scores=tuple(dict(item) for item in acquisition_scores),
        stop_reason=stop_reason,
        posterior_version=posterior_version,
    )


def run_pivot_voi_round(
    incumbent: Policy | None,
    candidates: Sequence[Mapping[str, Any] | PolicyTransition],
    hf: Callable[[Any], Mapping[str, Any] | float],
    posterior: BayesianLinearDeltaPosterior,
    max_budget: int,
    *,
    seed: int = 0,
    delta: float = 0.05,
    eta: float = 0.0,
    fantasies: int = 64,
    posterior_samples: int = 256,
) -> RoundResult:
    """Run PIVOT-VOI with posterior-confidence/EVSI stopping.

    The acquisition scores are computed before a query. If the current
    posterior is already decisive, the method selects from model estimates and
    spends zero HF budget; otherwise it consumes at most ``max_budget``.
    """

    validate_budget(candidates, max_budget)
    scores = score_pivot_voi(
        candidates,
        posterior,
        seed=seed,
        fantasies=fantasies,
        posterior_samples=posterior_samples,
    )
    selection_probability = float(scores[0]["selection_probability"]) if scores else 1.0
    max_acquisition = max((float(item["acquisition"]) for item in scores), default=0.0)
    stop, reason = should_stop(
        selection_probability=selection_probability,
        max_acquisition=max_acquisition,
        delta=delta,
        eta=eta,
    )
    budget = 0 if stop else max_budget
    acquisition: Callable[..., list[str]]
    if budget == 0:
        acquisition = lambda rows, model, budget: []
    else:
        acquisition = select_pivot_voi
    return run_pivot_round(
        incumbent,
        candidates,
        proxy=None,
        hf=hf,
        acquisition=acquisition,
        budget=budget,
        model=posterior,
        acquisition_kwargs={
            "seed": seed,
            "fantasies": fantasies,
            "posterior_samples": posterior_samples,
        }
        if budget
        else None,
        acquisition_method="PIVOT-VOI",
        acquisition_scores=scores,
        stop_reason=reason if stop else None,
        posterior_version="bayesian-linear-v1",
    )
