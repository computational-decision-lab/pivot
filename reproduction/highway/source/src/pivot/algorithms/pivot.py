from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
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
from pivot.transfer.features import transition_feature_vector


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
    query_count: int = 0
    audit_trace: tuple[Mapping[str, Any], ...] = ()
    posterior: Any | None = None

    @property
    def hf_query_count(self) -> int:
        """Backward-compatible explicit name for the number of HF queries."""

        return self.query_count


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
    sequential: bool = False,
    update_posterior: bool = True,
    stop: Callable[..., bool | tuple[bool, str | None]] | None = None,
    score: Callable[..., Sequence[Mapping[str, Any]]] | None = None,
    incumbent_id: str = "incumbent",
) -> RoundResult:
    """Execute one budgeted transition-selection round.

    `proxy` and `hf` are explicit callables so the same orchestration can be
    used with controlled worlds, replay evaluators, or later strategic worlds.
    Every HF query is recorded; unqueried candidates remain proxy/model
    estimates and are never silently relabeled as ground truth.
    """

    if sequential:
        return run_pivot_round_sequential(
            incumbent,
            candidates,
            proxy,
            hf,
            acquisition,
            budget,
            model=model,
            acquisition_kwargs=acquisition_kwargs,
            stop=stop,
            score=score,
            acquisition_method=acquisition_method,
            acquisition_scores=acquisition_scores,
            stop_reason=stop_reason,
            posterior_version=posterior_version,
            incumbent_id=incumbent_id,
            update_posterior=update_posterior,
        )

    _ = incumbent
    validate_budget(candidates, budget)
    if not isinstance(incumbent_id, str) or not incumbent_id:
        raise ValueError("incumbent_id must be a non-empty string")
    if incumbent_id in {candidate_id(row) for row in candidates}:
        raise ValueError("incumbent_id collides with a candidate transition ID")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        base = candidate.to_record() if isinstance(candidate, PolicyTransition) else dict(candidate)
        if proxy is not None:
            base.update(dict(proxy(candidate)))
        if base.get("hf_queried") or base.get("observed_delta") is not None:
            raise ValueError("prequeried observations require a separate evidence ledger")
        base.setdefault("hf_queried", False)
        rows.append(base)
    rows.sort(key=candidate_id)
    extra = dict(acquisition_kwargs or {})
    decision_rows = _decision_rows(rows)
    if model is None:
        try:
            selected_ids = acquisition(decision_rows, budget, **extra)
        except TypeError:
            selected_ids = acquisition(decision_rows, budget=budget, **extra)
    else:
        try:
            selected_ids = acquisition(decision_rows, model, budget, **extra)
        except TypeError:
            selected_ids = acquisition(decision_rows, model=model, budget=budget, **extra)
    selected_set = set(selected_ids)
    if len(selected_set) != budget or not selected_set <= {candidate_id(row) for row in rows}:
        raise ValueError("acquisition must return exactly valid unique candidate IDs")
    ledger: list[Mapping[str, Any]] = []
    for row in rows:
        identifier = candidate_id(row)
        if identifier not in selected_set:
            if model is not None and hasattr(model, "predict_correction"):
                prediction = model.predict_correction(_decision_rows([row])[0])
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
        if not math.isfinite(float(true_delta)) or not math.isfinite(cost) or cost < 0:
            raise ValueError("HF delta and cost must be finite; cost must be nonnegative")
        row["delta_true"] = float(true_delta)
        row["predicted_delta"] = float(true_delta)
        row["observed_delta"] = float(true_delta)
        row["hf_queried"] = True
        row["hf_query_cost"] = cost
        ledger.append({"transition_id": identifier, "cost": cost, "paired": True})
    estimates = {
        incumbent_id: 0.0,
        **{
            candidate_id(row): float(row.get("predicted_delta", row.get("delta_proxy", 0.0)))
            for row in rows
        },
    }
    selected_id = max(estimates, key=lambda identifier: estimates[identifier])
    selected_row = next((row for row in rows if candidate_id(row) == selected_id), None)
    selected_true = (
        0.0
        if selected_id == incumbent_id
        else selected_row.get("delta_true")
        if selected_row is not None
        else None
    )
    for row in rows:
        row["selected"] = candidate_id(row) == selected_id
    true_values = [row.get("delta_true") for row in rows]
    regret = None
    if all(value is not None for value in true_values) and selected_true is not None:
        numeric_true_values = [float(value) for value in true_values if value is not None]
        regret = float(max([0.0, *numeric_true_values]) - float(selected_true))
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
        query_count=len(ledger),
        acquisition_method=acquisition_method,
        acquisition_scores=tuple(dict(item) for item in acquisition_scores),
        stop_reason=stop_reason,
        posterior_version=posterior_version,
    )


def run_pivot_round_sequential(
    incumbent: Policy | None,
    candidates: Sequence[Mapping[str, Any] | PolicyTransition],
    proxy: Callable[[Any], Mapping[str, Any]] | None,
    hf: Callable[[Any], Mapping[str, Any] | float],
    acquisition: Callable[..., list[str]],
    budget: int,
    *,
    model: Any | None = None,
    acquisition_kwargs: Mapping[str, Any] | None = None,
    stop: Callable[..., bool | tuple[bool, str | None]] | None = None,
    score: Callable[..., Sequence[Mapping[str, Any]]] | None = None,
    acquisition_method: str = "unspecified",
    acquisition_scores: Sequence[Mapping[str, Any]] = (),
    stop_reason: str | None = None,
    posterior_version: str | None = None,
    incumbent_id: str = "incumbent",
    update_posterior: bool = True,
) -> RoundResult:
    """Run an adaptive, one-query-at-a-time PIVOT round.

    Candidate rows are put in canonical transition-ID order before scoring so
    ties and acquisition callbacks cannot depend on their input order.  The
    synthetic incumbent is decision index ``j=0`` with a zero query cost and a
    known zero delta.  After every HF observation the model and candidate
    predictions are refreshed before the next query is selected. Setting
    update_posterior=False freezes shared coefficients for the no-update control;
    observed outcomes remain available and fantasy updates are unaffected.
    """

    validate_budget(candidates, budget)
    if not isinstance(incumbent_id, str) or not incumbent_id:
        raise ValueError("incumbent_id must be a non-empty string")
    candidate_rows = _prepare_rows(candidates, proxy)
    if incumbent_id in {candidate_id(row) for row in candidate_rows}:
        raise ValueError("incumbent_id collides with a candidate transition ID")
    candidate_rows.sort(key=candidate_id)
    for index, row in enumerate(candidate_rows, start=1):
        row["j"] = index
        row.setdefault("hf_queried", False)
        row.setdefault("query_cost", float(row.get("hf_query_cost", 1.0) or 1.0))
        row.setdefault("cost", float(row.get("hf_query_cost", row.get("query_cost", 1.0)) or 1.0))

    incumbent_row: dict[str, Any] = {
        "transition_id": incumbent_id,
        "candidate_index": 0,
        "j": 0,
        "delta_proxy": 0.0,
        "predicted_delta": 0.0,
        "delta_true": 0.0,
        "hf_queried": False,
        "hf_query_cost": 0.0,
        "query_cost": 0.0,
        "cost": 0.0,
        "is_incumbent": True,
    }
    rows: list[dict[str, Any]] = [incumbent_row, *candidate_rows]
    model_state = model
    ledger: list[Mapping[str, Any]] = []
    queried_ids: list[str] = []
    score_history: list[Mapping[str, Any]] = [dict(item) for item in acquisition_scores]
    audit_trace: list[Mapping[str, Any]] = []
    reason = stop_reason
    extra = dict(acquisition_kwargs or {})
    last_statistics: dict[str, Any] = {}

    def score_statistics(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not scores:
            return {}
        return {
            key: scores[0].get(key)
            for key in (
                "selection_probability",
                "selection_probability_lower",
                "selection_probability_upper",
                "incumbent_probability",
            )
        } | {
            "max_acquisition": max(float(item["acquisition"]) for item in scores),
            "max_acquisition_upper": max(
                (
                    float(item["acquisition_upper"])
                    for item in scores
                    if "acquisition_upper" in item
                ),
                default=None,
            ),
        }

    def record(event: str, **details: Any) -> None:
        observations = getattr(model_state, "n_observations", None)
        version = _posterior_version(model_state, posterior_version)
        audit_trace.append(
            {
                "event": event,
                "query_index": len(ledger),
                "transition_id": None,
                "query_count": len(ledger),
                "remaining_budget": budget - len(ledger),
                "posterior_observations": observations,
                "posterior_version": version,
                "posterior_version_before": version,
                "posterior_version_after": version,
                "selection_probability": None,
                "selection_probability_lower": None,
                "selection_probability_upper": None,
                "incumbent_probability": None,
                "max_acquisition": None,
                "max_acquisition_upper": None,
                "stop_reason": None,
                "hf_query_cost": 0.0,
                "update_posterior_enabled": update_posterior,
                "cumulative_hf_cost": sum(float(item["cost"]) for item in ledger),
                **details,
            }
        )

    if budget <= 0:
        _refresh_predictions(candidate_rows, model_state)
        last_statistics = score_statistics(acquisition_scores)
        record(
            "rescore",
            acquisition_scores=list(acquisition_scores),
            predictions={candidate_id(row): row["predicted_delta"] for row in candidate_rows},
            **last_statistics,
        )
    else:
        while len(ledger) < budget:
            remaining = budget - len(ledger)
            available = [row for row in candidate_rows if not bool(row.get("hf_queried", False))]
            available.sort(key=candidate_id)
            _refresh_predictions(available, model_state)
            if not available:
                reason = reason or "candidate_exhausted"
                break

            current_scores: Sequence[Mapping[str, Any]] = ()
            if score is not None:
                score_extra = dict(extra)
                if score is score_pivot_voi:
                    score_extra["decision_candidates"] = _decision_rows(candidate_rows)
                current_scores = _call_score(score, available, model_state, score_extra)
                score_history.extend(dict(item) for item in current_scores)
            last_statistics = score_statistics(current_scores)
            record(
                "rescore",
                acquisition_scores=list(current_scores),
                predictions={candidate_id(row): row["predicted_delta"] for row in candidate_rows},
                **last_statistics,
            )
            if stop is not None:
                should_stop_now, stop_detail = _call_stop(
                    stop,
                    rows,
                    model_state,
                    len(ledger),
                    remaining,
                    extra,
                )
                if should_stop_now:
                    reason = stop_detail or reason or "stopped"
                    break

            acquisition_extra = dict(extra)
            if acquisition is select_pivot_voi:
                acquisition_extra["decision_candidates"] = _decision_rows(candidate_rows)
            selected_ids = _call_acquisition(
                acquisition,
                available,
                model_state,
                1,
                acquisition_extra,
            )
            if not selected_ids:
                reason = reason or "acquisition"
                break
            if len(selected_ids) != 1:
                raise ValueError("sequential acquisition must return exactly one candidate ID")
            selected_id = str(selected_ids[0])
            available_ids = {candidate_id(row) for row in available}
            if selected_id not in available_ids:
                raise ValueError("acquisition must return one valid unqueried candidate ID")

            row = next(row for row in available if candidate_id(row) == selected_id)
            record(
                "query",
                transition_id=selected_id,
                query_index=len(ledger) + 1,
                cost=float(row.get("query_cost", 1.0)),
                expected_hf_query_cost=float(row.get("query_cost", 1.0)),
            )
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
            if not math.isfinite(float(true_delta)) or not math.isfinite(cost) or cost < 0:
                raise ValueError("HF delta and cost must be finite; cost must be nonnegative")
            row["delta_true"] = float(true_delta)
            row["predicted_delta"] = float(true_delta)
            row["observed_delta"] = float(true_delta)
            row["hf_queried"] = True
            row["hf_query_cost"] = cost
            row["query_cost"] = cost
            row["cost"] = cost
            queried_ids.append(selected_id)
            ledger.append(
                {
                    "transition_id": selected_id,
                    "cost": cost,
                    "paired": True,
                    "query_index": len(ledger) + 1,
                }
            )
            record(
                "observe",
                transition_id=selected_id,
                delta=float(true_delta),
                cost=cost,
                hf_query_cost=cost,
            )
            before_observations = getattr(model_state, "n_observations", None)
            before_version = _posterior_version(model_state, posterior_version)
            if update_posterior:
                model_state = _condition_model(model_state, row, float(true_delta))
            record(
                "condition",
                transition_id=selected_id,
                previous_posterior_observations=before_observations,
                posterior_version_before=before_version,
                updated=before_observations != getattr(model_state, "n_observations", None),
            )

    _refresh_predictions(candidate_rows, model_state)
    final_scores: Sequence[Mapping[str, Any]] = ()
    if score is not None and queried_ids:
        final_available = [row for row in candidate_rows if not bool(row.get("hf_queried", False))]
        final_available.sort(key=candidate_id)
        if final_available:
            score_extra = dict(extra)
            if score is score_pivot_voi:
                score_extra["decision_candidates"] = _decision_rows(candidate_rows)
            final_scores = _call_score(score, final_available, model_state, score_extra)
            score_history.extend(dict(item) for item in final_scores)
    if queried_ids:
        last_statistics = score_statistics(final_scores)
        if score is score_pivot_voi and all(row.get("hf_queried") for row in candidate_rows):
            last_statistics = {
                "selection_probability": 1.0,
                "selection_probability_lower": 1.0,
                "selection_probability_upper": 1.0,
                "incumbent_probability": float(
                    max(float(row["observed_delta"]) for row in candidate_rows) <= 0.0
                ),
                "max_acquisition": 0.0,
                "max_acquisition_upper": 0.0,
            }
        record(
            "rescore",
            acquisition_scores=list(final_scores),
            predictions={candidate_id(row): row["predicted_delta"] for row in candidate_rows},
            **last_statistics,
        )
    reason = reason or ("budget_exhausted" if len(ledger) >= budget else "candidate_exhausted")
    record("stop", stop_reason=reason, **last_statistics)
    estimates = {
        candidate_id(incumbent_row): 0.0,
        **{
            candidate_id(row): float(row.get("predicted_delta", row.get("delta_proxy", 0.0)))
            for row in candidate_rows
        },
    }
    selected_id = incumbent_id
    selected_estimate = 0.0
    for row in candidate_rows:
        identifier = candidate_id(row)
        estimate = estimates[identifier]
        if estimate > selected_estimate:
            selected_id = identifier
            selected_estimate = estimate
    selected_row = (
        incumbent_row
        if selected_id == incumbent_id
        else next(row for row in candidate_rows if candidate_id(row) == selected_id)
    )
    selected_true = selected_row.get("delta_true")
    for row in rows:
        row["selected"] = candidate_id(row) == selected_id
    true_values = [row.get("delta_true") for row in rows]
    regret = None
    if all(value is not None for value in true_values) and selected_true is not None:
        numeric_true_values = [float(value) for value in true_values if value is not None]
        regret = float(max(numeric_true_values) - float(selected_true))
    record(
        "select",
        selected_candidate_id=selected_id,
        selected_delta_estimate=selected_estimate,
        incumbent_utility=0.0,
        **last_statistics,
    )
    return RoundResult(
        selected_candidate_id=selected_id,
        selected_delta_true=None if selected_true is None else float(selected_true),
        selected_delta_estimate=float(selected_estimate),
        queried_ids=tuple(queried_ids),
        query_ledger=tuple(ledger),
        rows=tuple(rows),
        update_selection_regret=regret,
        cti_delta=None if selected_true is None else float(selected_true),
        hf_budget=len(ledger),
        hf_cost=sum(float(item["cost"]) for item in ledger),
        query_count=len(ledger),
        acquisition_method=acquisition_method,
        acquisition_scores=tuple(score_history),
        stop_reason=reason,
        posterior_version=_posterior_version(model_state, posterior_version),
        audit_trace=tuple(audit_trace),
        posterior=model_state,
    )


def _prepare_rows(
    candidates: Sequence[Mapping[str, Any] | PolicyTransition],
    proxy: Callable[[Any], Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        base = candidate.to_record() if isinstance(candidate, PolicyTransition) else dict(candidate)
        if proxy is not None:
            base.update(dict(proxy(candidate)))
        if base.get("hf_queried") or base.get("observed_delta") is not None:
            raise ValueError("prequeried observations require a separate evidence ledger")
        base.setdefault("hf_queried", False)
        rows.append(base)
    return rows


def _call_acquisition(
    acquisition: Callable[..., list[str]],
    rows: Sequence[Mapping[str, Any]],
    model: Any | None,
    budget: int,
    extra: Mapping[str, Any],
) -> list[str]:
    rows = _decision_rows(rows)
    if model is None:
        try:
            return [str(item) for item in acquisition(rows, budget, **extra)]
        except TypeError:
            return [str(item) for item in acquisition(rows, budget=budget, **extra)]
    try:
        return [str(item) for item in acquisition(rows, model, budget, **extra)]
    except TypeError:
        return [str(item) for item in acquisition(rows, model=model, budget=budget, **extra)]


def _call_score(
    score: Callable[..., Sequence[Mapping[str, Any]]],
    rows: Sequence[Mapping[str, Any]],
    model: Any | None,
    extra: Mapping[str, Any],
) -> Sequence[Mapping[str, Any]]:
    rows = _decision_rows(rows)
    if model is None:
        try:
            return score(rows, **extra)
        except TypeError:
            return score(rows, model=None, **extra)
    try:
        return score(rows, model, **extra)
    except TypeError:
        return score(rows, model=model, **extra)


def _call_stop(
    stop: Callable[..., bool | tuple[bool, str | None]],
    rows: Sequence[Mapping[str, Any]],
    model: Any | None,
    query_count: int,
    remaining_budget: int,
    extra: Mapping[str, Any],
) -> tuple[bool, str | None]:
    rows = _decision_rows(rows)
    try:
        outcome = stop(
            rows,
            model=model,
            query_count=query_count,
            remaining_budget=remaining_budget,
            **extra,
        )
    except TypeError:
        try:
            outcome = stop(rows, query_count, **extra)
        except TypeError:
            outcome = stop(rows)
    if isinstance(outcome, tuple):
        if len(outcome) != 2:
            raise ValueError("stop callback tuples must be (should_stop, reason)")
        return bool(outcome[0]), None if outcome[1] is None else str(outcome[1])
    return bool(outcome), None


def _refresh_predictions(rows: Sequence[dict[str, Any]], model: Any | None) -> None:
    for row in rows:
        if bool(row.get("hf_queried", False)):
            continue
        if model is not None and hasattr(model, "predict_correction"):
            prediction = model.predict_correction(_decision_rows([row])[0])
            row["predicted_delta"] = float(prediction.predicted_delta)
            row["predicted_correction"] = float(getattr(prediction, "correction", 0.0))
            row["prediction_standard_deviation"] = float(
                getattr(prediction, "standard_deviation", 0.0)
            )
        else:
            row["predicted_delta"] = float(row.get("delta_proxy", 0.0))


def _condition_model(model: Any | None, row: Mapping[str, Any], true_delta: float) -> Any | None:
    if model is None or not hasattr(model, "condition"):
        return model
    features = row["features"] if "features" in row else transition_feature_vector(row)
    correction = float(true_delta) - float(row.get("delta_proxy", 0.0))
    observation_variance = row.get("observation_variance")
    try:
        if observation_variance is None:
            conditioned = model.condition(features, correction)
        else:
            conditioned = model.condition(
                features, correction, observation_variance=float(observation_variance)
            )
    except TypeError:
        conditioned = model.condition(features, correction)
    return model if conditioned is None else conditioned


def _posterior_version(model: Any | None, base: str | None) -> str | None:
    if model is None:
        return base
    name = base or type(model).__name__
    observations = getattr(model, "n_observations", None)
    return name if observations is None else f"{name}:n={observations}"


def _decision_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Explicit observable schema; novel labels/containers are excluded by default.

    Features and policy parameters must be constructed from pre-query inputs by
    the adapter. Already-observed outcomes are exposed only as observed_delta.
    """

    allowed = {
        "transition_id",
        "round_id",
        "candidate_index",
        "candidate_id",
        "j",
        "incumbent_policy_id",
        "candidate_policy_id",
        "incumbent_parameters",
        "candidate_parameters",
        "improvement_operator",
        "edit_type",
        "operator_family",
        "proxy_world_id",
        "high_fidelity_world_id",
        "proxy_incumbent_value",
        "proxy_candidate_value",
        "delta_proxy",
        "features",
        "update_footprint",
        "footprint_components",
        "policy_distance",
        "action_distribution_distance",
        "response_strength",
        "competition_strength",
        "optimization_strength",
        "operator_shift",
        "chi_square_shift",
        "seed",
        "paired_seed_ids",
        "config_id",
        "hf_queried",
        "hf_query_cost",
        "hf_cost",
        "query_cost",
        "cost",
        "observation_variance",
        "predicted_delta",
        "predicted_correction",
        "prediction_standard_deviation",
        "is_incumbent",
        "observed_delta",
    }

    oracle_names = {
        "delta_true",
        "delta_actor",
        "delta_strategic",
        "delta_direct",
        "mechanical_effect",
        "competition_effect",
        "improvement_reversal",
        "strategic_improvement_reversal",
        "failure_type",
        "deployment_delta_true",
        "ground_truth",
        "reward_true",
        "hf_result",
        "final_outcome",
    }

    def masked(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: masked(item)
                for key, item in value.items()
                if key not in oracle_names
                and not str(key).startswith(("oracle_", "true_", "actor_", "strategic_"))
            }
        if isinstance(value, list):
            return [masked(item) for item in value]
        if isinstance(value, tuple):
            return tuple(masked(item) for item in value)
        return deepcopy(value)

    return [
        masked(
            {
                key: value
                for key, value in row.items()
                if key in allowed and (key != "observed_delta" or row.get("hf_queried"))
            }
        )
        for row in rows
    ]


run_sequential_pivot_round = run_pivot_round_sequential
run_pivot_sequential_round = run_pivot_round_sequential


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
    ordered_candidates = sorted(candidates, key=candidate_id)
    scores = score_pivot_voi(
        _decision_rows(_prepare_rows(ordered_candidates, None)),
        posterior,
        seed=seed,
        fantasies=fantasies,
        posterior_samples=posterior_samples,
    )
    selection_probability = float(scores[0]["selection_probability_lower"]) if scores else 1.0
    max_acquisition = max((float(item["acquisition_upper"]) for item in scores), default=0.0)
    stop, reason = should_stop(
        selection_probability=selection_probability,
        max_acquisition=max_acquisition,
        delta=delta,
        eta=eta,
    )
    budget = 0 if stop else max_budget

    def voi_stop(
        rows: Sequence[Mapping[str, Any]],
        model: Any | None = None,
        **_: Any,
    ) -> tuple[bool, str | None]:
        available = [
            row
            for row in rows
            if not bool(row.get("is_incumbent", False)) and not bool(row.get("hf_queried", False))
        ]
        if not available:
            return True, "candidate_exhausted"
        current_scores = score_pivot_voi(
            available,
            model if model is not None else posterior,
            seed=seed,
            fantasies=fantasies,
            posterior_samples=posterior_samples,
            decision_candidates=rows,
        )
        current_selection_probability = (
            float(current_scores[0]["selection_probability_lower"]) if current_scores else 1.0
        )
        current_max_acquisition = max(
            (float(item["acquisition_upper"]) for item in current_scores),
            default=0.0,
        )
        return should_stop(
            selection_probability=current_selection_probability,
            max_acquisition=current_max_acquisition,
            delta=delta,
            eta=eta,
        )

    return run_pivot_round_sequential(
        incumbent,
        ordered_candidates,
        proxy=None,
        hf=hf,
        acquisition=select_pivot_voi,
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
        stop=voi_stop,
        score=score_pivot_voi,
        stop_reason=reason if stop else None,
        posterior_version="bayesian-linear-v1",
    )
