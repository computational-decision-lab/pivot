from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pivot.metrics.improvement import compute_improvement_metrics

from .differential import DifferentialModel, GradientBoostedDifferentialModel
from .global_value import GlobalValueModel, spearman_rank_correlation


def compare_global_vs_local(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    budget: int,
) -> dict[str, Any]:
    """Compare matched-budget policy-value and transition-differential models."""

    if budget <= 0 or budget > len(train_rows):
        raise ValueError("budget must be within the training rows")
    train_ids = {str(row["transition_id"]) for row in train_rows[:budget]}
    test_ids = {str(row["transition_id"]) for row in test_rows}
    if train_ids & test_ids:
        raise ValueError("train and test transition IDs must be disjoint")
    selected_train = list(train_rows[:budget])
    global_model = GlobalValueModel()
    global_features = [row.get("candidate_parameters", {}) for row in selected_train] + [
        row.get("incumbent_parameters", {}) for row in selected_train
    ]
    global_values = [float(row["true_candidate_value"]) for row in selected_train] + [
        float(row["true_incumbent_value"]) for row in selected_train
    ]
    global_model.fit(
        global_features,
        global_values,
        policy_ids=[str(row["candidate_policy_id"]) for row in selected_train],
    )
    # HF budget is counted in paired transitions, not individual policy-value
    # labels; each queried transition yields incumbent and candidate values.
    global_model.hf_budget = budget
    differential_model = DifferentialModel()
    correction_targets = [float(row["delta_true"]) - float(row["delta_proxy"]) for row in selected_train]
    differential_model.fit(
        selected_train,
        correction_targets,
        transition_ids=[str(row["transition_id"]) for row in selected_train],
    )
    boosted_model = GradientBoostedDifferentialModel()
    boosted_model.fit(selected_train, correction_targets)

    global_predicted_values = [global_model.predict_row(row, role="candidate") for row in test_rows]
    global_true_values = [float(row["true_candidate_value"]) for row in test_rows]
    global_rank = spearman_rank_correlation(global_predicted_values, global_true_values)
    global_metric_rows = []
    local_rows: list[dict[str, Any]] = []
    boosted_metric_rows = []
    for row in test_rows:
        selection_group = _selection_group(row)
        prediction = differential_model.predict_correction(row)
        boosted_prediction = boosted_model.predict_correction(row)
        global_delta = global_model.predict_row(row, role="candidate") - global_model.predict_row(
            row, role="incumbent"
        )
        global_metric_rows.append(
            {
                "delta_proxy": global_delta,
                "delta_true": row["delta_true"],
                "round_id": selection_group,
            }
        )
        boosted_metric_rows.append(
            {
                "delta_proxy": boosted_prediction.predicted_delta,
                "delta_true": row["delta_true"],
                "round_id": selection_group,
            }
        )
        local_rows.append(
            {
                "transition_id": row["transition_id"],
                "round_id": selection_group,
                "delta_proxy": row["delta_proxy"],
                "delta_true": row["delta_true"],
                "predicted_delta": prediction.predicted_delta,
                "global_predicted_delta": global_delta,
                "boosted_predicted_delta": boosted_prediction.predicted_delta,
                "correction_uncertainty": prediction.standard_deviation,
            }
        )
    local_metric_rows = [
        {"delta_proxy": row["predicted_delta"], "delta_true": row["delta_true"], "round_id": row["round_id"]}
        for row in local_rows
    ]
    _mark_selected(global_metric_rows)
    _mark_selected(local_metric_rows)
    _mark_selected(boosted_metric_rows)
    for index, row in enumerate(local_rows):
        row["global_selected"] = bool(global_metric_rows[index]["selected"])
        row["local_selected"] = bool(local_metric_rows[index]["selected"])
        row["boosted_selected"] = bool(boosted_metric_rows[index]["selected"])
    local_metrics = compute_improvement_metrics(local_metric_rows)
    global_update_metrics = compute_improvement_metrics(global_metric_rows)
    boosted_metrics = compute_improvement_metrics(boosted_metric_rows)
    global_mae = float(np.mean(np.abs(np.asarray(global_predicted_values) - np.asarray(global_true_values))))
    return {
        "policy_value_mae": global_mae,
        "policy_rank_correlation": global_rank,
        "global_improvement_differential_error": global_update_metrics["ide"],
        "global_improvement_sign_consistency": global_update_metrics["isc"],
        "global_improvement_reversal_rate": global_update_metrics["irr"],
        "global_update_selection_regret": global_update_metrics["isr"],
        "improvement_differential_error": local_metrics["ide"],
        "improvement_sign_consistency": local_metrics["isc"],
        "improvement_reversal_rate": local_metrics["irr"],
        "update_selection_regret": local_metrics["isr"],
        "boosted_improvement_differential_error": boosted_metrics["ide"],
        "boosted_improvement_sign_consistency": boosted_metrics["isc"],
        "boosted_improvement_reversal_rate": boosted_metrics["irr"],
        "boosted_update_selection_regret": boosted_metrics["isr"],
        "global_hf_budget": global_model.hf_budget,
        "local_hf_budget": differential_model.hf_budget,
        "train_transition_ids": sorted(train_ids),
        "test_transition_ids": sorted(test_ids),
        "local_rows": local_rows,
    }


def _selection_group(row: Mapping[str, Any]) -> str:
    fields = (
        "round_id",
        "seed",
        "response_strength",
        "optimization_strength",
        "config_id",
        "incumbent_policy_id",
    )
    return "|".join(f"{field}={row.get(field)}" for field in fields)


def _mark_selected(rows: Sequence[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["round_id"]), []).append(row)
    for candidates in groups.values():
        selected = max(candidates, key=lambda row: float(row["delta_proxy"]))
        for row in candidates:
            row["selected"] = row is selected
