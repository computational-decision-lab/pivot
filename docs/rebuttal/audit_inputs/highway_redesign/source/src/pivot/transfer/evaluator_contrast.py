"""Controlled evaluator contrast for value versus transition fidelity."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from pivot.evaluation.uncertainty import bootstrap_mean_ci
from pivot.metrics.improvement import compute_improvement_metrics

from .global_value import spearman_rank_correlation


@dataclass(frozen=True)
class EvaluatorContrastConfig:
    """Parameters for the transparent controlled evaluator construction."""

    differential_bias_scale: float = 4.0
    common_value_offset: float = 0.6
    tau_sign: float = 1e-9
    hf_budget: int = 24
    bootstrap_seed: int = 20260820


@dataclass(frozen=True)
class EvaluatorContrastMetrics:
    """Policy-value and transition-level metrics for one evaluator."""

    policy_value_mae: float
    policy_rank_correlation: float
    improvement_differential_error: float
    improvement_sign_consistency: float | None
    improvement_reversal_rate: float | None
    update_selection_regret: float | None
    cumulative_true_improvement: float | None
    n_test_transitions: int
    n_positive_proxy: int
    n_reversals: int
    policy_value_mae_ci_low: float
    policy_value_mae_ci_high: float
    improvement_differential_error_ci_low: float
    improvement_differential_error_ci_high: float
    improvement_sign_consistency_ci_low: float
    improvement_sign_consistency_ci_high: float


def run_evaluator_contrast(
    rows: Sequence[Mapping[str, Any]],
    train_transition_ids: set[str],
    config: EvaluatorContrastConfig | None = None,
) -> dict[str, Any]:
    """Compare two transparent evaluators on a held-out transition split.

    Evaluator A has a small policy-value error but an asymmetric candidate-side
    bias that grows with visible response and footprint. Evaluator B has a
    larger policy-independent value offset, which is deliberately poor for
    isolated value accuracy but cancels exactly in paired deltas. This is a
    constructive diagnostic of the estimand, not a learned-model benchmark.
    """

    cfg = config or EvaluatorContrastConfig()
    if cfg.differential_bias_scale < 0 or cfg.common_value_offset < 0:
        raise ValueError("contrast scales must be non-negative")
    if cfg.hf_budget <= 0:
        raise ValueError("hf_budget must be positive")
    test_rows = [row for row in rows if str(row["transition_id"]) not in train_transition_ids]
    if not test_rows:
        raise ValueError("held-out transition set must not be empty")
    source_ids = {str(row["transition_id"]) for row in rows}
    if not train_transition_ids <= source_ids:
        raise ValueError("train transition IDs must come from the source rows")

    evaluator_rows: dict[str, list[dict[str, Any]]] = {"value_fidelity": [], "transition_fidelity": []}
    for row in test_rows:
        response = float(row.get("response_strength", 0.0) or 0.0)
        footprint = float(row.get("update_footprint", 0.0) or 0.0)
        candidate_bias = cfg.differential_bias_scale * response * footprint
        true_incumbent = float(row["true_incumbent_value"])
        true_candidate = float(row["true_candidate_value"])
        base = _base_record(row)
        evaluator_rows["value_fidelity"].append(
            {
                **base,
                "predicted_incumbent_value": true_incumbent,
                "predicted_candidate_value": true_candidate + candidate_bias,
                "delta_proxy": float(row["delta_true"]) + candidate_bias,
                "evaluator_bias": candidate_bias,
            }
        )
        evaluator_rows["transition_fidelity"].append(
            {
                **base,
                "predicted_incumbent_value": true_incumbent + cfg.common_value_offset,
                "predicted_candidate_value": true_candidate + cfg.common_value_offset,
                "delta_proxy": float(row["delta_true"]),
                "evaluator_bias": cfg.common_value_offset,
            }
        )

    metrics: dict[str, dict[str, Any]] = {}
    for name, contrast_rows in evaluator_rows.items():
        _mark_selected(contrast_rows)
        metrics[name] = asdict(_metrics(contrast_rows, cfg))

    return {
        "experiment": "e4-value-vs-improvement-fidelity",
        "diagnostic_type": "controlled_evaluator_contrast",
        "interpretation": (
            "Evaluator A has lower isolated-value error but an asymmetric candidate bias; "
            "Evaluator B has a policy-independent value offset that cancels in paired deltas."
        ),
        "config": asdict(cfg),
        "source_row_count": len(rows),
        "train_row_count": len(train_transition_ids),
        "test_row_count": len(test_rows),
        "train_transition_ids": sorted(train_transition_ids),
        "test_transition_ids": sorted(str(row["transition_id"]) for row in test_rows),
        "metrics": metrics,
        "rows": evaluator_rows,
    }


def _base_record(row: Mapping[str, Any]) -> dict[str, Any]:
    selection_group = (
        f"{row.get('config_id')}|seed={row.get('seed')}|"
        f"response={row.get('response_strength')}|optimization={row.get('optimization_strength')}"
    )
    return {
        "transition_id": str(row["transition_id"]),
        "selection_group": selection_group,
        "source_transition_id": str(row["transition_id"]),
        "true_incumbent_value": float(row["true_incumbent_value"]),
        "true_candidate_value": float(row["true_candidate_value"]),
        "delta_true": float(row["delta_true"]),
        "response_strength": float(row.get("response_strength", 0.0) or 0.0),
        "update_footprint": float(row.get("update_footprint", 0.0) or 0.0),
        "candidate_index": int(row.get("candidate_index", 0)),
        "seed": int(row.get("seed", 0)),
    }


def _metrics(rows: Sequence[Mapping[str, Any]], config: EvaluatorContrastConfig) -> EvaluatorContrastMetrics:
    if not rows:
        raise ValueError("evaluator rows must not be empty")
    policy_errors = [
        abs(float(row["predicted_incumbent_value"]) - float(row["true_incumbent_value"]))
        for row in rows
    ] + [
        abs(float(row["predicted_candidate_value"]) - float(row["true_candidate_value"]))
        for row in rows
    ]
    predicted_candidates = [float(row["predicted_candidate_value"]) for row in rows]
    true_candidates = [float(row["true_candidate_value"]) for row in rows]
    metric_rows = [
        {
            "round_id": row["selection_group"],
            "delta_proxy": float(row["delta_proxy"]),
            "delta_true": float(row["delta_true"]),
            "selected": bool(row.get("selected", False)),
        }
        for row in rows
    ]
    improvement = compute_improvement_metrics(metric_rows, tau_sign=config.tau_sign)
    sign_matches = [
        int(_sign(float(row["delta_proxy"]), config.tau_sign) == _sign(float(row["delta_true"]), config.tau_sign))
        for row in rows
        if _sign(float(row["delta_proxy"]), config.tau_sign) != 0
        and _sign(float(row["delta_true"]), config.tau_sign) != 0
    ]
    policy_low, policy_high = bootstrap_mean_ci(policy_errors, seed=config.bootstrap_seed)
    differential_errors = [
        abs(float(row["delta_proxy"]) - float(row["delta_true"])) for row in rows
    ]
    differential_low, differential_high = bootstrap_mean_ci(
        differential_errors, seed=config.bootstrap_seed + 1
    )
    sign_low, sign_high = bootstrap_mean_ci(sign_matches or [0.0], seed=config.bootstrap_seed + 2)
    return EvaluatorContrastMetrics(
        policy_value_mae=float(np.mean(policy_errors)),
        policy_rank_correlation=spearman_rank_correlation(predicted_candidates, true_candidates),
        improvement_differential_error=_required_float(improvement["ide"]),
        improvement_sign_consistency=_as_optional_float(improvement["isc"]),
        improvement_reversal_rate=_as_optional_float(improvement["irr"]),
        update_selection_regret=_as_optional_float(improvement["isr"]),
        cumulative_true_improvement=_as_optional_float(improvement["cti"]),
        n_test_transitions=len(rows),
        n_positive_proxy=_required_int(improvement["n_positive_proxy"]),
        n_reversals=_required_int(improvement["n_reversals"]),
        policy_value_mae_ci_low=float(policy_low),
        policy_value_mae_ci_high=float(policy_high),
        improvement_differential_error_ci_low=float(differential_low),
        improvement_differential_error_ci_high=float(differential_high),
        improvement_sign_consistency_ci_low=float(sign_low),
        improvement_sign_consistency_ci_high=float(sign_high),
    )


def _mark_selected(rows: Sequence[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["selection_group"])].append(row)
    for candidates in grouped.values():
        selected = max(candidates, key=lambda row: float(row["delta_proxy"]))
        selected["selected"] = True


def _sign(value: float, tolerance: float) -> int:
    if abs(value) <= tolerance:
        return 0
    return 1 if value > 0 else -1


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise TypeError(f"expected a numeric metric, got {type(value).__name__}")
    return float(value)


def _required_float(value: object) -> float:
    if value is None:
        raise ValueError("required metric is missing")
    if not isinstance(value, (int, float)):
        raise TypeError(f"expected a numeric metric, got {type(value).__name__}")
    return float(value)


def _required_int(value: object) -> int:
    if value is None:
        raise ValueError("required metric is missing")
    if not isinstance(value, int):
        raise TypeError(f"expected an integer metric, got {type(value).__name__}")
    return int(value)
