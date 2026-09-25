from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

V9_TERMINAL_STATES = (
    "IMPLEMENTATION_FAILURE",
    "DESIGN_INVALID",
    "UNDERPOWERED",
    "HYPOTHESIS_SUPPORTED",
    "HYPOTHESIS_NOT_SUPPORTED",
)

V9_TRANSITION_COLUMNS = (
    "experiment_id",
    "environment_id",
    "environment_family",
    "trajectory_id",
    "round_id",
    "seed",
    "operator_id",
    "operator_family",
    "candidate_id",
    "incumbent_policy_id",
    "candidate_policy_id",
    "delta_proxy",
    "delta_direct",
    "delta_actor",
    "delta_strategic",
    "delta_true",
    "policy_distance",
    "action_distribution_distance",
    "response_strength",
    "operator_shift",
    "chi_square_shift",
    "method",
    "hf_queried",
    "hf_query_order",
    "hf_cost",
    "posterior_mean",
    "posterior_std",
    "p_positive",
    "p_best",
    "expected_regret",
    "evsi",
    "evsi_per_cost",
    "selected",
    "true_best",
    "IDE",
    "ISC",
    "IRR",
    "ISR",
    "CISR",
    "CTI",
    "config_hash",
    "source_commit",
)


def stable_id(*parts: object, length: int = 20) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def make_transition_row(
    *,
    experiment_id: str,
    environment_id: str,
    environment_family: str,
    trajectory_id: str,
    round_id: int,
    seed: int,
    operator_id: str,
    operator_family: str,
    candidate_id: int,
    incumbent_policy_id: str,
    candidate_policy_id: str,
    delta_proxy: float | None,
    delta_direct: float | None = None,
    delta_actor: float | None = None,
    delta_strategic: float | None = None,
    delta_true: float | None = None,
    policy_distance: float | None = None,
    action_distribution_distance: float | None = None,
    response_strength: float | None = None,
    operator_shift: float | None = None,
    chi_square_shift: float | None = None,
    method: str | None = None,
    hf_queried: bool = False,
    hf_query_order: int | None = None,
    hf_cost: float = 0.0,
    posterior_mean: float | None = None,
    posterior_std: float | None = None,
    p_positive: float | None = None,
    p_best: float | None = None,
    expected_regret: float | None = None,
    evsi: float | None = None,
    evsi_per_cost: float | None = None,
    selected: bool = False,
    true_best: bool = False,
    config_hash: str | None = None,
    source_commit: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Create one null-preserving transition record.

    Causal layers are never inferred: a missing actor or strategic quantity is
    serialized as JSON null. Extra diagnostics are allowed, but the canonical
    fields remain stable for figure and table consumers.
    """

    row: dict[str, Any] = {
        "experiment_id": experiment_id,
        "environment_id": environment_id,
        "environment_family": environment_family,
        "trajectory_id": trajectory_id,
        "round_id": int(round_id),
        "seed": int(seed),
        "operator_id": operator_id,
        "operator_family": operator_family,
        "candidate_id": int(candidate_id),
        "incumbent_policy_id": incumbent_policy_id,
        "candidate_policy_id": candidate_policy_id,
        "delta_proxy": delta_proxy,
        "delta_direct": delta_direct,
        "delta_actor": delta_actor,
        "delta_strategic": delta_strategic,
        "delta_true": delta_true,
        "policy_distance": policy_distance,
        "action_distribution_distance": action_distribution_distance,
        "response_strength": response_strength,
        "operator_shift": operator_shift,
        "chi_square_shift": chi_square_shift,
        "method": method,
        "hf_queried": bool(hf_queried),
        "hf_query_order": hf_query_order,
        "hf_cost": float(hf_cost),
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "p_positive": p_positive,
        "p_best": p_best,
        "expected_regret": expected_regret,
        "evsi": evsi,
        "evsi_per_cost": evsi_per_cost,
        "selected": bool(selected),
        "true_best": bool(true_best),
        "IDE": None,
        "ISC": None,
        "IRR": None,
        "ISR": None,
        "CISR": None,
        "CTI": None,
        "config_hash": config_hash,
        "source_commit": source_commit,
    }
    row.update(extra)
    return row


def numeric_features(row: Mapping[str, Any]) -> list[float]:
    """Return only pre-query features; outcome columns are intentionally absent."""

    values = (
        row.get("delta_proxy", 0.0),
        row.get("policy_distance", 0.0),
        row.get("action_distribution_distance", 0.0),
        row.get("response_strength", 0.0),
        row.get("operator_shift", 0.0),
        row.get("chi_square_shift", 0.0),
        row.get("candidate_id", 0.0),
    )
    return [0.0 if value is None else float(value) for value in values]
