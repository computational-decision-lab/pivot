from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def compute_improvement_fidelity(
    rows: Sequence[Mapping[str, Any]],
    *,
    loss: str = "absolute_delta_error",
    tau_sign: float = 1e-9,
) -> float | None:
    """Estimate ``IF(V, A)`` from transitions sampled from ``Q_A``.

    Each row is one draw from the operator-induced transition distribution.
    Repeated rows therefore represent their empirical probability mass.  The
    absolute-delta loss is the usual IDE estimand; ``sign_error`` is
    ``1 - ISC`` and ignores sign ties using ``tau_sign``.
    """

    if loss not in {"absolute_delta_error", "sign_error"}:
        raise ValueError("loss must be 'absolute_delta_error' or 'sign_error'")
    metrics = compute_improvement_metrics(rows, tau_sign=tau_sign)
    if loss == "absolute_delta_error":
        value = metrics["ide"]
        return None if value is None else float(value)
    isc = metrics["isc"]
    return None if isc is None else 1.0 - float(isc)


def _sign(value: float | None, tolerance: float) -> int:
    if value is None or abs(float(value)) <= tolerance:
        return 0
    return 1 if float(value) > 0 else -1


def compute_improvement_metrics(
    rows: Sequence[Mapping[str, Any]], tau_sign: float = 1e-9, tau_mtr: float = 1e-8
) -> dict[str, float | int | None]:
    if not rows:
        raise ValueError("rows must not be empty")
    proxy_true = [
        (float(row["delta_proxy"]), float(row.get("delta_true", row.get("delta_actor"))))
        for row in rows
        if row.get("delta_proxy") is not None
        and row.get("delta_true", row.get("delta_actor")) is not None
    ]
    if not proxy_true:
        raise ValueError("rows must contain proxy and true deltas")
    errors = [abs(proxy - true) for proxy, true in proxy_true]
    comparable = [
        (proxy, true)
        for proxy, true in proxy_true
        if _sign(proxy, tau_sign) != 0 and _sign(true, tau_sign) != 0
    ]
    agreements = sum(_sign(proxy, tau_sign) == _sign(true, tau_sign) for proxy, true in comparable)
    positive_proxy = [(proxy, true) for proxy, true in proxy_true if proxy > tau_sign]
    reversals = sum(true < -tau_sign for _, true in positive_proxy)
    mtr_values = [true / proxy for proxy, true in proxy_true if abs(proxy) > tau_mtr]
    actor_strategic = [
        (float(row["delta_actor"]), float(row["delta_strategic"]))
        for row in rows
        if row.get("delta_actor") is not None and row.get("delta_strategic") is not None
    ]
    actor_positive = [(actor, strategic) for actor, strategic in actor_strategic if actor > tau_sign]
    strategic_reversals = sum(strategic < -tau_sign for _, strategic in actor_positive)
    ties = sum(_sign(proxy, tau_sign) == 0 or _sign(true, tau_sign) == 0 for proxy, true in proxy_true)
    selection_regrets: list[float] = []
    grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("round_id", 0)].append(row)
    for candidates in grouped.values():
        available = [row for row in candidates if row.get("delta_true", row.get("delta_actor")) is not None]
        selected = [row for row in available if row.get("selected")]
        if selected:
            values = [float(row.get("delta_true", row.get("delta_actor"))) for row in available]
            chosen = float(selected[0].get("delta_true", selected[0].get("delta_actor")))
            selection_regrets.append(max(values) - chosen)
    selected_rows = [
        row for row in rows
        if row.get("selected") and row.get("delta_true", row.get("delta_actor")) is not None
    ]
    selected_values = [float(row.get("delta_true", row.get("delta_actor"))) for row in selected_rows]
    return {
        "ide": sum(errors) / len(errors),
        "isc": agreements / len(comparable) if comparable else None,
        "irr": reversals / len(positive_proxy) if positive_proxy else None,
        "sirr": strategic_reversals / len(actor_positive) if actor_positive else None,
        "mtr": sum(mtr_values) / len(mtr_values) if mtr_values else None,
        "isr": sum(selection_regrets) / len(selection_regrets) if selection_regrets else None,
        "cti": sum(selected_values) if selected_rows else None,
        "unselected_true_sum": sum(
            float(row.get("delta_true", row.get("delta_actor")))
            for row in rows
            if row.get("delta_true", row.get("delta_actor")) is not None
        ),
        "n_selected": len(selected_rows),
        "n_rows": len(proxy_true),
        "n_comparable_signs": len(comparable),
        "n_positive_proxy": len(positive_proxy),
        "n_reversals": reversals,
        "n_strategic_reversals": strategic_reversals,
        "n_ties": ties,
        "n_hf_transitions": sum(bool(row.get("hf_queried")) for row in rows),
    }
