from __future__ import annotations

from pivot.core.policy import Policy


def compute_finance_footprint(incumbent: Policy, candidate: Policy) -> dict[str, float]:
    keys = (
        "position_size",
        "participation",
        "urgency",
        "holding_horizon",
        "rebalance_frequency",
        "threshold",
        "signal",
    )
    values = {key: abs(candidate.parameters.get(key, 0.0) - incumbent.parameters.get(key, 0.0)) for key in keys}
    values["turnover"] = values["rebalance_frequency"] * values["position_size"]
    values["size"] = values["position_size"]
    values["liquidity_consumption"] = values["participation"] * values["urgency"]
    values["aggressive_passive_ratio"] = values["urgency"] / max(values["holding_horizon"], 1e-12)
    values["inventory_duration"] = values["holding_horizon"]
    values["spread_crossing"] = values["urgency"]
    return values
