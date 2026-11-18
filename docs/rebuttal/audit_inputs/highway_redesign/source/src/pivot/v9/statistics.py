from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def sign(value: float | None, tau: float = 1e-9) -> int:
    if value is None or abs(float(value)) <= tau:
        return 0
    return 1 if float(value) > 0 else -1


def bootstrap_mean_ci(values: Sequence[float], *, seed: int, draws: int = 10000, alpha: float = 0.05) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be finite and non-empty")
    if draws < 100:
        raise ValueError("draws must be at least 100")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(draws, array.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, alpha / 2.0)), float(np.quantile(samples, 1.0 - alpha / 2.0))


def hierarchical_mean_ci(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    group_key: str,
    *,
    seed: int,
    draws: int = 10000,
) -> dict[str, float | int]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        group = row.get(group_key)
        if value is not None and group is not None and math.isfinite(float(value)):
            groups[str(group)].append(float(value))
    if not groups:
        raise ValueError(f"no finite {value_key} values")
    group_means = np.asarray([sum(values) / len(values) for values in groups.values()], dtype=float)
    low, high = bootstrap_mean_ci(group_means.tolist(), seed=seed, draws=draws)
    return {
        "mean": float(group_means.mean()),
        "median": float(np.median(group_means)),
        "std": float(group_means.std(ddof=1)) if len(group_means) > 1 else 0.0,
        "ci_low": low,
        "ci_high": high,
        "n_groups": len(group_means),
        "n_rows": sum(len(values) for values in groups.values()),
    }


def improvement_metrics(rows: Sequence[Mapping[str, Any]], *, tau: float = 1e-9) -> dict[str, float | int | None]:
    pairs = [
        (float(row["delta_proxy"]), float(row["delta_true"]))
        for row in rows
        if row.get("delta_proxy") is not None and row.get("delta_true") is not None
    ]
    if not pairs:
        return {"IDE": None, "ISC": None, "IRR": None, "ISR": None, "n": 0}
    ide = float(np.mean([abs(proxy - true) for proxy, true in pairs]))
    comparable = [(proxy, true) for proxy, true in pairs if sign(proxy, tau) and sign(true, tau)]
    isc = float(np.mean([sign(proxy, tau) == sign(true, tau) for proxy, true in comparable])) if comparable else None
    positive = [(proxy, true) for proxy, true in pairs if proxy > tau]
    irr = float(np.mean([true < -tau for _, true in positive])) if positive else None
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("trajectory_id", row.get("seed", "0")))].append(row)
    regrets: list[float] = []
    for group in groups.values():
        available = [float(row["delta_true"]) for row in group if row.get("delta_true") is not None]
        selected = [float(row["delta_true"]) for row in group if row.get("selected") and row.get("delta_true") is not None]
        if available and selected:
            regrets.append(max(available) - selected[0])
    return {"IDE": ide, "ISC": isc, "IRR": irr, "ISR": float(np.mean(regrets)) if regrets else None, "n": len(pairs)}


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("rank inputs must have equal non-empty length")
    def ranks(values: Sequence[float]) -> NDArray[np.float64]:
        order = np.argsort(np.asarray(values), kind="stable")
        output = np.empty(len(values), dtype=float)
        output[order] = np.arange(1, len(values) + 1, dtype=float)
        return output
    a, b = ranks(left), ranks(right)
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator == 0.0 else float(np.dot(a, b) / denominator)


def density_diagnostics(levels: Iterable[float]) -> list[dict[str, float]]:
    values = [float(level) for level in levels]
    return [
        {
            "operator_shift": level,
            "chi_square_shift": float(np.expm1(level * level)),
            "effective_sample_size_ratio": float(1.0 / (1.0 + np.expm1(level * level))),
            "mmd_proxy": float(level * level / (1.0 + level * level)),
        }
        for level in values
    ]
