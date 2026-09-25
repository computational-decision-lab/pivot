"""Selection-only Melting Pot adapter for the *actual* author E5C entry points.

Put colin_pivot and colin_pivot/src on PYTHONPATH. This module imports neither
PPO nor Melting Pot, starts no training, and never opens an audit directory.
The external feature/calibration bridge is explicit; the query allocation and
promotion rules come directly from experiments.v9.e5c_efficiency.

E5C is fixed-budget, batch allocation: no early stopping or real-observation
posterior conditioning. Its Random-HF keeps proxy predictions for unqueried
candidates, whereas PIVOT/Global-VOI/Paired LUCB keep posterior predictions.
Names such as Global-VOI and Paired LUCB mean the author's E5C implementations,
not a claim that these are textbook implementations of the named algorithms.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.v9 import e5c_efficiency as author_e5
from pivot.acquisition import pivot_voi as author_voi
from pivot.acquisition.pivot_voi import BayesianLinearDeltaPosterior

BUDGETS = (0, 1, 2, 4, 8)
METHODS = ("proxy_only", "random_hf", "global_voi", "paired_lucb", "pivot_voi")
SUPPORTED_METHODS = METHODS + ("top_proxy_hf", "uncertainty_hf", "pivot_h", "all_hf")
CORRECTED_METHODS = frozenset(
    ("uncertainty_hf", "paired_lucb", "global_voi", "pivot_h", "pivot_voi")
)
DEFAULT_STATISTICS = {"voi_fantasies": 8, "voi_posterior_samples": 32}


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def source_provenance() -> dict[str, Any]:
    """Hash the loaded source, so a run can pin the actual author entry points."""
    sources = {}
    for module in (author_e5, author_voi):
        path = Path(module.__file__).resolve()
        sources[module.__name__] = {
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        }
    return {
        "implementation": "author_E5C_fixed_budget",
        "allocation": "experiments.v9.e5c_efficiency._select",
        "promotion": "experiments.v9.e5c_efficiency._select_outcome",
        "early_stopping": False,
        "real_query_posterior_update": False,
        "selection_reads_audit": False,
        "sources": sources,
    }


def _raw_features(features: Any) -> np.ndarray:
    # features.json is a matrix in crossbench and {"rows": matrix} in older MPE runs.
    if isinstance(features, Mapping):
        features = features["rows"]
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError("features must be a nonempty [proxy, footprints...] matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("features must be finite")
    return matrix


def fit_external_posterior(
    calibration_rows: Sequence[Mapping[str, Any]],
) -> tuple[BayesianLinearDeltaPosterior, dict[str, Any]]:
    """Bridge crossbench calibration summaries to the unchanged E5C fit.

    Input rows have features=[proxy, footprints...] and target_correction.
    Standardization uses calibration footprints only. As in author E5C,
    noise_variance=max(var(calibration corrections), 1e-4); unlike the previous
    replay this does NOT silently substitute replicate-noise scaling.
    """
    raw = _raw_features([row["features"] for row in calibration_rows])
    mean = raw[:, 1:].mean(axis=0)
    scale = raw[:, 1:].std(axis=0)
    scale[scale < 1e-8] = 1.0
    features = np.c_[np.ones(len(raw)), (raw[:, 1:] - mean) / scale]
    converted = [
        {
            "features": features[i].tolist(),
            "delta_proxy": float(raw[i, 0]),
            "delta_true": float(raw[i, 0]) + _finite(row["target_correction"], "correction"),
        }
        for i, row in enumerate(calibration_rows)
    ]
    posterior = author_e5._fit_posterior(converted)
    payload = {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "mean": posterior.mean.tolist(),
        "covariance": posterior.covariance.tolist(),
        "prior_precision": posterior.prior_precision,
        "noise_variance": posterior.noise_variance,
        "n_observations": posterior.n_observations,
        "feature_rule": "intercept + calibration-standardized footprints; proxy added separately",
        "posterior_fit": "author_E5C._fit_posterior",
        "noise_rule": "max(population variance of calibration corrections, 1e-4)",
        "test_labels_used": False,
    }
    return posterior, payload


def posterior_from_payload(payload: Mapping[str, Any]) -> BayesianLinearDeltaPosterior:
    """Restore this adapter's payload or an old replay_author_melting posterior.

    Existing mean/covariance/query_mean_noise values are preserved exactly;
    restoring an old fit does not relabel that fit as author E5C calibration.
    """
    mean_key = "coefficient_mean" if "coefficient_mean" in payload else "mean"
    noise_key = "query_mean_noise" if "query_mean_noise" in payload else "noise_variance"
    mean = np.asarray(payload[mean_key], dtype=float)
    covariance = np.asarray(payload["covariance"], dtype=float)
    if mean.ndim != 1 or not len(mean) or covariance.shape != (len(mean), len(mean)):
        raise ValueError("posterior dimensions are inconsistent")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("posterior parameters must be finite")
    if not np.allclose(covariance, covariance.T, rtol=1e-8, atol=1e-10):
        raise ValueError("posterior covariance must be symmetric")
    if np.linalg.eigvalsh(covariance).min() < -1e-9:
        raise ValueError("posterior covariance must be positive semidefinite")
    return BayesianLinearDeltaPosterior(
        prior_precision=float(payload.get("prior_precision", 1.0)),
        noise_variance=float(payload[noise_key]),
        mean=mean.copy(), covariance=covariance.copy(),
        n_observations=int(payload.get("n_observations", 0)),
    )


def make_candidates(
    raw_features: Any,
    transform: Mapping[str, Any],
    query_cost: float | Sequence[float],
    *,
    candidate_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert existing features.json data to the E5C candidate representation.

    K is inferred (K=8 for the new benchmark; K=4 historical data also work).
    Costs must be declared before observing selection responses.
    """
    raw = _raw_features(raw_features)
    mean = np.asarray(transform["feature_mean"], dtype=float)
    scale = np.asarray(transform["feature_scale"], dtype=float)
    if mean.shape != (raw.shape[1] - 1,) or scale.shape != mean.shape:
        raise ValueError("feature transform has the wrong dimensions")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("feature transform must be finite with positive scales")
    z = np.c_[np.ones(len(raw)), (raw[:, 1:] - mean) / scale]
    costs = np.broadcast_to(np.asarray(query_cost, dtype=float), (len(raw),))
    ids = [str(i) for i in range(len(raw))] if candidate_ids is None else list(map(str, candidate_ids))
    if len(ids) != len(raw) or len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique and match the number of rows")
    if not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError("query costs must be finite and positive")
    return [
        {"transition_id": ids[i], "delta_proxy": float(raw[i, 0]),
         "features": z[i].tolist(), "hf_query_cost": float(costs[i])}
        for i in range(len(raw))
    ]


def run_e5_decision(
    candidates: Sequence[Mapping[str, Any]],
    posterior: BayesianLinearDeltaPosterior,
    query: Callable[[Mapping[str, Any]], float | Mapping[str, Any]],
    *,
    method: str,
    budget: int,
    seed: int,
    statistics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Choose using only proxies, the frozen posterior, and paid query responses.

    query receives a clean candidate mapping and returns a selection delta or
    {delta, total_env_steps, ...}. Only allocated candidates are queried once.
    The author _select_outcome's returned full-information score is discarded:
    external unqueried/audit outcomes are unavailable during selection.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported E5C method: {method}")
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    rows = []
    for candidate in candidates:
        # Whitelist, never copy/read delta_true, audit, or outcome metadata.
        features = np.asarray(candidate["features"], dtype=float)
        if features.shape != (posterior.feature_dim,) or not np.isfinite(features).all():
            raise ValueError("candidate features do not match the fitted posterior")
        cost = _finite(candidate["hf_query_cost"], "hf_query_cost")
        if cost <= 0:
            raise ValueError("hf_query_cost must be positive")
        rows.append({"transition_id": str(candidate["transition_id"]),
                     "delta_proxy": _finite(candidate["delta_proxy"], "delta_proxy"),
                     "features": features.tolist(), "hf_query_cost": cost})
    if not rows or len({row["transition_id"] for row in rows}) != len(rows):
        raise ValueError("a nonempty panel of unique candidates is required")
    stats = dict(DEFAULT_STATISTICS)
    if statistics is not None:
        stats.update(statistics)
    config = {"statistics": stats}
    # E5C.run clips budget to K, and _select bypasses queries for Proxy Only.
    queried = author_e5._select(method, rows, posterior, min(int(budget), len(rows)), int(seed), config)
    if len(set(queried)) != len(queried) or not set(queried).issubset(row["transition_id"] for row in rows):
        raise RuntimeError("author allocation returned invalid candidate IDs")
    by_id = {row["transition_id"]: row for row in rows}
    observed: dict[str, float] = {}
    ledger = []
    for identifier in queried:
        row = by_id[identifier]
        # A fresh mapping also prevents a query callback from mutating the panel.
        response = query({**row, "features": list(row["features"])})
        if isinstance(response, Mapping):
            if response.get("stream", response.get("split", "selection")) != "selection":
                raise ValueError("only selection responses may enter E5C decisions")
            delta = _finite(response["delta"], "selection delta")
            measured_cost = response.get("total_env_steps", response.get("hf_query_cost"))
            if measured_cost is not None and not math.isclose(
                _finite(measured_cost, "query cost"), row["hf_query_cost"], rel_tol=1e-10, abs_tol=1e-8
            ):
                raise ValueError("query response cost differs from its predeclared cost")
        else:
            delta = _finite(response, "selection delta")
        observed[identifier] = delta
        entry = {"transition_id": identifier, "delta": delta, "cost": row["hf_query_cost"]}
        if isinstance(response, Mapping):
            for key in ("input_files", "training_seeds", "evaluation_seeds", "replicates"):
                if key in response:
                    entry[key] = response[key]
        ledger.append(entry)
    # _select_outcome needs a delta_true field on the final selected row even
    # when unqueried. NaN is an internal unavailable-value marker ONLY; the
    # returned score is discarded and never serialized as a deployment value.
    outcome_rows = [{**row, "delta_true": observed.get(row["transition_id"], float("nan"))} for row in rows]
    selected_id, _ = author_e5._select_outcome(outcome_rows, queried, method, posterior)
    estimates = {
        row["transition_id"]: observed[row["transition_id"]]
        if row["transition_id"] in observed else
        author_e5._posterior_for(row, posterior)[0] if method in CORRECTED_METHODS else row["delta_proxy"]
        for row in rows
    }
    result = {
        "implementation": "author_E5C_fixed_budget", "method": method,
        "budget": int(budget), "budget_packages": int(budget), "candidate_count": len(rows),
        "selected_candidate_id": selected_id,
        "selected": int(selected_id) if selected_id.isdecimal() else selected_id,
        "selected_estimate": estimates[selected_id],
        "selected_query_delta": observed.get(selected_id),
        "estimates": estimates, "queried_ids": list(queried),
        "query_packages_used": len(queried), "hf_budget": len(queried),
        "query_cost": float(sum(item["cost"] for item in ledger)), "query_ledger": ledger,
        "stop_reason": "proxy_only" if method == "proxy_only" else None,
        "statistics": stats, "selection_reads_audit": False,
    }
    # Fail rather than let unavailable/NaN scores leak into a frozen decision.
    json.dumps(result, allow_nan=False)
    return result


def selection_query(
    panel_dir: str | Path, replicates: int = 1,
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """Lazy reader for existing crossbench selection/candidate_j/replicate_r.

    No directory walk and no test summary.json read: historical summaries may
    already contain audit scores. Every package averages a fixed replicate count.
    Symlinks escaping the explicit selection directory are rejected before read.
    """
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    selection_root = Path(panel_dir).resolve() / "selection"

    def read_query(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        identifier = str(candidate["transition_id"])
        if not identifier.isdecimal():
            raise ValueError("file-backed candidate IDs must be nonnegative integers")
        values, costs, paths, training_seeds, evaluation_seeds = [], [], [], [], []
        for replicate in range(replicates):
            path = selection_root / f"candidate_{int(identifier)}" / f"replicate_{replicate}" / "paired.json"
            resolved = path.resolve()
            if not resolved.is_relative_to(selection_root):
                raise ValueError("query path escaped the selection directory")
            encoded = resolved.read_bytes()
            paired = json.loads(encoded)
            if paired.get("stream", paired.get("split", "selection")) != "selection":
                raise ValueError("file is not a selection response")
            if int(paired.get("candidate", identifier)) != int(identifier) or int(paired.get("replicate", replicate)) != replicate:
                raise ValueError("selection response identity does not match its path")
            value = _finite(paired["delta"], "selection delta")
            cost = _finite(paired["total_env_steps"], "selection cost")
            if cost <= 0:
                raise ValueError("selection cost must be positive")
            if "old" in paired or "new" in paired:
                old, new = paired["old"], paired["new"]
                if not old or len(old) != len(new) or [r["seed"] for r in old] != [r["seed"] for r in new]:
                    raise ValueError("incumbent/candidate evaluation seeds are not paired")
                if "evaluation_seeds" in paired and paired["evaluation_seeds"] != [r["seed"] for r in old]:
                    raise ValueError("evaluation seed metadata disagrees with paired records")
                actual = float(np.mean([_finite(n["focal_return"], "return") - _finite(o["focal_return"], "return") for o, n in zip(old, new)]))
                if not math.isclose(value, actual, rel_tol=1e-8, abs_tol=1e-8):
                    raise ValueError("selection delta disagrees with paired returns")
            values.append(value)
            costs.append(cost)
            training_seed = paired.get("training_seed", paired.get("adaptation_seed"))
            if training_seed is not None:
                training_seeds.append(int(training_seed))
            evaluation_seeds.extend(int(s) for s in paired.get("evaluation_seeds", []))
            paths.append({"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest()})
        return {"delta": float(np.mean(values)), "total_env_steps": float(sum(costs)),
                "stream": "selection", "replicates": replicates, "input_files": paths,
                "training_seeds": training_seeds, "evaluation_seeds": evaluation_seeds}

    return read_query
