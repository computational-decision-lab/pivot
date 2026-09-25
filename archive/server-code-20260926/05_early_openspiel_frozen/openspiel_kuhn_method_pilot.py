#!/usr/bin/env python3
"""Sealed OpenSpiel Kuhn Poker method pilot for PIVOT-CG-v2.

The runner has four ordered phases:

  calibrate -> freeze -> select -> audit

Fresh exact deployment values are not computed in ``select``.  Selection uses
only proxy values, the frozen calibration posterior, and candidate-specific
64-pair HF observations.  ``audit`` verifies the selection seal and all source
hashes before exact game-tree values are generated.

One HF pair evaluates candidate+its responder and incumbent+its responder on
the same rollout seed.  Therefore a nominal 64-hand query consumes 64 paired
differences and 128 physical profile hands.  Every method uses the same bank.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyspiel
from open_spiel.python import policy


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from joint_gaussian_selector_v2 import run_joint_gaussian_selector  # noqa: E402
from kuhn_rollout_sampler import StreamPurpose, sample_kuhn_returns  # noqa: E402
from openspiel_kuhn_discovery import (  # noqa: E402
    _best_response_policy,
    _cfr_incumbent,
    _clone,
    _merge_players,
    _mix_player,
    _perturbed_opponent,
    _player_rows,
    _value,
)


VERSION = "openspiel_kuhn_pivot_cg_v2_sealed_pilot_v1"
ALPHAS = tuple(float(value) for value in np.linspace(0.0, 1.0, 9))
HORIZONS = {"short": 1, "long": 8}
BUDGETS = (0, 64, 128, 256, 512)
PRIMARY_BUDGET = 128
QUERY_PAIRS = 64
PHYSICAL_PROFILES_PER_PAIR = 2
ETA = 0.062
BETA_MIN = 0.35
BETA_MAX = 0.95
DIRICHLET_CONCENTRATION = 0.35
CFR_ITERATIONS = 5000
RIDGE = 1e-3
COVARIANCE_SHRINKAGE = 0.20
RANDOM_ALLOCATION_REPLICATES = 4
CALIBRATION_ROOTS = tuple(range(61000, 61096))
FRESH_ROOTS = tuple(range(71000, 71100))
METHODS = (
    "pivot_cg_v2",
    "uniform_random",
    "global_ivr",
    "posterior_lucb",
    "author_global_voi_e5c_adapter",
    "calibrated_no_hf",
    "proxy_only",
    "all_hf_raw_reference",
)

EXPECTED_FROZEN_RUNTIME = {
    "open_spiel": "2.0.2",
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "absl-py": "2.5.0",
    "attrs": "26.1.0",
    "ml-collections": "1.1.0",
    "PyYAML": "6.0.3",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_seed(namespace: str, *parts: object) -> int:
    payload = "|".join([VERSION, namespace, *[str(part) for part in parts]])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _policy_hash(tabular: policy.TabularPolicy) -> str:
    return _sha256_bytes(np.asarray(tabular.action_probability_array, dtype="<f8").tobytes())


def _source_paths() -> dict[str, Path]:
    return {
        "method_runner": Path(__file__).resolve(),
        "discovery_runner": (HERE / "openspiel_kuhn_discovery.py").resolve(),
        "rollout_sampler": (HERE / "kuhn_rollout_sampler.py").resolve(),
        "joint_gaussian_selector": (PARENT / "joint_gaussian_selector_v2.py").resolve(),
        "requirements": (HERE / "requirements.txt").resolve(),
    }


def _source_hashes() -> dict[str, str]:
    return {name: _sha256_file(path) for name, path in _source_paths().items()}


def _runtime_manifest() -> dict[str, Any]:
    versions = {}
    for distribution in EXPECTED_FROZEN_RUNTIME:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "MISSING"
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "requirements_sha256": _sha256_file(HERE / "requirements.txt"),
    }
    payload["environment_sha256"] = _sha256_bytes(_canonical_bytes(payload))
    return payload


def _root_beta(root_seed: int) -> float:
    rng = np.random.default_rng(root_seed + 314159)
    return float(rng.uniform(BETA_MIN, BETA_MAX))


def _candidate_id(index: int) -> str:
    return "incumbent" if index == 0 else f"alpha_{ALPHAS[index]:.3f}"


def _opponent_footprint(opponent: policy.TabularPolicy, rows: np.ndarray) -> np.ndarray:
    return np.asarray(opponent.action_probability_array[rows == 1], dtype=float).reshape(-1)


def _raw_features(
    *, alpha: float, proxy_improvement: float, opponent_footprint: np.ndarray
) -> np.ndarray:
    # All features are observable before deployment.  The root's hidden exact
    # deployment label and response audit never enter this vector.
    return np.concatenate(
        [
            np.asarray(
                [alpha, alpha * alpha, proxy_improvement, alpha * proxy_improvement],
                dtype=float,
            ),
            opponent_footprint,
            alpha * opponent_footprint,
        ]
    )


def _build_world(
    game: pyspiel.Game,
    incumbent: policy.TabularPolicy,
    rows: np.ndarray,
    root_seed: int,
) -> dict[str, Any]:
    beta = _root_beta(root_seed)
    proxy_opponent = _perturbed_opponent(
        game,
        incumbent,
        rows,
        np.random.default_rng(root_seed),
        beta=beta,
        concentration=DIRICHLET_CONCENTRATION,
    )
    joint_incumbent = _merge_players(game, incumbent, proxy_opponent, rows)
    focal_br = _best_response_policy(game, joint_incumbent, player_id=0)
    candidates = [
        _mix_player(game, incumbent, focal_br, 0, alpha, rows) for alpha in ALPHAS
    ]
    proxy_baseline = _value(game, incumbent, proxy_opponent)
    proxy_improvements = np.asarray(
        [_value(game, candidate, proxy_opponent) - proxy_baseline for candidate in candidates],
        dtype=float,
    )
    footprint = _opponent_footprint(proxy_opponent, rows)
    features = np.stack(
        [
            _raw_features(
                alpha=alpha,
                proxy_improvement=float(proxy_improvement),
                opponent_footprint=footprint,
            )
            for alpha, proxy_improvement in zip(ALPHAS, proxy_improvements)
        ]
    )
    responders: dict[str, list[policy.TabularPolicy]] = {}
    incumbent_responders: dict[str, policy.TabularPolicy] = {}
    for horizon, steps in HORIZONS.items():
        weight = 1.0 - (1.0 - ETA) ** steps
        incumbent_br = _best_response_policy(game, joint_incumbent, player_id=1)
        incumbent_responders[horizon] = _mix_player(
            game, proxy_opponent, incumbent_br, 1, weight, rows
        )
        responders[horizon] = []
        for candidate in candidates:
            joint_candidate = _merge_players(game, candidate, proxy_opponent, rows)
            candidate_br = _best_response_policy(game, joint_candidate, player_id=1)
            responders[horizon].append(
                _mix_player(game, proxy_opponent, candidate_br, 1, weight, rows)
            )
    return {
        "root_seed": root_seed,
        "beta": beta,
        "proxy_opponent": proxy_opponent,
        "candidates": candidates,
        "proxy_improvements": proxy_improvements,
        "raw_features": features,
        "responders": responders,
        "incumbent_responders": incumbent_responders,
    }


def _paired_hf_observation(
    *,
    game: pyspiel.Game,
    candidate: policy.TabularPolicy,
    candidate_responder: policy.TabularPolicy,
    incumbent: policy.TabularPolicy,
    incumbent_responder: policy.TabularPolicy,
    seed: int,
    pairs: int = QUERY_PAIRS,
    stream_purpose: StreamPurpose = StreamPurpose.QUERY,
) -> dict[str, Any]:
    # The two profiles use identical hierarchical random streams.  This pairs
    # chance cards and player-specific uniforms without allowing one policy's
    # path length to advance the other profile's RNG.
    stream_key = (stream_purpose, 0)
    candidate_returns = sample_kuhn_returns(
        game,
        (candidate, candidate_responder),
        pairs,
        seed=seed,
        stream_key=stream_key,
    )[:, 0]
    incumbent_returns = sample_kuhn_returns(
        game,
        (incumbent, incumbent_responder),
        pairs,
        seed=seed,
        stream_key=stream_key,
    )[:, 0]
    differences = candidate_returns - incumbent_returns
    return {
        "observation": float(np.mean(differences)),
        "sample_variance": float(np.var(differences, ddof=1)) if pairs > 1 else 0.0,
        "paired_differences": pairs,
        "physical_profile_hands": pairs * PHYSICAL_PROFILES_PER_PAIR,
        "seed": seed,
    }


def _exact_deployment_values(
    game: pyspiel.Game,
    incumbent: policy.TabularPolicy,
    world: Mapping[str, Any],
    horizon: str,
) -> np.ndarray:
    baseline = _value(game, incumbent, world["incumbent_responders"][horizon])
    return np.asarray(
        [
            _value(game, candidate, responder) - baseline
            for candidate, responder in zip(
                world["candidates"], world["responders"][horizon]
            )
        ],
        dtype=float,
    )


def _ridge_fit(raw: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    if raw.ndim != 2 or target.shape != (len(raw),):
        raise ValueError("invalid ridge data")
    mean = np.mean(raw, axis=0)
    scale = np.std(raw, axis=0)
    scale[scale < 1e-10] = 1.0
    design = np.column_stack([np.ones(len(raw)), (raw - mean) / scale])
    penalty = np.eye(design.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return {"mean": mean, "scale": scale, "beta": beta}


def _ridge_predict(model: Mapping[str, np.ndarray], raw: np.ndarray) -> np.ndarray:
    design = np.column_stack(
        [np.ones(len(raw)), (raw - model["mean"]) / model["scale"]]
    )
    return np.asarray(design @ model["beta"], dtype=float)


def _calibrate_horizon(
    *,
    game: pyspiel.Game,
    incumbent: policy.TabularPolicy,
    worlds: Sequence[Mapping[str, Any]],
    horizon: str,
) -> dict[str, Any]:
    n_roots = len(worlds)
    n_candidates = len(ALPHAS)
    raw = np.stack([world["raw_features"] for world in worlds])
    proxy = np.stack([world["proxy_improvements"] for world in worlds])
    exact = np.stack(
        [_exact_deployment_values(game, incumbent, world, horizon) for world in worlds]
    )
    correction = exact - proxy
    loo_prediction = np.zeros_like(correction)
    for held_out in range(n_roots):
        keep = np.arange(n_roots) != held_out
        model = _ridge_fit(
            raw[keep].reshape(-1, raw.shape[-1]), correction[keep].reshape(-1)
        )
        loo_prediction[held_out] = _ridge_predict(model, raw[held_out])
    loo_mean = proxy + loo_prediction
    residuals = exact - loo_mean
    residuals[:, 0] = 0.0
    empirical = np.cov(residuals, rowvar=False, ddof=1)
    empirical = np.asarray(empirical, dtype=float)
    covariance = (
        (1.0 - COVARIANCE_SHRINKAGE) * empirical
        + COVARIANCE_SHRINKAGE * np.diag(np.diag(empirical))
    )
    covariance[0, :] = 0.0
    covariance[:, 0] = 0.0
    diagonal = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    standardized = np.abs(residuals[:, 1:] / diagonal[None, 1:])
    quantile = float(np.quantile(standardized, 0.95))
    conformal_scale = max(1.0, (quantile / 1.96) ** 2)
    covariance *= conformal_scale
    covariance += np.diag([0.0] + [1e-10] * (n_candidates - 1))

    differences_by_candidate: list[list[float]] = [list() for _ in ALPHAS]
    calibration_observation_records = []
    for world in worlds:
        root_seed = int(world["root_seed"])
        for candidate_index in range(1, n_candidates):
            observations = []
            for block in ("a", "b"):
                seed = _stable_seed(
                    f"calibration_{block}", root_seed, horizon, candidate_index
                )
                record = _paired_hf_observation(
                    game=game,
                    candidate=world["candidates"][candidate_index],
                    candidate_responder=world["responders"][horizon][candidate_index],
                    incumbent=incumbent,
                    incumbent_responder=world["incumbent_responders"][horizon],
                    seed=seed,
                    stream_purpose=StreamPurpose.CALIBRATION,
                )
                observations.append(float(record["observation"]))
                calibration_observation_records.append(
                    {
                        "root_seed": root_seed,
                        "horizon": horizon,
                        "candidate_index": candidate_index,
                        "block": block,
                        **record,
                    }
                )
            differences_by_candidate[candidate_index].append(observations[0] - observations[1])
    observation_noise = np.zeros(n_candidates, dtype=float)
    for candidate_index in range(1, n_candidates):
        # Two independent 64-pair means have Var(A-B)=2R.
        observation_noise[candidate_index] = max(
            float(np.var(differences_by_candidate[candidate_index], ddof=1) / 2.0),
            1e-6,
        )

    full_model = _ridge_fit(raw.reshape(-1, raw.shape[-1]), correction.reshape(-1))
    marginal_sd = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    coverage = float(
        np.mean(
            np.abs(residuals[:, 1:])
            <= 1.96 * marginal_sd[None, 1:]
        )
    )
    return {
        "feature_mean": full_model["mean"].tolist(),
        "feature_scale": full_model["scale"].tolist(),
        "coefficient_mean": full_model["beta"].tolist(),
        "latent_covariance": covariance.tolist(),
        "observation_noise": observation_noise.tolist(),
        "conformal_scale": conformal_scale,
        "loo_marginal_95_coverage": coverage,
        "loo_rmse": float(np.sqrt(np.mean(residuals[:, 1:] ** 2))),
        "exact_value_summary": {
            "mean_abs_gap": float(np.mean(np.abs(proxy - exact))),
            "proxy_deployment_mismatch_rate": float(
                np.mean(np.argmax(proxy, axis=1) != np.argmax(exact, axis=1))
            ),
        },
        "calibration_observations": calibration_observation_records,
    }


def _posterior_for_world(
    calibration: Mapping[str, Any], world: Mapping[str, Any], horizon: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = calibration["horizons"][horizon]
    fit = {
        "mean": np.asarray(model["feature_mean"], dtype=float),
        "scale": np.asarray(model["feature_scale"], dtype=float),
        "beta": np.asarray(model["coefficient_mean"], dtype=float),
    }
    correction = _ridge_predict(fit, np.asarray(world["raw_features"], dtype=float))
    mean = np.asarray(world["proxy_improvements"], dtype=float) + correction
    covariance = np.asarray(model["latent_covariance"], dtype=float)
    noise = np.asarray(model["observation_noise"], dtype=float)
    mean[0] = 0.0
    covariance[0, :] = 0.0
    covariance[:, 0] = 0.0
    noise[0] = 0.0
    return mean, covariance, noise


def calibrate(output_dir: Path, *, roots: Sequence[int] = CALIBRATION_ROOTS) -> dict[str, Any]:
    game = pyspiel.load_game("kuhn_poker")
    incumbent = _cfr_incumbent(game, CFR_ITERATIONS)
    rows = _player_rows(incumbent)
    worlds = [_build_world(game, incumbent, rows, root_seed) for root_seed in roots]
    horizons = {
        horizon: _calibrate_horizon(
            game=game, incumbent=incumbent, worlds=worlds, horizon=horizon
        )
        for horizon in HORIZONS
    }
    payload = {
        "version": VERSION,
        "status": "development_calibration_only",
        "source_hashes_at_calibration": _source_hashes(),
        "runtime_manifest_at_calibration": _runtime_manifest(),
        "root_seeds": list(roots),
        "root_seed_count": len(roots),
        "incumbent_policy_sha256": _policy_hash(incumbent),
        "horizons": horizons,
        "query_definition": {
            "paired_differences_per_query": QUERY_PAIRS,
            "physical_profile_hands_per_query": QUERY_PAIRS * PHYSICAL_PROFILES_PER_PAIR,
            "common_random_numbers": True,
        },
        "seed_namespaces": ["calibration_a", "calibration_b"],
    }
    path = output_dir / "calibration.json"
    _write_json(path, payload)
    return payload


def _protocol_core(
    *, output_dir: Path, calibration: Mapping[str, Any], profile: str
) -> dict[str, Any]:
    game = pyspiel.load_game("kuhn_poker")
    incumbent = _cfr_incumbent(game, CFR_ITERATIONS)
    fresh_roots = FRESH_ROOTS if profile == "frozen" else tuple(range(72000, 72004))
    budgets = BUDGETS if profile == "frozen" else (0, 64, 128)
    return {
        "version": VERSION,
        "profile": profile,
        "status": "FROZEN_BEFORE_FRESH_SELECTION",
        "game": "kuhn_poker",
        "open_spiel_version": getattr(pyspiel, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "cfr_iterations": CFR_ITERATIONS,
        "incumbent_policy_sha256": _policy_hash(incumbent),
        "candidate_alphas": list(ALPHAS),
        "eta": ETA,
        "horizons": HORIZONS,
        "beta_range": [BETA_MIN, BETA_MAX],
        "dirichlet_concentration": DIRICHLET_CONCENTRATION,
        "calibration_roots": list(calibration["root_seeds"]),
        "fresh_roots": list(fresh_roots),
        "budgets_paired_hands": list(budgets),
        "primary_budget_paired_hands": PRIMARY_BUDGET,
        "query_pairs": QUERY_PAIRS,
        "physical_profile_hands_per_query": QUERY_PAIRS * PHYSICAL_PROFILES_PER_PAIR,
        "random_allocation_replicates": RANDOM_ALLOCATION_REPLICATES,
        "methods": list(METHODS),
        "primary_contrast": "mean uniform_random regret - pivot_cg_v2 regret, long response, budget 128",
        "source_hashes": _source_hashes(),
        "runtime_manifest": _runtime_manifest(),
        "calibration_sha256": _sha256_file(output_dir / "calibration.json"),
        "seed_derivation": "SHA256(version|namespace|root|horizon|candidate|replicate)",
        "selection_seed_namespace": "fresh_selection",
        "audit_target": "exact expected_game_score generated only after selection seal",
        "audit_label_visible_to_selector": False,
        "all_hf_definition": "8 queryable non-incumbent arms x 64 paired differences = 512 nominal paired hands = 1024 physical profile hands; raw queried observations; noisy reference, not oracle",
        "posterior": {
            "mean": "ridge correction over proxy/alpha/opponent-footprint features",
            "covariance": "root-LOO residual covariance; 0.20 diagonal shrinkage; marginal conformal scale",
            "query_noise": "Var(calibration_A-calibration_B)/2 for independent 64-pair blocks",
        },
    }


def freeze(output_dir: Path, *, profile: str) -> dict[str, Any]:
    calibration_path = output_dir / "calibration.json"
    if not calibration_path.exists():
        raise FileNotFoundError("calibration.json must exist before freeze")
    calibration = _read_json(calibration_path)
    runtime = _runtime_manifest()
    if calibration.get("source_hashes_at_calibration") != _source_hashes():
        raise RuntimeError(
            "calibration was generated by different source files; rerun calibrate"
        )
    if calibration.get("runtime_manifest_at_calibration") != runtime:
        raise RuntimeError(
            "calibration was generated in a different runtime; rerun calibrate"
        )
    if profile == "frozen" and runtime["packages"] != EXPECTED_FROZEN_RUNTIME:
        raise RuntimeError(
            "frozen runtime differs from requirements lock: "
            f"expected {EXPECTED_FROZEN_RUNTIME}, observed {runtime['packages']}"
        )
    core = _protocol_core(output_dir=output_dir, calibration=calibration, profile=profile)
    lock = {**core, "lock_sha256": _sha256_bytes(_canonical_bytes(core))}
    _write_json(output_dir / "protocol_lock.json", lock)
    return lock


def _verify_lock(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = _read_json(output_dir / "protocol_lock.json")
    supplied_hash = lock.pop("lock_sha256")
    if _sha256_bytes(_canonical_bytes(lock)) != supplied_hash:
        raise RuntimeError("protocol lock hash mismatch")
    lock["lock_sha256"] = supplied_hash
    if lock["source_hashes"] != _source_hashes():
        raise RuntimeError("source files changed after protocol freeze")
    if lock["runtime_manifest"] != _runtime_manifest():
        raise RuntimeError("runtime environment changed after protocol freeze")
    if lock["calibration_sha256"] != _sha256_file(output_dir / "calibration.json"):
        raise RuntimeError("calibration changed after protocol freeze")
    calibration = _read_json(output_dir / "calibration.json")
    return lock, calibration


def _selection_bank_for_world(
    *,
    game: pyspiel.Game,
    incumbent: policy.TabularPolicy,
    world: Mapping[str, Any],
    horizon: str,
) -> dict[str, dict[str, Any]]:
    bank = {}
    for candidate_index in range(1, len(ALPHAS)):
        seed = _stable_seed(
            "fresh_selection", world["root_seed"], horizon, candidate_index
        )
        bank[_candidate_id(candidate_index)] = _paired_hf_observation(
            game=game,
            candidate=world["candidates"][candidate_index],
            candidate_responder=world["responders"][horizon][candidate_index],
            incumbent=incumbent,
            incumbent_responder=world["incumbent_responders"][horizon],
            seed=seed,
        )
    return bank


def _run_active_method(
    *,
    method: str,
    mean: np.ndarray,
    covariance: np.ndarray,
    noise: np.ndarray,
    bank: Mapping[str, Mapping[str, Any]],
    budget: int,
    seed: int,
) -> dict[str, Any]:
    candidate_ids = [_candidate_id(index) for index in range(len(ALPHAS))]
    requested: list[str] = []

    def query(payload: Mapping[str, Any]) -> dict[str, float]:
        candidate_id = str(payload["candidate_id"])
        if candidate_id == "incumbent" or candidate_id not in bank:
            raise RuntimeError("selector requested an invalid HF candidate")
        if candidate_id in requested:
            raise RuntimeError("selector repeated an HF query")
        requested.append(candidate_id)
        return {
            "observation": float(bank[candidate_id]["observation"]),
            "cost": float(QUERY_PAIRS),
        }

    selector_method = {
        "pivot_cg_v2": "joint_gaussian_voi",
        "uniform_random": "uniform_random",
        "global_ivr": "global_ivr",
        "posterior_lucb": "posterior_lucb",
    }[method]
    result = run_joint_gaussian_selector(
        candidate_ids,
        mean,
        covariance,
        noise,
        [float(QUERY_PAIRS)] * len(ALPHAS),
        query,
        method=selector_method,
        budget=float(budget),
        seed=seed,
        queryable=[False] + [True] * (len(ALPHAS) - 1),
        incumbent_id="incumbent",
        stop_on_zero_voi=False,
    )
    if requested != list(result["queried_ids"]):
        raise RuntimeError("query callback trace differs from selector trace")
    if not math.isclose(float(result["spent_cost"]), QUERY_PAIRS * len(requested)):
        raise RuntimeError("selector cost ledger mismatch")
    return {
        "selected_id": str(result["selected_id"]),
        "queried_ids": requested,
        "nominal_paired_hands": int(result["spent_cost"]),
        "physical_profile_hands": int(
            PHYSICAL_PROFILES_PER_PAIR * result["spent_cost"]
        ),
        "selector_stop_reason": result["stop_reason"],
        "selector_trace": result["steps"],
    }


def _author_global_voi_ranking(
    *,
    proxy: np.ndarray,
    covariance: np.ndarray,
    noise: np.ndarray,
) -> list[int]:
    """Exact author E5C batch Global-VOI ordering for this adapter.

    The repository rule is predictive_std / (1 + abs(delta_proxy)).  Candidate
    zero is the incumbent and is deliberately excluded from the query ranking.
    """
    predictive_std = np.sqrt(np.maximum(np.diag(covariance) + noise, 0.0))
    return sorted(
        range(1, len(proxy)),
        key=lambda index: (
            -float(predictive_std[index] / (1.0 + abs(float(proxy[index])))),
            index,
        ),
    )


def _run_author_global_voi_e5c(
    *,
    mean: np.ndarray,
    covariance: np.ndarray,
    noise: np.ndarray,
    proxy: np.ndarray,
    bank: Mapping[str, Mapping[str, Any]],
    budget: int,
) -> dict[str, Any]:
    if budget % QUERY_PAIRS != 0:
        raise ValueError("author Global-VOI budget must be a multiple of query cost")
    query_count = min(budget // QUERY_PAIRS, len(ALPHAS) - 1)
    ranking = _author_global_voi_ranking(
        proxy=proxy, covariance=covariance, noise=noise
    )
    queried_indices = ranking[:query_count]
    estimates = np.asarray(mean, dtype=float).copy()
    trace = []
    for order, index in enumerate(queried_indices):
        candidate_id = _candidate_id(index)
        observation = float(bank[candidate_id]["observation"])
        # Preserve the author's E5C terminal semantics: a queried candidate is
        # raw-overwritten; unqueried candidates retain the initial posterior mean.
        estimates[index] = observation
        trace.append(
            {
                "step": order,
                "query_id": candidate_id,
                "observation": observation,
                "charged_cost": QUERY_PAIRS,
                "author_e5c_score": float(
                    math.sqrt(max(covariance[index, index] + noise[index], 0.0))
                    / (1.0 + abs(float(proxy[index])))
                ),
            }
        )
    selected = int(np.argmax(estimates))
    return {
        "selected_id": _candidate_id(selected),
        "queried_ids": [_candidate_id(index) for index in queried_indices],
        "nominal_paired_hands": QUERY_PAIRS * query_count,
        "physical_profile_hands": (
            PHYSICAL_PROFILES_PER_PAIR * QUERY_PAIRS * query_count
        ),
        "selector_stop_reason": "author_e5c_fixed_batch",
        "selector_trace": trace,
        "terminal_semantics": "queried_raw_observation_unqueried_initial_posterior_mean",
        "author_entry_point_parity": "experiments.v9.e5c_efficiency._select global_voi score",
    }


def _run_all_hf_raw_reference(
    *, bank: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    queried_ids = [_candidate_id(index) for index in range(1, len(ALPHAS))]
    estimates = np.zeros(len(ALPHAS), dtype=float)
    for index in range(1, len(ALPHAS)):
        estimates[index] = float(bank[_candidate_id(index)]["observation"])
    selected = int(np.argmax(estimates))
    return {
        "selected_id": _candidate_id(selected),
        "queried_ids": queried_ids,
        "nominal_paired_hands": 512,
        "physical_profile_hands": 1024,
        "selector_stop_reason": "all_queryable_candidates_observed",
        "selector_trace": [
            {
                "step": order,
                "query_id": candidate_id,
                "observation": float(bank[candidate_id]["observation"]),
                "charged_cost": QUERY_PAIRS,
            }
            for order, candidate_id in enumerate(queried_ids)
        ],
        "terminal_semantics": "queried_raw_observation_incumbent_zero",
        "oracle": False,
    }


def run_fresh_selection(output_dir: Path) -> dict[str, Any]:
    """Run fresh selection without computing exact deployment audit values."""
    if (output_dir / "audit_results.json").exists():
        raise RuntimeError("audit already exists; use a new directory for fresh selection")
    lock, calibration = _verify_lock(output_dir)
    game = pyspiel.load_game("kuhn_poker")
    incumbent = _cfr_incumbent(game, CFR_ITERATIONS)
    if _policy_hash(incumbent) != lock["incumbent_policy_sha256"]:
        raise RuntimeError("incumbent differs from frozen hash")
    rows = _player_rows(incumbent)
    decisions = []
    selection_banks = []
    fresh_inputs = []
    budgets = tuple(int(value) for value in lock["budgets_paired_hands"])
    for root_seed in lock["fresh_roots"]:
        world = _build_world(game, incumbent, rows, int(root_seed))
        fresh_inputs.append(
            {
                "root_seed": int(root_seed),
                "beta": float(world["beta"]),
                "proxy_improvements": world["proxy_improvements"].tolist(),
                "raw_features_sha256": _sha256_bytes(
                    np.asarray(world["raw_features"], dtype="<f8").tobytes()
                ),
            }
        )
        for horizon in HORIZONS:
            mean, covariance, noise = _posterior_for_world(calibration, world, horizon)
            bank = _selection_bank_for_world(
                game=game, incumbent=incumbent, world=world, horizon=horizon
            )
            selection_banks.append(
                {
                    "root_seed": int(root_seed),
                    "horizon": horizon,
                    "observations": bank,
                }
            )
            for budget in budgets:
                for method in ("pivot_cg_v2", "global_ivr", "posterior_lucb"):
                    result = _run_active_method(
                        method=method,
                        mean=mean.copy(),
                        covariance=covariance.copy(),
                        noise=noise.copy(),
                        bank=bank,
                        budget=budget,
                        seed=_stable_seed("method", root_seed, horizon, budget, method),
                    )
                    decisions.append(
                        {
                            "root_seed": int(root_seed),
                            "horizon": horizon,
                            "budget": budget,
                            "method": method,
                            "allocation_replicate": 0,
                            **result,
                        }
                    )
                for replicate in range(RANDOM_ALLOCATION_REPLICATES):
                    result = _run_active_method(
                        method="uniform_random",
                        mean=mean.copy(),
                        covariance=covariance.copy(),
                        noise=noise.copy(),
                        bank=bank,
                        budget=budget,
                        # The same root/horizon/replicate permutation is used
                        # at every budget, so smaller budgets are exact prefixes.
                        seed=_stable_seed(
                            "random_allocation", root_seed, horizon, replicate
                        ),
                    )
                    decisions.append(
                        {
                            "root_seed": int(root_seed),
                            "horizon": horizon,
                            "budget": budget,
                            "method": "uniform_random",
                            "allocation_replicate": replicate,
                            **result,
                        }
                    )
                author_global = _run_author_global_voi_e5c(
                    mean=mean.copy(),
                    covariance=covariance.copy(),
                    noise=noise.copy(),
                    proxy=np.asarray(world["proxy_improvements"], dtype=float),
                    bank=bank,
                    budget=budget,
                )
                decisions.append(
                    {
                        "root_seed": int(root_seed),
                        "horizon": horizon,
                        "budget": budget,
                        "method": "author_global_voi_e5c_adapter",
                        "allocation_replicate": 0,
                        **author_global,
                    }
                )
                decisions.extend(
                    [
                        {
                            "root_seed": int(root_seed),
                            "horizon": horizon,
                            "budget": budget,
                            "method": "calibrated_no_hf",
                            "allocation_replicate": 0,
                            "selected_id": _candidate_id(int(np.argmax(mean))),
                            "queried_ids": [],
                            "nominal_paired_hands": 0,
                            "physical_profile_hands": 0,
                            "selector_stop_reason": "no_hf_method",
                            "selector_trace": [],
                        },
                        {
                            "root_seed": int(root_seed),
                            "horizon": horizon,
                            "budget": budget,
                            "method": "proxy_only",
                            "allocation_replicate": 0,
                            "selected_id": _candidate_id(
                                int(np.argmax(world["proxy_improvements"]))
                            ),
                            "queried_ids": [],
                            "nominal_paired_hands": 0,
                            "physical_profile_hands": 0,
                            "selector_stop_reason": "no_hf_method",
                            "selector_trace": [],
                        },
                    ]
                )
            all_hf = _run_all_hf_raw_reference(bank=bank)
            decisions.append(
                {
                    "root_seed": int(root_seed),
                    "horizon": horizon,
                    "budget": 512,
                    "method": "all_hf_raw_reference",
                    "allocation_replicate": 0,
                    **all_hf,
                }
            )
    bank_path = output_dir / "fresh_selection_observations.json"
    input_path = output_dir / "fresh_proxy_inputs.json"
    decision_path = output_dir / "fresh_decisions.json"
    _write_json(bank_path, selection_banks)
    _write_json(input_path, fresh_inputs)
    selection_payload = {
        "version": VERSION,
        "status": "SEALED_BEFORE_EXACT_AUDIT",
        "protocol_lock_sha256": lock["lock_sha256"],
        "selection_observations_sha256": _sha256_file(bank_path),
        "fresh_proxy_inputs_sha256": _sha256_file(input_path),
        "decisions": decisions,
        "exact_audit_generated": False,
        "audit_labels_used": False,
    }
    _write_json(decision_path, selection_payload)
    seal_core = {
        "version": VERSION,
        "status": "SEALED_BEFORE_EXACT_AUDIT",
        "protocol_lock_sha256": lock["lock_sha256"],
        "decisions_sha256": _sha256_file(decision_path),
        "selection_observations_sha256": _sha256_file(bank_path),
        "fresh_proxy_inputs_sha256": _sha256_file(input_path),
        "source_hashes": _source_hashes(),
        "fresh_root_count": len(lock["fresh_roots"]),
        "decision_count": len(decisions),
    }
    seal = {**seal_core, "seal_sha256": _sha256_bytes(_canonical_bytes(seal_core))}
    _write_json(output_dir / "selection_seal.json", seal)
    return seal


def _bootstrap_mean(values: Sequence[float], *, seed: int, draws: int = 5000) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.mean(
        array[rng.integers(0, len(array), size=(draws, len(array)))], axis=1
    )
    return {
        "mean": float(np.mean(array)),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "n_roots": len(array),
    }


def _verify_selection_seal(output_dir: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    seal = _read_json(output_dir / "selection_seal.json")
    supplied = seal.pop("seal_sha256")
    if _sha256_bytes(_canonical_bytes(seal)) != supplied:
        raise RuntimeError("selection seal self-hash mismatch")
    seal["seal_sha256"] = supplied
    if seal["protocol_lock_sha256"] != lock["lock_sha256"]:
        raise RuntimeError("selection seal points to another protocol")
    if seal["source_hashes"] != _source_hashes():
        raise RuntimeError("source changed between selection and audit")
    checks = {
        "fresh_decisions.json": "decisions_sha256",
        "fresh_selection_observations.json": "selection_observations_sha256",
        "fresh_proxy_inputs.json": "fresh_proxy_inputs_sha256",
    }
    for filename, key in checks.items():
        if _sha256_file(output_dir / filename) != seal[key]:
            raise RuntimeError(f"sealed artifact changed: {filename}")
    return seal


def audit_fresh_selection(output_dir: Path) -> dict[str, Any]:
    lock, _ = _verify_lock(output_dir)
    seal = _verify_selection_seal(output_dir, lock)
    selections = _read_json(output_dir / "fresh_decisions.json")
    if selections["exact_audit_generated"] or selections["audit_labels_used"]:
        raise RuntimeError("selection payload violates audit separation")
    game = pyspiel.load_game("kuhn_poker")
    incumbent = _cfr_incumbent(game, CFR_ITERATIONS)
    rows = _player_rows(incumbent)
    exact_rows = []
    exact_lookup: dict[tuple[int, str], np.ndarray] = {}
    proxy_lookup: dict[int, np.ndarray] = {}
    for root_seed in lock["fresh_roots"]:
        world = _build_world(game, incumbent, rows, int(root_seed))
        proxy_lookup[int(root_seed)] = np.asarray(world["proxy_improvements"], dtype=float)
        for horizon in HORIZONS:
            exact = _exact_deployment_values(game, incumbent, world, horizon)
            exact_lookup[(int(root_seed), horizon)] = exact
            for candidate_index, value in enumerate(exact):
                exact_rows.append(
                    {
                        "root_seed": int(root_seed),
                        "horizon": horizon,
                        "candidate_index": candidate_index,
                        "candidate_id": _candidate_id(candidate_index),
                        "proxy_improvement": float(world["proxy_improvements"][candidate_index]),
                        "exact_deployment_improvement": float(value),
                    }
                )
    scored = []
    id_to_index = {_candidate_id(index): index for index in range(len(ALPHAS))}
    for decision in selections["decisions"]:
        key = (int(decision["root_seed"]), str(decision["horizon"]))
        exact = exact_lookup[key]
        selected_index = id_to_index[str(decision["selected_id"])]
        selected_value = float(exact[selected_index])
        oracle_index = int(np.argmax(exact))
        scored.append(
            {
                **decision,
                "selected_index": selected_index,
                "selected_exact_value": selected_value,
                "oracle_index": oracle_index,
                "oracle_id": _candidate_id(oracle_index),
                "oracle_exact_value": float(exact[oracle_index]),
                "selection_regret": float(exact[oracle_index] - selected_value),
                "correct_selection": bool(selected_index == oracle_index),
                "harmful_promotion": bool(selected_value < 0.0),
            }
        )
    groups: dict[tuple[str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        groups[
            (
                str(row["horizon"]),
                int(row["budget"]),
                str(row["method"]),
                int(row["allocation_replicate"]),
            )
        ].append(row)
    summaries = []
    for (horizon, budget, method, replicate), group in sorted(groups.items()):
        summaries.append(
            {
                "horizon": horizon,
                "budget": budget,
                "method": method,
                "allocation_replicate": replicate,
                "mean_selection_regret": float(
                    np.mean([row["selection_regret"] for row in group])
                ),
                "selection_accuracy": float(
                    np.mean([row["correct_selection"] for row in group])
                ),
                "harmful_promotion_rate": float(
                    np.mean([row["harmful_promotion"] for row in group])
                ),
                "mean_nominal_paired_hands": float(
                    np.mean([row["nominal_paired_hands"] for row in group])
                ),
                "mean_physical_profile_hands": float(
                    np.mean([row["physical_profile_hands"] for row in group])
                ),
                "n_roots": len(group),
            }
        )

    def per_root_method(horizon: str, budget: int, method: str) -> dict[int, float]:
        selected = [
            row
            for row in scored
            if row["horizon"] == horizon
            and row["budget"] == budget
            and row["method"] == method
        ]
        by_root: dict[int, list[float]] = defaultdict(list)
        for row in selected:
            by_root[int(row["root_seed"])].append(float(row["selection_regret"]))
        return {root: float(np.mean(values)) for root, values in by_root.items()}

    contrasts = {}
    advantages: dict[str, list[float]] = {}
    for horizon in HORIZONS:
        pivot = per_root_method(horizon, PRIMARY_BUDGET, "pivot_cg_v2")
        random = per_root_method(horizon, PRIMARY_BUDGET, "uniform_random")
        roots = sorted(set(pivot) & set(random))
        values = [random[root] - pivot[root] for root in roots]
        advantages[horizon] = values
        contrasts[f"random_minus_pivot_{horizon}_budget_{PRIMARY_BUDGET}"] = _bootstrap_mean(
            values,
            seed=_stable_seed("bootstrap_primary", horizon),
        )
    interaction_values = [
        long_value - short_value
        for long_value, short_value in zip(advantages["long"], advantages["short"])
    ]
    contrasts["pivot_advantage_long_minus_short_interaction"] = _bootstrap_mean(
        interaction_values, seed=_stable_seed("bootstrap_interaction")
    )

    mechanism = {}
    per_root_gap: dict[str, list[float]] = {"short": [], "long": []}
    per_root_proxy_regret: dict[str, list[float]] = {"short": [], "long": []}
    for horizon in HORIZONS:
        mismatches = []
        winners = Counter()
        for root_seed in lock["fresh_roots"]:
            proxy = proxy_lookup[int(root_seed)]
            exact = exact_lookup[(int(root_seed), horizon)]
            per_root_gap[horizon].append(float(np.mean(np.abs(proxy - exact))))
            proxy_best = int(np.argmax(proxy))
            oracle = int(np.argmax(exact))
            mismatches.append(proxy_best != oracle)
            winners[oracle] += 1
            per_root_proxy_regret[horizon].append(float(exact[oracle] - exact[proxy_best]))
        mechanism[horizon] = {
            "mean_abs_gap": float(np.mean(per_root_gap[horizon])),
            "proxy_deployment_mismatch_rate": float(np.mean(mismatches)),
            "mean_proxy_best_selection_regret": float(
                np.mean(per_root_proxy_regret[horizon])
            ),
            "winner_counts": {str(key): value for key, value in sorted(winners.items())},
            "max_winner_share": max(winners.values()) / len(lock["fresh_roots"]),
        }
    mechanism["long_minus_short"] = {
        "mean_abs_gap": _bootstrap_mean(
            np.asarray(per_root_gap["long"]) - np.asarray(per_root_gap["short"]),
            seed=_stable_seed("bootstrap_mechanism_gap"),
        ),
        "proxy_best_selection_regret": _bootstrap_mean(
            np.asarray(per_root_proxy_regret["long"])
            - np.asarray(per_root_proxy_regret["short"]),
            seed=_stable_seed("bootstrap_mechanism_regret"),
        ),
    }
    primary = contrasts[f"random_minus_pivot_long_budget_{PRIMARY_BUDGET}"]
    interaction = contrasts["pivot_advantage_long_minus_short_interaction"]
    decision = {
        "mechanism_supported": bool(
            mechanism["long_minus_short"]["mean_abs_gap"]["ci_low"] > 0.0
            and mechanism["long_minus_short"]["proxy_best_selection_regret"]["ci_low"]
            > 0.0
        ),
        "primary_pivot_advantage_supported": bool(primary["ci_low"] > 0.0),
        "advantage_interaction_supported": bool(interaction["ci_low"] > 0.0),
    }
    decision["full_mechanism_to_method_chain_supported"] = all(decision.values())
    payload = {
        "version": VERSION,
        "status": "AUDITED_AFTER_SELECTION_SEAL",
        "protocol_lock_sha256": lock["lock_sha256"],
        "selection_seal_sha256": seal["seal_sha256"],
        "audit_target": "exact OpenSpiel expected_game_score",
        "selection_audit_seed_overlap": False,
        "exact_candidate_values": exact_rows,
        "scored_decisions": scored,
        "summaries": summaries,
        "contrasts": contrasts,
        "mechanism": mechanism,
        "decision": decision,
    }
    _write_json(output_dir / "audit_results.json", payload)
    return payload


def _validate_artifacts(output_dir: Path) -> dict[str, Any]:
    calibration_path = output_dir / "calibration.json"
    if not calibration_path.exists():
        raise FileNotFoundError("calibration.json is missing")
    calibration = _read_json(calibration_path)
    report: dict[str, Any] = {
        "calibration_exists": True,
        "calibration_root_count": len(calibration["root_seeds"]),
        "calibration_seed_namespaces": list(calibration["seed_namespaces"]),
    }
    # A calibrate-only phase intentionally precedes protocol freeze.  Validate
    # the artifact that exists without pretending a future lock is present.
    if not (output_dir / "protocol_lock.json").exists():
        report["protocol_lock_exists"] = False
        report["all_pass"] = True
        _write_json(output_dir / "validation.json", report)
        return report

    lock, calibration = _verify_lock(output_dir)
    report.update({
        "protocol_lock_exists": True,
        "lock_valid": True,
        "calibration_roots_disjoint_fresh": set(lock["calibration_roots"]).isdisjoint(
            lock["fresh_roots"]
        ),
        "fresh_roots_unique": len(lock["fresh_roots"]) == len(set(lock["fresh_roots"])),
        "seed_namespaces_disjoint": set(calibration["seed_namespaces"]).isdisjoint(
            {lock["selection_seed_namespace"]}
        ),
    })
    if (output_dir / "selection_seal.json").exists():
        seal = _verify_selection_seal(output_dir, lock)
        report["selection_seal_valid"] = True
        report["decision_count"] = seal["decision_count"]
    report["all_pass"] = all(
        value for key, value in report.items() if isinstance(value, bool)
    )
    _write_json(output_dir / "validation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("calibrate", "freeze", "select", "audit", "validate", "all"),
        required=True,
    )
    parser.add_argument("--profile", choices=("smoke", "frozen"), default="frozen")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase in ("calibrate", "all"):
        roots = CALIBRATION_ROOTS if args.profile == "frozen" else tuple(range(62000, 62008))
        calibrate(output_dir, roots=roots)
    if args.phase in ("freeze", "all"):
        freeze(output_dir, profile=args.profile)
    if args.phase in ("select", "all"):
        run_fresh_selection(output_dir)
    if args.phase in ("audit", "all"):
        audit_fresh_selection(output_dir)
    validation = _validate_artifacts(output_dir)
    print(
        json.dumps(
            {
                "version": VERSION,
                "phase": args.phase,
                "profile": args.profile,
                "output_dir": str(output_dir),
                "validation": validation,
                "audit_exists": (output_dir / "audit_results.json").exists(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
