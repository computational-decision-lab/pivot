"""Offline MetaDrive adapter calling the unchanged author's E5C implementation.

Import with both <author-repo>/src and <author-repo> on PYTHONPATH.  No native
episodes are generated here.  This is an author-code domain adapter, not a
literal rerun of the paper's original environments.  Four fixed, normalized
controller differences are the domain footprint; no response labels enter it.
The no-update option is admitted at final selection with value exactly zero.
Author acquisition retains its original candidate-only, top-b batch semantics.
Only inside the author model, returns are divided by the protocol's fixed planned
horizon (300), not realized episode length or any observed calibration statistic.
Selections are scored and all displayed gain estimates use native return units.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

METHOD_MAP = {
    "author_PIVOT_VOI_E5C": "pivot_voi",
    "author_GlobalVOI_E5C": "global_voi",
    "author_Uniform_E5C": "random_hf",
    "author_LUCB_E5C": "paired_lucb",
}
PARAMETER_KEYS = ("speed", "headway", "distance", "lane_change_distance")
FEATURE_SCALES = np.asarray([20.0, 1.5, 5.0, 15.0])
# Frozen values in the author's configs/v9/e5c.yaml, not tuned on MetaDrive.
AUTHOR_STATISTICS = {"voi_fantasies": 8, "voi_posterior_samples": 32}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _author():
    from experiments.v9 import e5c_efficiency as e5c
    from pivot.acquisition.pivot_voi import BayesianLinearDeltaPosterior
    return e5c, BayesianLinearDeltaPosterior


def _provenance():
    e5c, _ = _author()
    repo = Path(e5c.__file__).resolve().parents[2]
    files = {}
    for name, module in tuple(sys.modules.items()):
        if not (name == "pivot" or name.startswith("pivot.") or
                name == "experiments" or name.startswith("experiments.")):
            continue
        raw_path = getattr(module, "__file__", None)
        if not raw_path:
            continue
        path = Path(raw_path).resolve()
        if path.suffix == ".py" and path.is_relative_to(repo):
            files[str(path.relative_to(repo))] = _sha(path)
    if not files:
        raise RuntimeError("Author source provenance is empty")
    config = repo / "configs/v9/e5c.yaml"
    return {
        "author_repo": str(repo), "author_source_sha256": dict(sorted(files.items())),
        "author_e5c_config_sha256": _sha(config) if config.exists() else None,
        "adapter_source_sha256": _sha(__file__),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
    }


def _verify_provenance(fit):
    e5c, _ = _author()
    repo = Path(e5c.__file__).resolve().parents[2]
    for relative, expected in fit["provenance"]["author_source_sha256"].items():
        if _sha(repo / relative) != expected:
            raise ValueError("Author source differs from frozen fit: " + relative)
    if _sha(__file__) != fit["provenance"]["adapter_source_sha256"]:
        raise ValueError("Author adapter differs from frozen fit")


def _id(root):
    return int(root["root"] if "root" in root else root["seed"])


def _inputs(root, h):
    adaptations = list(root.get("adaptation_steps", [4, 12]))
    hi = adaptations.index(int(h))
    proxy = np.asarray(root["proxy"], dtype=float)
    bank = np.asarray(root["selection"], dtype=float)
    if proxy.shape != (6,) or bank.ndim != 3 or bank.shape[:2] != (2, 6) or bank.shape[2] < 2:
        raise ValueError("Expected six candidates and selection (2,6,S>=2)")
    if not np.isfinite(proxy).all() or not np.isfinite(bank).all():
        raise ValueError("Nonfinite proxy/selection input")
    return hi, proxy, bank[hi]


def _labels(root, hi):
    aa, ab = [np.asarray(root[key], dtype=float) for key in ("audit_a", "audit_b")]
    if aa.shape != ab.shape or aa.ndim != 3 or aa.shape[:2] != (2, 6) or aa.shape[2] < 2:
        raise ValueError("Expected independent audit splits (2,6,A>=2)")
    if not np.isfinite(aa).all() or not np.isfinite(ab).all():
        raise ValueError("Nonfinite audit input")
    return (aa[hi].mean(axis=1) + ab[hi].mean(axis=1)) / 2.0


def fit_author(calroots: list[Mapping], h: int, protocol: Mapping) -> dict:
    """Fit original E5C posterior using only independent calibration roots."""
    e5c, _ = _author()
    if len(calroots) < 2 or len({_id(r) for r in calroots}) != len(calroots):
        raise ValueError("Need at least two unique calibration roots")
    incumbent = np.asarray([protocol["incumbent"][k] for k in PARAMETER_KEYS], dtype=float)
    candidates = np.asarray([[c[k] for k in PARAMETER_KEYS] for c in protocol["candidates"]], dtype=float)
    if candidates.shape != (6, 4) or not np.isfinite(candidates).all():
        raise ValueError("Expected six finite controller profiles")
    return_normalization = float(protocol["environment"]["horizon"])
    if not np.isfinite(return_normalization) or return_normalization <= 0:
        raise ValueError("Planned horizon must be a finite positive return normalization")
    features = (candidates - incumbent) / FEATURE_SCALES
    rows, block_sizes = [], set()
    for root in calroots:
        hi, proxy, bank = _inputs(root, h)
        labels = _labels(root, hi)
        block_sizes.add(bank.shape[1])
        rows.extend({"features": features[j].tolist(),
                     "delta_proxy": float(proxy[j]) / return_normalization,
                     "delta_true": float(labels[j]) / return_normalization} for j in range(6))
    if block_sizes != {int(protocol["selection_episodes"])}:
        raise ValueError("Calibration query block sizes differ from protocol")
    post = e5c._fit_posterior(rows)
    return {
        "schema": "metadrive_author_E5C_adapter_v1", "adaptation": int(h),
        "calibration_roots": [_id(r) for r in calroots], "features": features.tolist(),
        "feature_keys": list(PARAMETER_KEYS), "feature_scales": FEATURE_SCALES.tolist(),
        "feature_rule": "(candidate-incumbent)/fixed_scale; incumbent footprint exactly zero",
        "return_normalization": return_normalization,
        "return_normalization_rule": "native cumulative return divided by frozen planned horizon, not realized steps or fitted outcome scale",
        "selection_episodes_per_arm": int(protocol["selection_episodes"]),
        "training_episodes_per_profile": int(protocol["response"]["training_episodes_per_profile"]),
        "statistics": dict(AUTHOR_STATISTICS),
        "posterior": {"prior_precision": float(post.prior_precision),
                      "noise_variance": float(post.noise_variance),
                      "mean": post.mean.tolist(), "covariance": post.covariance.tolist(),
                      "n_observations": int(post.n_observations)},
        "calibration_rows_sha256": _digest(rows), "protocol_sha256_canonical": _digest(protocol),
        "provenance": _provenance(),
        "scope": "unchanged E5C functions, new MetaDrive domain footprints, no-update added at final choice only",
        "noise_note": "unchanged author E5C max(population_variance(normalized_calibration_corrections),1e-4); variance is in squared native-return/planned-horizon units",
    }


def decide_author(root: Mapping, fit: dict, budgets=(1, 2, 4)) -> list[dict]:
    """Author choices see query observations, never confirmation audit labels."""
    _verify_provenance(fit)
    e5c, cls = _author()
    rid, h = _id(root), int(fit["adaptation"])
    if rid in fit["calibration_roots"]:
        raise ValueError("Decision root overlaps calibration roots")
    _, proxy, bank = _inputs(root, h)
    return_normalization = float(fit["return_normalization"])
    if not np.isfinite(return_normalization) or return_normalization <= 0:
        raise ValueError("Invalid frozen return normalization")
    proxy = proxy / return_normalization
    bank = bank / return_normalization
    s = int(fit["selection_episodes_per_arm"])
    if bank.shape[1] != s:
        raise ValueError("Selection block size differs from frozen author fit")
    budgets = tuple(int(b) for b in budgets)
    if not budgets or len(set(budgets)) != len(budgets) or any(not 0 <= b <= 6 for b in budgets):
        raise ValueError("Invalid or duplicate query budgets")
    post = cls(**fit["posterior"])
    # The author constructor accepts arrays but does not coerce the JSON lists.
    post.mean, post.covariance = np.asarray(post.mean), np.asarray(post.covariance)
    setup = h * int(fit["training_episodes_per_profile"])
    query_cost = setup + 2 * s
    rows = [{"transition_id": str(j + 1), "features": fit["features"][j],
             "delta_proxy": float(proxy[j]), "hf_query_cost": query_cost} for j in range(6)]
    incumbent = {"transition_id": "incumbent", "features": [0.0] * 4,
                 "delta_proxy": 0.0, "hf_query_cost": 1.0, "delta_true": 0.0}
    decisions = []
    for budget in budgets:
        for name, original in METHOD_MAP.items():
            queried = e5c._select(original, rows, post, budget, rid,
                                  {"statistics": fit["statistics"]})
            if len(queried) != budget or len(set(queried)) != budget or not set(queried) <= {r["transition_id"] for r in rows}:
                raise AssertionError("Author returned invalid query allocation")
            observed = {j: float(bank[int(j) - 1].mean()) for j in queried}
            # These NaNs are placeholders, never actual unqueried deployment
            # labels. _select_outcome's unused second return may itself be NaN.
            outcome = [dict(row, delta_true=observed.get(row["transition_id"], float("nan"))) for row in rows]
            selected, _unused_selected_label = e5c._select_outcome(outcome + [incumbent], queried, original, post)
            selected_idx = 0 if selected == "incumbent" else int(selected)
            estimates = [0.0]
            for row in rows:
                j = row["transition_id"]
                estimates.append(observed[j] if j in observed else
                                 float(row["delta_proxy"]) if original == "random_hf" else
                                 float(e5c._posterior_for(row, post)[0]))
            if estimates[selected_idx] != max(estimates):
                raise AssertionError("Author outcome disagrees with documented estimates")
            decisions.append({
                "root": rid, "adaptation": h, "method": name, "original_author_method": original,
                "budget_queries": budget, "selection_rng_seed": rid, "selected_idx": selected_idx,
                "queries": [int(j) for j in queried], "hf_queries": len(queried),
                "hf_episode_cost": setup + budget * query_cost if budget else 0,
                "hf_episode_budget": setup + budget * query_cost,
                "incumbent_setup_episode_cost": setup if budget else 0,
                "per_query_episode_cost": query_cost, "stop_reason": "author_E5C_fixed_batch",
                "estimates_incumbent_first": [v * return_normalization for v in estimates],
                "observed_query_means": {j: v * return_normalization for j, v in observed.items()},
                "return_normalization": return_normalization,
                "estimate_and_observation_units": "native cumulative return; author model alone uses return/planned_horizon",
                "unqueried_observations_exposed": False,
                "author_outcome_rule": "queried=observed_mean; unqueried=author model or proxy for Random; incumbent=0",
            })
    return decisions


def score_author(root: Mapping, fit: dict, decisions: list[dict], decision_sha256: str) -> dict:
    """Score only after verifying all author decisions have been sealed."""
    if _digest(decisions) != decision_sha256:
        raise ValueError("Author decision seal mismatch before audit access")
    h = int(fit["adaptation"])
    hi = list(root.get("adaptation_steps", [4, 12])).index(h)
    label = np.concatenate([[0.0], _labels(root, hi)])
    oracle = float(label.max())
    scored = []
    for decision in decisions:
        if decision["root"] != _id(root) or decision["adaptation"] != h:
            raise ValueError("Author decision root/adaptation mismatch")
        gain = float(label[decision["selected_idx"]])
        scored.append({**decision, "selected_audit_gain": gain, "empirical_oracle_regret": oracle - gain})
    return {
        "root": _id(root), "adaptation": h, "selected_idx_convention": "0=incumbent; 1..6=candidates",
        "decisions": decisions, "decision_sha256": decision_sha256, "scored": scored,
        "audit_gain_estimates_incumbent_first": label.tolist(), "empirical_oracle_gain": oracle,
        "author_fit_sha256": _digest(fit), "author_provenance": fit["provenance"],
        "audit_label_source": "independent audit A/B mean; not exact deployment values",
    }


def run_author(root: Mapping, fit: dict, budgets=(1, 2, 4)) -> dict:
    decisions = decide_author(root, fit, budgets)
    seal = _digest(decisions)
    return score_author(root, fit, decisions, seal)


def self_test(protocol: Mapping) -> dict:
    rng = np.random.default_rng(80421)
    def world(seed):
        return {"seed": seed, "proxy": rng.normal(size=6).tolist(),
                "selection": rng.normal(size=(2, 6, 4)).tolist(),
                "audit_a": rng.normal(size=(2, 6, 4)).tolist(),
                "audit_b": rng.normal(size=(2, 6, 4)).tolist()}
    calroots = [world(1), world(2), world(3)]
    fit = fit_author(calroots, 4, protocol)
    root = world(99)
    class NoAudit(dict):
        def __getitem__(self, key):
            if key in ("audit_a", "audit_b"):
                raise AssertionError("Author decision touched audit labels")
            return super().__getitem__(key)
    decisions = decide_author(NoAudit(root), fit)
    for d in decisions:
        assert d["hf_queries"] == d["budget_queries"] == len(set(d["queries"]))
        assert d["hf_episode_cost"] == 8 + 16 * d["hf_queries"]
    one = run_author(root, fit)
    two = run_author(dict(root, audit_a=np.full((2, 6, 4), 999).tolist()), fit)
    assert one["decision_sha256"] == two["decision_sha256"]
    assert one["audit_gain_estimates_incumbent_first"] != two["audit_gain_estimates_incumbent_first"]
    try:
        decide_author(dict(root, seed=1), fit)
    except ValueError as exc:
        assert "overlaps" in str(exc)
    else:
        raise AssertionError("Calibration/decision overlap accepted")
    try:
        score_author(NoAudit(root), fit, decisions, "bad")
    except ValueError as exc:
        assert "seal mismatch" in str(exc)
    else:
        raise AssertionError("Bad decision seal accepted")
    np.testing.assert_array_equal(np.zeros(4), (np.asarray([protocol["incumbent"][k] for k in PARAMETER_KEYS]) -
                                              np.asarray([protocol["incumbent"][k] for k in PARAMETER_KEYS])) / FEATURE_SCALES)
    # A manually normalized dataset with unit normalization must induce exactly
    # the same author posterior and choices, and proportional native outputs.
    scale = fit["return_normalization"]
    def normalize(r):
        return {**r, **{k: (np.asarray(r[k]) / scale).tolist()
                       for k in ("proxy", "selection", "audit_a", "audit_b")}}
    reference_protocol = {**protocol, "environment": {**protocol["environment"], "horizon": 1}}
    reference_fit = fit_author([normalize(r) for r in calroots], 4, reference_protocol)
    for key in ("mean", "covariance", "noise_variance", "prior_precision"):
        np.testing.assert_allclose(fit["posterior"][key], reference_fit["posterior"][key], rtol=1e-12, atol=1e-14)
    reference = run_author(normalize(root), reference_fit)
    for native, normalized in zip(one["scored"], reference["scored"]):
        assert native["queries"] == normalized["queries"]
        assert native["selected_idx"] == normalized["selected_idx"]
        np.testing.assert_allclose(native["estimates_incumbent_first"],
                                   np.asarray(normalized["estimates_incumbent_first"]) * scale,
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(native["selected_audit_gain"],
                                   normalized["selected_audit_gain"] * scale,
                                   rtol=1e-12, atol=1e-12)
    return {"status": "OK", "author_methods": list(METHOD_MAP),
            "checks": ["actual_author_imports", "audit_firewall", "audit_mutation_choice_invariance",
                       "unique_full_query_budget", "setup_and_response_costs", "calibration_separation",
                       "seal_before_scoring", "zero_incumbent_footprint",
                       "fixed_horizon_normalization_matches_explicit_normalized_dataset"],
            "provenance": fit["provenance"]}
