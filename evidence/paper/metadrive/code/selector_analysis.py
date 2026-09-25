"""MetaDrive fixed-block selectors, independent-audit scoring, and root-level CIs.

Reuses the unchanged pivot_v2 posterior and four acquisition rules.  The separately
named global_voi_heuristic is a matched-posterior adaptation of the author's E5C
std/(1+abs(proxy)) score, NOT the author's original full implementation or IVR.

Input: root/seed, proxy (K,), selection (2,K,S), audit_a/audit_b (2,K,A), and
optional adaptation_steps (default [4,12]).  External selected_idx=0 is the
incumbent; candidates have selected_idx=1..K.  This deliberately translates
pivot_v2's internal convention (candidate 0..K-1; incumbent K).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import numpy as np

import pivot_v2
from pivot_v2 import PosteriorV2Spec, fit_posterior_v2, make_posterior, run_selector

ADAPTATION_STEPS = (4, 12)
METHODS = ("pivot_kg", "uniform_v2", "ivr_v2", "lucb_v2", "no_hf_v2",
           "proxy_only", "global_voi_heuristic")
UNIFORM_REPETITIONS = 100
NOISE_FLOOR = 1e-8
BOOTSTRAP_DRAWS = 10000


def _root_id(root: Mapping) -> int:
    if "root" in root:
        return int(root["root"])
    if "seed" in root:
        return int(root["seed"])
    raise ValueError("root requires root or seed identifier")


def _adaptation_index(root: Mapping, adaptation: int) -> int:
    values = root.get("adaptation_steps", ADAPTATION_STEPS)
    if isinstance(values, Mapping):
        values = values.values()
    values = tuple(int(v) for v in values)
    if values != ADAPTATION_STEPS:
        raise ValueError(f"adaptation_steps must be {ADAPTATION_STEPS}, got {values}")
    return values.index(int(adaptation))


def _array(root: Mapping, key: str, dimensions: int) -> np.ndarray:
    value = np.asarray(root[key], dtype=float)
    if value.ndim != dimensions or not np.all(np.isfinite(value)):
        raise ValueError(f"{key} must be a finite {dimensions}-dimensional array")
    return value


def _selection_inputs(root: Mapping, adaptation: int):
    hi = _adaptation_index(root, adaptation)
    proxy = _array(root, "proxy", 1)
    bank = _array(root, "selection", 3)
    if proxy.shape != (6,) or bank.shape[:2] != (2, 6) or bank.shape[2] < 2:
        raise ValueError("expected proxy (6,) and selection (2,6,S>=2)")
    return proxy.copy(), bank[hi].copy()


def _audit_inputs(root: Mapping, adaptation: int, k: int):
    hi = _adaptation_index(root, adaptation)
    a, b = _array(root, "audit_a", 3), _array(root, "audit_b", 3)
    if a.shape != b.shape or a.shape[:2] != (2, k) or a.shape[2] < 2:
        raise ValueError("audit_a/audit_b must share shape (2,K,A>=2)")
    return a[hi].mean(axis=1), b[hi].mean(axis=1)


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(obj) -> str:
    return hashlib.sha256(_canonical(obj)).hexdigest()


def fit_model(calibration_roots: list[Mapping], adaptation: int) -> dict:
    """Fit exclusively on calibration roots; return fully serializable provenance.

    Covariance follows pivot_v2: correction sample covariance minus diagonal
    E[(A-B)^2]/4, shrinkage, PSD projection, then mean-estimation uncertainty.
    Observation R uses sample variance of paired selection episodes / S, averaged
    across calibration roots.  The noise-estimation input to the unchanged fit
    function contains that prescribed estimator, not the older cross-product.
    """
    if len(calibration_roots) < 2:
        raise ValueError("at least two independent calibration roots required")
    ids = [_root_id(w) for w in calibration_roots]
    if len(set(ids)) != len(ids):
        raise ValueError("calibration root identifiers must be unique")
    g, ab, block_variance, bank_sizes = [], [], [], []
    for root in calibration_roots:
        proxy, bank = _selection_inputs(root, adaptation)
        a, b = _audit_inputs(root, adaptation, len(proxy))
        g.append((a + b) / 2.0 - proxy)
        ab.append(a - b)
        block_variance.append(bank.var(axis=1, ddof=1) / bank.shape[1])
        bank_sizes.append(bank.shape[1])
    if len(set(bank_sizes)) != 1:
        raise ValueError("calibration query block sizes differ")
    spec = fit_posterior_v2(np.asarray(g), np.asarray(ab), np.asarray(block_variance),
                            r_smoothing="none", distances=None, r_floor=NOISE_FLOOR)
    spec.notes["R_estimator"] = "mean_calibration(sample_var(paired_selection_episodes,ddof=1)/S)"
    spec.notes["R_floor"] = NOISE_FLOOR
    spec.notes["audit_label_source"] = "mean_of_independent_noisy_audit_A_B"
    source = Path(pivot_v2.__file__).resolve()
    return {
        "schema_version": 1, "adaptation": int(adaptation), "n_candidates": 6,
        "selection_episodes_per_arm": bank_sizes[0], "calibration_roots": ids,
        "spec": asdict(spec), "calibration_arrays_sha256": _digest({
            "root_ids": ids, "g": np.asarray(g).tolist(), "ab": np.asarray(ab).tolist(),
            "block_variance": np.asarray(block_variance).tolist()}),
        "pivot_v2_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "calibration_provenance": {
            "training_only": True, "n_independent_roots": len(ids),
            "observation_noise": spec.notes["R_estimator"], "noise_floor": NOISE_FLOOR,
            "audit_noise_subtraction": "diagonal mean((audit_A_mean-audit_B_mean)^2)/4",
            "covariance_limitation": "off-diagonal audit measurement noise is not subtracted",
        },
    }


def _post(fit: dict, proxy: np.ndarray):
    return make_posterior(PosteriorV2Spec(**fit["spec"]), proxy)


def _query_callback(means: np.ndarray):
    queried = []

    def query(index: int) -> float:
        i = int(index)
        if not 0 <= i < len(means) or i in queried:
            raise ValueError("query must address a previously unqueried candidate")
        queried.append(i)
        return float(means[i])

    return query, queried


def _global_heuristic(post, proxy, query, budget: int, seed: int) -> dict:
    """Sequential matched-posterior E5C-score heuristic; not an author-code run."""
    rng = np.random.default_rng(seed)
    queried, steps = [], []
    for iteration in range(min(budget, post.size)):
        available = [i for i in range(post.size) if i not in queried]
        sd = np.sqrt(np.maximum(np.diag(post.covariance) + post.observation_variance, 0))
        scores = sd / (1.0 + np.abs(proxy))
        j = max(available, key=lambda i: (scores[i], rng.random()))
        observed = query(j)
        post = post.condition(j, observed)
        queried.append(j)
        steps.append({"it": iteration, "query": j, "observed": observed,
                      "score": {str(i): float(scores[i]) for i in available}})
    return {"selected": post.best(), "queried": queried, "hf_queries": len(queried),
            "stop_reason": "budget_exhausted", "estimates": post.means_with_incumbent().tolist(),
            "steps": steps}


def _external_choice(internal: int, k: int) -> int:
    return 0 if internal == k else int(internal) + 1


def _one_decision(proxy, means, fit, method: str, budget: int, seed: int) -> dict:
    post = _post(fit, proxy)
    query, queried = _query_callback(means)
    h, s = int(fit["adaptation"]), int(fit["selection_episodes_per_arm"])
    query_cost, setup_cost = 2 * h + 2 * s, 2 * h
    if method == "global_voi_heuristic":
        result = _global_heuristic(post, proxy, query, budget, seed)
    else:
        result = run_selector(post, proxy, query, method=method, budget=budget,
                              costs=np.full(len(proxy), query_cost, dtype=float), seed=seed)
    q = len(queried)
    if q > budget or q != len(set(queried)):
        raise AssertionError("query budget or without-replacement constraint violated")
    if list(result["queried"]) != queried:
        raise AssertionError("selector query trace disagrees with callback")
    return {
        "method": method, "budget_queries": int(budget), "selection_rng_seed": int(seed),
        "selected_idx": _external_choice(int(result["selected"]), len(proxy)),
        "queries": [i + 1 for i in queried], "hf_queries": q,
        "hf_episode_cost": setup_cost + q * query_cost if q else 0,
        "hf_episode_budget": setup_cost + budget * query_cost,
        "incumbent_setup_episode_cost": setup_cost if q else 0,
        "per_query_episode_cost": query_cost,
        "stop_reason": result["stop_reason"], "estimates_incumbent_first":
        [float(result["estimates"][-1])] + [float(x) for x in result["estimates"][:-1]],
        "internal_trace_index_convention": "candidate=0..K-1; incumbent=K",
        "steps": result["steps"],
    }


def decide_root(root: Mapping, fit: dict, budgets=(1, 2, 4)) -> list[dict]:
    """Access only ID, proxy, and selection bank; never access either audit key."""
    rid, h = _root_id(root), int(fit["adaptation"])
    if rid in fit["calibration_roots"]:
        raise ValueError("held-out root overlaps calibration training roots")
    proxy, bank = _selection_inputs(root, h)
    if bank.shape != (fit["n_candidates"], fit["selection_episodes_per_arm"]):
        raise ValueError("selection bank differs from frozen calibration dimensions")
    budgets = [int(b) for b in budgets]
    if not budgets or len(set(budgets)) != len(budgets) or any(b < 0 or b > len(proxy) for b in budgets):
        raise ValueError("budgets must be unique integer query counts between 0 and K")
    means, decisions = bank.mean(axis=1), []
    for budget in budgets:
        for method in METHODS:
            record = _one_decision(proxy, means, fit, method, budget, rid)
            decisions.append({"root": rid, "adaptation": h, **record})
        draws = [_one_decision(proxy, means, fit, "uniform_v2", budget,
                               rid * 1000 + draw + 1) for draw in range(UNIFORM_REPETITIONS)]
        decisions.append({
            "root": rid, "adaptation": h, "method": "uniform_v2_expected_100",
            "budget_queries": budget, "selected_idx": None,
            "selected_distribution": [d["selected_idx"] for d in draws],
            "queries_by_draw": [d["queries"] for d in draws],
            "allocation_rng_seeds": [d["selection_rng_seed"] for d in draws],
            "hf_queries": budget, "hf_episode_cost": draws[0]["hf_episode_cost"],
            "hf_episode_budget": draws[0]["hf_episode_budget"],
            "n_allocation_repetitions": UNIFORM_REPETITIONS,
            "distribution_kind": "monte_carlo_100_uniform_without_replacement_allocations",
            "cost_note": "per deployment decision; repetitions reuse the sealed bank for baseline integration",
        })
    return decisions


def seal_decisions(decisions: list[dict]) -> str:
    """Audit labels are unavailable until the complete decision list is hashed."""
    return _digest(decisions)


def score_root(root: Mapping, fit: dict, decisions: list[dict], decision_sha256: str) -> dict:
    if seal_decisions(decisions) != decision_sha256:
        raise ValueError("decision seal mismatch before opening audit labels")
    a, b = _audit_inputs(root, int(fit["adaptation"]), int(fit["n_candidates"]))
    label = np.concatenate([[0.0], (a + b) / 2.0])
    oracle = float(label.max())
    scored = []
    for decision in decisions:
        if decision["root"] != _root_id(root) or decision["adaptation"] != fit["adaptation"]:
            raise ValueError("decision root/adaptation mismatch")
        if "selected_distribution" in decision:
            gains = label[np.asarray(decision["selected_distribution"], dtype=int)]
            gain = float(gains.mean())
            extra = {"allocation_mc_se_gain": float(gains.std(ddof=1) / np.sqrt(len(gains)))}
        else:
            gain = float(label[decision["selected_idx"]])
            extra = {}
        scored.append({**decision, "selected_audit_gain": gain,
                       "empirical_oracle_regret": oracle - gain, **extra})
    return {
        "root": _root_id(root), "adaptation": int(fit["adaptation"]),
        "selected_idx_convention": "0=incumbent; 1..K=candidates in proxy array order",
        "decisions": decisions, "decision_sha256": decision_sha256, "scored": scored,
        "audit_gain_estimates_incumbent_first": label.tolist(), "empirical_oracle_gain": oracle,
        "audit_label_source": "mean of independent audit_A and audit_B; not exact values",
        "cost_unit": "simulated episodes; actual environment steps/wallclock recorded by runner",
        "cost_formula": "0 if q=0 else 2*adaptation + q*(2*adaptation + 2*S)",
    }


def run_root(root: Mapping, fit: dict, budgets=(1, 2, 4)) -> dict:
    decisions = decide_root(root, fit, budgets)
    decision_sha256 = seal_decisions(decisions)
    return score_root(root, fit, decisions, decision_sha256)


def bootstrap_mean(values, *, seed=78131, draws=BOOTSTRAP_DRAWS) -> dict:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError("bootstrap requires non-empty finite root-level values")
    rng = np.random.default_rng(seed)
    estimates = x[rng.integers(len(x), size=(draws, len(x)))].mean(axis=1)
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi),
            "n_independent_roots": len(x), "bootstrap_draws": draws, "bootstrap_seed": seed}


def summarize(root_results: list[dict]) -> dict:
    """Paired root-bootstrap CIs; allocation repetitions are not independent roots."""
    groups = {}
    seen = set()
    for result in root_results:
        key = (int(result["root"]), int(result["adaptation"]))
        if key in seen:
            raise ValueError("duplicate root/adaptation in summary")
        seen.add(key)
        for row in result["scored"]:
            group = (int(row["adaptation"]), int(row["budget_queries"]))
            groups.setdefault(group, {}).setdefault(row["method"], {})[row["root"]] = row
    tables, contrasts = [], []
    for (h, budget), methods in sorted(groups.items()):
        uniform = methods["uniform_v2_expected_100"]
        registered = methods["uniform_v2"]
        for method, rows in sorted(methods.items()):
            ids = sorted(rows)
            if set(ids) != set(uniform) or set(ids) != set(registered):
                raise ValueError("unmatched roots across methods")
            tables.append({
                "adaptation": h, "budget_queries": budget, "method": method,
                "selected_audit_gain": bootstrap_mean([rows[i]["selected_audit_gain"] for i in ids]),
                "empirical_oracle_regret": bootstrap_mean([rows[i]["empirical_oracle_regret"] for i in ids]),
                "mean_hf_episode_cost": float(np.mean([rows[i]["hf_episode_cost"] for i in ids])),
                "mean_hf_queries": float(np.mean([rows[i]["hf_queries"] for i in ids])),
            })
            contrasts.append({
                "adaptation": h, "budget_queries": budget, "method": method,
                "gain_minus_uniform_expected_100": bootstrap_mean([
                    rows[i]["selected_audit_gain"] - uniform[i]["selected_audit_gain"] for i in ids]),
                "gain_minus_uniform_registered_draw": bootstrap_mean([
                    rows[i]["selected_audit_gain"] - registered[i]["selected_audit_gain"] for i in ids]),
            })
    return {
        "method_table": tables, "paired_contrasts": contrasts,
        "primary_comparison_direction": "positive gain difference favors named method",
        "uniform_expected_note": "Monte Carlo average of 100 allocations per root; not 100 extra independent roots",
        "oracle_note": "max of noisy independent-audit means, so absolute regret has oracle-estimation bias; paired gain contrasts do not use max",
        "global_voi_heuristic_note": "matched-posterior sequential std/(1+abs(proxy)); not original author Global-VOI execution",
        "covariance_note": "calibration only; diagonal audit-noise correction; root bootstrap does not prove posterior calibration",
    }


def self_test() -> dict:
    """Meaningful smoke tests for the audit firewall, budget, and index translation."""
    rng = np.random.default_rng(33)
    def world(seed):
        return {"seed": seed, "proxy": np.arange(6) / 10,
                "selection": rng.normal(size=(2, 6, 4)),
                "audit_a": rng.normal(size=(2, 6, 4)), "audit_b": rng.normal(size=(2, 6, 4))}
    fit = fit_model([world(1), world(2), world(3)], 4)
    root = world(20)
    class AuditForbidden(dict):
        def __getitem__(self, key):
            if key in ("audit_a", "audit_b"):
                raise AssertionError("decision phase accessed audit")
            return super().__getitem__(key)
        def get(self, key, default=None):
            if key in ("audit_a", "audit_b"):
                raise AssertionError("decision phase accessed audit")
            return super().get(key, default)
    decisions = decide_root(AuditForbidden(root), fit)
    for d in decisions:
        assert d["hf_queries"] <= d["budget_queries"]
        if "queries" in d:
            assert len(d["queries"]) == len(set(d["queries"])) == d["hf_queries"]
        q = d["hf_queries"]
        assert d["hf_episode_cost"] == (8 + q * 16 if q else 0)
    try:
        score_root(AuditForbidden(root), fit, decisions, "bad seal")
    except ValueError as exc:
        assert "seal mismatch" in str(exc)
    else:
        raise AssertionError("tampered seal accepted")
    seal = seal_decisions(decisions)
    one = score_root(root, fit, decisions, seal)
    changed = dict(root, audit_a=np.full((2, 6, 4), 999), audit_b=np.full((2, 6, 4), -100))
    two = run_root(changed, fit)
    assert one["decision_sha256"] == two["decision_sha256"]
    assert one["audit_gain_estimates_incumbent_first"] != two["audit_gain_estimates_incumbent_first"]
    fit_negative = dict(fit, spec=dict(fit["spec"], mu=[0.0] * 6,
                                      K_prior=(np.eye(6) * 1e-8).tolist(), R=[1e-8] * 6))
    negative = dict(root, proxy=np.full(6, -2.0), selection=np.full((2, 6, 4), -2.0))
    for d in decide_root(negative, fit_negative):
        assert d["selected_idx"] == 0 if d["selected_idx"] is not None else set(d["selected_distribution"]) == {0}
    try:
        decide_root(dict(root, seed=1), fit)
    except ValueError as exc:
        assert "overlaps" in str(exc)
    else:
        raise AssertionError("calibration/test overlap accepted")
    summary = summarize([one])
    assert all(r["selected_audit_gain"]["n_independent_roots"] == 1 for r in summary["method_table"])
    return {"passed": ["decision_audit_firewall", "unique_queries_and_full_episode_cost",
                       "seal_checked_before_audit", "audit_mutation_cannot_change_decisions",
                       "incumbent_external_index_zero", "train_test_root_separation",
                       "root_level_bootstrap_count"], "status": "OK"}
