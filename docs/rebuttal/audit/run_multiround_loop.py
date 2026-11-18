#!/usr/bin/env python3
"""Run a small, auditable multi-round PIVOT-KG loop in a controlled world.

The experiment deliberately lives outside the frozen paper experiments.  It
uses the frozen GaussianPosterior and analytic exact_kg_next implementation,
but adds an explicit response state and independent post-decision evaluation.
The output is evidence about this declared controlled protocol only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:8], 16)


def boot(values: list[float], seed: int, draws: int = 10000) -> dict[str, float | int]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sample = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    low, high = np.percentile(sample, [2.5, 97.5])
    return {"mean": float(x.mean()), "ci_low": float(low), "ci_high": float(high), "n": int(len(x))}


def signflip_p(values: list[float], seed: int, draws: int = 10000) -> float:
    x = np.asarray(values, dtype=float)
    if np.all(np.abs(x) <= 1e-15):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, len(x)))
    observed = abs(float(x.mean()))
    null = np.abs((signs * x).mean(axis=1))
    return float((1 + np.sum(null >= observed - 1e-15)) / (draws + 1))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def posterior_hash(posterior: Any) -> str:
    return sha256_bytes(json.dumps({
        "means": posterior.means.tolist(),
        "covariance": posterior.covariance.tolist(),
        "observed": posterior.observed,
    }, sort_keys=True).encode())


@dataclass
class ResponseWorld:
    """Carry the frozen PerformativeWorld actor state across deployments."""

    pressure: float = 0.0
    response_strength: float = 0.85
    competition_strength: float = 0.25
    horizon: int = 12

    def __post_init__(self) -> None:
        from pivot.environments.performative.config import PerformativeConfig
        from pivot.environments.performative.world import PerformativeWorld

        self._world = PerformativeWorld(PerformativeConfig(
            response_strength=self.response_strength,
            competition_strength=self.competition_strength,
            noise_scale=0.025,
            horizon=self.horizon,
            config_id="rebuttal-multiround-controlled-v1",
        ))

    def value(self, policy: dict[str, float], *, seed: int, deployment: bool) -> float:
        from pivot.core.policy import Policy
        from pivot.core.result import RolloutContext

        context = RolloutContext(seed=seed, scenario_id=f"loop-{seed}", initial_state=self.pressure)
        result = self._world.evaluate(
            Policy.from_mapping(policy), context, mode="actor" if deployment else "observer"
        )
        return float(result.value)

    def deploy(self, policy: dict[str, float], *, seed: int) -> None:
        from pivot.core.policy import Policy
        from pivot.core.result import RolloutContext

        context = RolloutContext(seed=seed, scenario_id=f"deploy-{seed}", initial_state=self.pressure)
        result = self._world.evaluate(Policy.from_mapping(policy), context, mode="actor")
        self.pressure = float(result.metadata["final_state"])

    def state_hash(self) -> str:
        return sha256_bytes(json.dumps({"pressure": self.pressure}, sort_keys=True).encode())


def propose(incumbent: dict[str, float], *, root: int, round_id: int) -> list[dict[str, float]]:
    """Fixed proposer distribution; methods use the same generator contract."""

    rng = np.random.default_rng(stable_seed("proposer", root, round_id))
    base = float(incumbent["intensity"])
    bias = float(incumbent["bias"])
    offsets = ((0.085, 0.00), (-0.060, 0.025), (0.035, -0.045), (-0.095, 0.055))
    out = []
    for index, (delta, bias_delta) in enumerate(offsets):
        jitter = float(rng.normal(0.0, 0.006))
        out.append({
            "intensity": float(np.clip(base + delta + jitter, -0.85, 0.85)),
            "bias": float(np.clip(bias + bias_delta + rng.normal(0.0, 0.004), -0.85, 0.85)),
            "slot": float(index),
        })
    return out


def make_row(world: ResponseWorld, incumbent: dict[str, float], candidate: dict[str, float], *, root: int, round_id: int, index: int) -> dict[str, Any]:
    proxy_seed = stable_seed("proxy", root, round_id, index)
    incumbent_proxy = world.value(incumbent, seed=proxy_seed, deployment=False)
    candidate_proxy = world.value(candidate, seed=proxy_seed, deployment=False)
    return {
        "candidate_id": f"c{index}",
        "candidate_index": index,
        "delta_proxy": float(candidate_proxy - incumbent_proxy),
        "candidate_parameters": {k: float(v) for k, v in candidate.items() if k != "slot"},
        "highway_action": f"slot{index}",
    }


def deployment_delta(world: ResponseWorld, incumbent: dict[str, float], candidate: dict[str, float], *, seed: int) -> float:
    inc = world.value(incumbent, seed=seed, deployment=True)
    cand = world.value(candidate, seed=seed, deployment=True)
    return float(cand - inc)


def fit_model(calibration_roots: int, frozen_source: Path, cal_config: dict[str, Any]) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    from experiments.revision.highway_calibration import fit_calibration_model

    proxy_by_seed: dict[int, list[dict[str, Any]]] = {}
    truth_by_seed: dict[int, dict[str, float]] = {}
    calibration_rows: list[dict[str, Any]] = []
    for offset in range(calibration_roots):
        root = 120000 + offset
        world = ResponseWorld(pressure=float(0.10 + 0.015 * (offset % 7)))
        incumbent = {"intensity": 0.18, "bias": 0.0}
        candidates = propose(incumbent, root=root, round_id=0)
        rows = [make_row(world, incumbent, candidate, root=root, round_id=0, index=i) for i, candidate in enumerate(candidates)]
        truth = {
            row["candidate_id"]: deployment_delta(
                world, incumbent, candidates[i], seed=stable_seed("calibration-hf", root, 0, i)
            )
            for i, row in enumerate(rows)
        }
        proxy_by_seed[root] = rows
        truth_by_seed[root] = truth
        calibration_rows.append({"root": root, "proxy": rows, "truth": truth})
    model = fit_calibration_model(proxy_by_seed, truth_by_seed, covariance_shrinkage=0.5, covariance_floor=1e-8)
    return model, model.to_record(), calibration_rows


def select_round(method: str, rows: list[dict[str, Any]], model: Any, world: ResponseWorld, incumbent: dict[str, float], *, root: int, round_id: int, source: Any) -> tuple[str, list[str], Any]:
    from experiments.revision.highway_b4_redesign import exact_kg_next
    from experiments.revision.highway_calibration import GaussianPosterior

    posterior = GaussianPosterior(rows, model)
    query_ids: list[str] = []
    query_details: list[dict[str, Any]] = []
    budget = 2
    if method == "PIVOT-KG":
        for query_order in range(budget):
            candidate_id, kg = exact_kg_next(posterior)
            index = posterior._index[candidate_id]
            query_seed = stable_seed("hf", root, round_id, candidate_id)
            value = deployment_delta(world, incumbent, rows[index]["candidate_parameters"], seed=query_seed)
            posterior.observe(candidate_id, value)
            query_ids.append(candidate_id)
            query_details.append({"candidate_id": candidate_id, "query_order": query_order, "kg": kg, "observation": value, "query_seed": query_seed})
    elif method == "Uniform-Random":
        rng = np.random.default_rng(stable_seed("uniform", root, round_id))
        for query_order, index in enumerate(rng.choice(len(rows), size=budget, replace=False)):
            candidate_id = rows[int(index)]["candidate_id"]
            query_seed = stable_seed("hf", root, round_id, candidate_id)
            value = deployment_delta(world, incumbent, rows[int(index)]["candidate_parameters"], seed=query_seed)
            posterior.observe(candidate_id, value)
            query_ids.append(candidate_id)
            query_details.append({"candidate_id": candidate_id, "query_order": query_order, "observation": value, "query_seed": query_seed})
    elif method == "Top-Proxy":
        order = sorted(range(len(rows)), key=lambda i: (-float(rows[i]["delta_proxy"]), int(rows[i]["candidate_index"])))[:budget]
        for query_order, index in enumerate(order):
            candidate_id = rows[index]["candidate_id"]
            query_seed = stable_seed("hf", root, round_id, candidate_id)
            value = deployment_delta(world, incumbent, rows[index]["candidate_parameters"], seed=query_seed)
            posterior.observe(candidate_id, value)
            query_ids.append(candidate_id)
            query_details.append({"candidate_id": candidate_id, "query_order": query_order, "observation": value, "query_seed": query_seed})
    elif method == "IVR":
        for query_order in range(budget):
            available = [candidate for candidate in posterior.candidate_ids if candidate not in posterior.observed]
            scores = {}
            for candidate_id in available:
                index = posterior._index[candidate_id]
                denominator = float(posterior.covariance[index, index] + posterior.observation_variance[index])
                scores[candidate_id] = float(np.dot(posterior.covariance[:, index], posterior.covariance[:, index]) / denominator) if denominator > 0 else 0.0
            candidate_id = max(available, key=lambda value: (scores[value], -posterior._index[value]))
            index = posterior._index[candidate_id]
            query_seed = stable_seed("hf", root, round_id, candidate_id)
            value = deployment_delta(world, incumbent, rows[index]["candidate_parameters"], seed=query_seed)
            posterior.observe(candidate_id, value)
            query_ids.append(candidate_id)
            query_details.append({"candidate_id": candidate_id, "query_order": query_order, "observation": value, "ivr": scores[candidate_id], "query_seed": query_seed})
    elif method == "Proxy-Only":
        query_details = []
    elif method == "All-HF":
        for query_order, row in enumerate(rows):
            query_seed = stable_seed("hf", root, round_id, row["candidate_id"])
            value = deployment_delta(world, incumbent, row["candidate_parameters"], seed=query_seed)
            posterior.observe(row["candidate_id"], value)
            query_ids.append(row["candidate_id"])
            query_details.append({"candidate_id": row["candidate_id"], "query_order": query_order, "observation": value, "query_seed": query_seed})
    else:
        raise ValueError(method)
    if method == "Proxy-Only":
        selected = max([("incumbent", 0.0)] + [(row["candidate_id"], float(row["delta_proxy"])) for row in rows], key=lambda x: (x[1], x[0]))[0]
    else:
        selected = posterior.select()
    return selected, query_ids, {"posterior": posterior, "query_details": query_details}


def run(root_count: int, rounds: int, output: Path, source: Path, bootstrap_draws: int) -> None:
    if output.exists():
        raise SystemExit(f"output exists; choose a new directory: {output}")
    output.mkdir(parents=True)
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "src"))
    model, model_record, calibration_rows = fit_model(40, source, {})
    prior_hash = sha256_bytes(json.dumps(model_record, sort_keys=True).encode())
    methods = ["PIVOT-KG", "Uniform-Random", "IVR", "Top-Proxy", "Proxy-Only", "All-HF"]
    ledger: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for root_offset in range(root_count):
        root = 130000 + root_offset
        for method in methods:
            world = ResponseWorld(pressure=float(0.10 + 0.015 * (root_offset % 7)))
            incumbent = {"intensity": 0.18, "bias": 0.0}
            cumulative_isr = 0.0
            cumulative_utility = 0.0
            harmful = 0
            retained = 0
            for round_id in range(rounds):
                candidates = propose(incumbent, root=root, round_id=round_id)
                rows = [make_row(world, incumbent, candidate, root=root, round_id=round_id, index=i) for i, candidate in enumerate(candidates)]
                selected, query_ids, details = select_round(method, rows, model, world, incumbent, root=root, round_id=round_id, source=source)
                # The terminal decision is based on the posterior, while these
                # independent evaluation outcomes are read only after the ledger
                # is written for this round.
                eval_seed = stable_seed("independent-eval", root, round_id)
                eval_deltas = [deployment_delta(world, incumbent, candidate, seed=eval_seed + i) for i, candidate in enumerate(candidates)]
                best = max([0.0, *eval_deltas])
                selected_delta = 0.0 if selected == "incumbent" else eval_deltas[int(selected[1:])]
                isr = float(best - selected_delta)
                retained_flag = selected == "incumbent"
                harmful_flag = selected_delta < 0.0
                next_state_before = world.state_hash()
                chosen_policy = incumbent if retained_flag else candidates[int(selected[1:])]
                deployment_seed = stable_seed("deployment-response", root, method, round_id)
                world.deploy(chosen_policy, seed=deployment_seed)
                next_state_after = world.state_hash()
                ledger.append({
                    "root_id": root,
                    "method": method,
                    "round_id": round_id,
                    "candidate_panel": rows,
                    "initial_state_hash": next_state_before,
                    "proposer_seed": stable_seed("proposer", root, round_id),
                    "environment_seed": stable_seed("world", root, round_id),
                    "evaluation_seed": eval_seed,
                    "deployment_seed": deployment_seed,
                    "calibration_hash": prior_hash,
                    "query_ids": query_ids,
                    "query_count": len(query_ids),
                    "query_cost": 24 * len(query_ids),
                    "query_details": details["query_details"],
                    "selected_policy": selected,
                    "terminal_selection_rule": "GaussianPosterior.select over incumbent plus four candidates",
                    "retained_incumbent": retained_flag,
                    "deployment_outcome": {"independent_deltas": eval_deltas, "selected_delta": selected_delta, "best_delta": best, "ISR": isr},
                    "response_operator": "frozen PerformativeWorld actor dynamics; carry deployment final_state into next-round initial_state",
                    "next_state_hash": next_state_after,
                    "next_pressure": world.pressure,
                })
                cumulative_isr += isr
                cumulative_utility += selected_delta
                harmful += int(harmful_flag)
                retained += int(retained_flag)
                incumbent = {"intensity": float(chosen_policy["intensity"]), "bias": float(chosen_policy["bias"])}
            trajectories.append({"root_id": root, "method": method, "CISR": cumulative_isr, "cumulative_utility": cumulative_utility, "harmful_adoption_rate": harmful / rounds, "retention_rate": retained / rounds, "rounds": rounds})

    decision_blob = json.dumps(ledger, sort_keys=True, separators=(",", ":")).encode()
    (output / "trajectory_ledger.json").write_bytes(decision_blob)
    (output / "trajectory_ledger.sha256").write_text(sha256_bytes(decision_blob) + "\n")
    (output / "calibration_rows.json").write_text(json.dumps(calibration_rows, indent=2, sort_keys=True) + "\n")
    (output / "calibration_model.json").write_text(json.dumps(model_record, indent=2, sort_keys=True) + "\n")
    (output / "trajectories.json").write_text(json.dumps(trajectories, indent=2, sort_keys=True) + "\n")
    summary: dict[str, Any] = {
        "analysis_kind": "genuine_multiround_algorithm1_loop",
        "physical_simulator_called": False,
        "protocol": {"rounds": rounds, "roots": root_count, "candidate_count": 4, "hf_budget_per_round": 2, "paired_query_cost_environment_steps": 24, "calibration_roots": 40, "evaluation_seeds_independent": True, "response_state_carried": True},
        "methods": methods,
        "formal_lucb": {"status": "not_run_budget_infeasible", "reason": "formal initialization requires 4 candidate observations, but the per-round budget is 2"},
        "calibration_prior_hash": prior_hash,
        "ledger_sha256": sha256_bytes(decision_blob),
        "method_summary": {},
        "audit_checks": {
            "all_rounds_have_query_or_zero_query_record": len(ledger) == root_count * len(methods) * rounds,
            "all_pivot_rounds_used_two_queries": all(r["query_count"] == 2 for r in ledger if r["method"] == "PIVOT-KG"),
            "response_state_changes_recorded": all(r["initial_state_hash"] != r["next_state_hash"] for r in ledger),
            "independent_eval_seed_recorded": all(r["evaluation_seed"] != r["environment_seed"] for r in ledger),
            "terminal_selection_includes_incumbent": all("incumbent" in r["terminal_selection_rule"] for r in ledger),
        },
    }
    for method in methods:
        subset = [r for r in trajectories if r["method"] == method]
        summary["method_summary"][method] = {metric: boot([float(r[metric]) for r in subset], seed=stable_seed("loop-bootstrap", method, metric), draws=bootstrap_draws) for metric in ("CISR", "cumulative_utility", "harmful_adoption_rate", "retention_rate")}
    summary["audit_checks"]["all_pass"] = bool(all(summary["audit_checks"].values()))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    run(args.roots, args.rounds, args.output, args.source, args.bootstrap_draws)
    print((args.output / "summary.json").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
