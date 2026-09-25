#!/usr/bin/env python3
"""Post-audit diagnostics for the sealed OpenSpiel formal run.

The script consumes already unblinded results and cannot affect selection.
It diagnoses whether PIVOT used positive EVSI, whether it collapsed to another
acquisition rule, and how each fixed budget changed regret relative to no HF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def stable_seed(*parts: object) -> int:
    data = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def bootstrap(values: list[float], *parts: object, draws: int = 5000) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(stable_seed("method_audit", *parts))
    samples = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "n_roots": int(len(array)),
        "bootstrap_draws": draws,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text())
    calibration = json.loads(args.calibration.read_text())
    scored = audit["scored_decisions"]

    rows_by_key: dict[tuple[int, str, int, str], list[dict]] = defaultdict(list)
    for row in scored:
        rows_by_key[
            (
                int(row["root_seed"]),
                str(row["horizon"]),
                int(row["budget"]),
                str(row["method"]),
            )
        ].append(row)

    def root_regret(root: int, horizon: str, budget: int, method: str) -> float:
        return float(
            np.mean(
                [
                    float(row["selection_regret"])
                    for row in rows_by_key[(root, horizon, budget, method)]
                ]
            )
        )

    roots = list(range(71000, 71100))
    paired_contrasts: dict[str, dict] = {}
    for horizon in ("short", "long"):
        for budget in (64, 128, 256, 512):
            pivot = {root: root_regret(root, horizon, budget, "pivot_cg_v2") for root in roots}
            comparators = {
                "uniform_random": {root: root_regret(root, horizon, budget, "uniform_random") for root in roots},
                "calibrated_no_hf": {root: root_regret(root, horizon, budget, "calibrated_no_hf") for root in roots},
                "global_ivr": {root: root_regret(root, horizon, budget, "global_ivr") for root in roots},
                "posterior_lucb": {root: root_regret(root, horizon, budget, "posterior_lucb") for root in roots},
            }
            for method, values in comparators.items():
                differences = [values[root] - pivot[root] for root in roots]
                paired_contrasts[f"{method}_minus_pivot_{horizon}_budget_{budget}"] = bootstrap(
                    differences, horizon, budget, method
                )

    acquisition_diagnostics: dict[str, dict] = {}
    for horizon in ("short", "long"):
        for budget in (64, 128, 256, 512):
            pivot_rows = {
                int(row["root_seed"]): row
                for row in scored
                if row["horizon"] == horizon
                and int(row["budget"]) == budget
                and row["method"] == "pivot_cg_v2"
            }
            ivr_rows = {
                int(row["root_seed"]): row
                for row in scored
                if row["horizon"] == horizon
                and int(row["budget"]) == budget
                and row["method"] == "global_ivr"
            }
            max_evsi = []
            zero_steps = 0
            total_steps = 0
            query_counts: Counter[str] = Counter()
            for row in pivot_rows.values():
                query_counts.update(row["queried_ids"])
                for trace in row["selector_trace"]:
                    values = [float(value) for value in trace.get("evsi", {}).values()]
                    if not values:
                        continue
                    step_max = max(values)
                    max_evsi.append(step_max)
                    total_steps += 1
                    if step_max <= 1e-15:
                        zero_steps += 1
            same_queries = [
                pivot_rows[root]["queried_ids"] == ivr_rows[root]["queried_ids"] for root in roots
            ]
            same_selection = [
                pivot_rows[root]["selected_id"] == ivr_rows[root]["selected_id"] for root in roots
            ]
            same_regret = [
                abs(float(pivot_rows[root]["selection_regret"]) - float(ivr_rows[root]["selection_regret"]))
                <= 1e-15
                for root in roots
            ]
            acquisition_diagnostics[f"{horizon}_budget_{budget}"] = {
                "total_pivot_query_steps": total_steps,
                "all_evsi_zero_step_count": zero_steps,
                "all_evsi_zero_step_rate": zero_steps / total_steps if total_steps else None,
                "max_evsi_min": float(min(max_evsi)) if max_evsi else None,
                "max_evsi_median": float(np.median(max_evsi)) if max_evsi else None,
                "max_evsi_max": float(max(max_evsi)) if max_evsi else None,
                "pivot_global_ivr_identical_query_path_rate": float(np.mean(same_queries)),
                "pivot_global_ivr_identical_selection_rate": float(np.mean(same_selection)),
                "pivot_global_ivr_identical_regret_rate": float(np.mean(same_regret)),
                "pivot_query_counts": dict(sorted(query_counts.items())),
            }

    selection_distributions: dict[str, dict[str, int]] = {}
    for horizon in ("short", "long"):
        for budget in (0, 64, 128, 256, 512):
            for method in (
                "pivot_cg_v2",
                "uniform_random",
                "global_ivr",
                "posterior_lucb",
                "author_global_voi_e5c_adapter",
                "calibrated_no_hf",
                "proxy_only",
                "all_hf_raw_reference",
            ):
                selected = [
                    row["selected_id"]
                    for row in scored
                    if row["horizon"] == horizon
                    and int(row["budget"]) == budget
                    and row["method"] == method
                ]
                if selected:
                    selection_distributions[f"{horizon}_budget_{budget}_{method}"] = dict(
                        sorted(Counter(selected).items())
                    )

    noise_scale = {}
    for horizon in ("short", "long"):
        noise = np.asarray(calibration["horizons"][horizon]["observation_noise"][1:], dtype=float)
        covariance = np.asarray(calibration["horizons"][horizon]["latent_covariance"], dtype=float)
        posterior_sd = np.sqrt(np.diag(covariance)[1:])
        noise_scale[horizon] = {
            "query_observation_sd_min": float(np.sqrt(noise.min())),
            "query_observation_sd_median": float(np.median(np.sqrt(noise))),
            "query_observation_sd_max": float(np.sqrt(noise.max())),
            "posterior_latent_sd_min": float(posterior_sd.min()),
            "posterior_latent_sd_median": float(np.median(posterior_sd)),
            "posterior_latent_sd_max": float(posterior_sd.max()),
            "median_query_sd_over_median_latent_sd": float(
                np.median(np.sqrt(noise)) / np.median(posterior_sd)
            ),
        }

    output = {
        "status": "POST_AUDIT_DIAGNOSTIC_ONLY",
        "formal_decision": audit["decision"],
        "protocol_lock_sha256": audit["protocol_lock_sha256"],
        "selection_seal_sha256": audit["selection_seal_sha256"],
        "paired_contrasts": paired_contrasts,
        "acquisition_diagnostics": acquisition_diagnostics,
        "selection_distributions": selection_distributions,
        "calibrated_noise_scale": noise_scale,
        "interpretation_guards": [
            "Only the contrasts stored in audit_results.json were predeclared confirmatory tests.",
            "These additional paired intervals diagnose the frozen result and are exploratory.",
            "Uniform/random allocation replicas are averaged within root before inference.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
