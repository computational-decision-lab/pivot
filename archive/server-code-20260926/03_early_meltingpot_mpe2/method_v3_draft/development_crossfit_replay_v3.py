"""Root-LOO development replay of the PIVOT-CG-v3 direct-deployment posterior.

This module never launches simulations.  It uses only the inspected
44000--44029 cohort, which is development data, and opens audit labels only
after the runtime selector has returned a decision.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from joint_gaussian_selector_v2 import run_joint_gaussian_selector
except ModuleNotFoundError:
    from colin_pivot_cloud.joint_gaussian_selector_v2 import run_joint_gaussian_selector

try:
    from method_v2_draft.development_crossfit_replay import fit_model as fit_v2_model
except ModuleNotFoundError:
    from colin_pivot_cloud.method_v2_draft.development_crossfit_replay import (
        fit_model as fit_v2_model,
    )


VERSION = "melting_pivot_cg_v3_direct_deployment_root_loo_development"
GRID = np.asarray([0.125, 0.225, 0.325, 0.425, 0.575, 0.675, 0.775, 0.875])
FEATURES = np.c_[(GRID - 0.5) / 0.25, np.abs(GRID - 0.5) / 0.25]
FEATURE_PINV = np.linalg.pinv(FEATURES)
FEATURE_PROJECTOR = FEATURES @ FEATURE_PINV
RESIDUAL_PROJECTOR = np.eye(len(GRID)) - FEATURE_PROJECTOR
CANDIDATE_IDS = tuple(str(i) for i in range(len(GRID)))
ALL_IDS = CANDIDATE_IDS + ("incumbent",)
ADAPTATIONS = (4, 32)
CAPS = (96, 192, 384)
METHODS = ("joint_gaussian_voi", "uniform_random", "global_ivr", "posterior_lucb")
DISPLAY_NAMES = {
    "joint_gaussian_voi": "PIVOT-CG-v3 (ours, development)",
    "uniform_random": "Uniform/Random HF (matched)",
    "global_ivr": "Global-IVR (matched)",
    "posterior_lucb": "Posterior LUCB heuristic (matched)",
    "calibrated_no_hf": "Calibrated, no HF",
    "proxy_only": "Proxy only",
}
COVARIANCE_DIAGONAL_SHRINKAGE = 0.25
QUERY_NOISE_POOLING = 0.50
NUMERICAL_FLOOR = 1e-8
NLL_FLOOR = 1e-6


def read_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _matrix_by_block(summary: dict, adaptation: int, field: str) -> dict[str, np.ndarray]:
    output = {"A": np.full(len(GRID), np.nan), "B": np.full(len(GRID), np.nan)}
    for row in summary["mechanism_audit"]:
        if int(row["adaptation"]) == adaptation:
            output[str(row["block"])][int(row["candidate"])] = float(row[field])
    if not all(np.all(np.isfinite(values)) for values in output.values()):
        raise ValueError(f"incomplete audit field={field}, h={adaptation}")
    return output


def load_root(path: Path) -> dict:
    seed = int(path.name.split("_")[-1])
    features = read_json(path / "features.json")
    if [str(row["id"]) for row in features] != list(CANDIDATE_IDS):
        raise ValueError(f"candidate order mismatch in {path}")
    matrix = np.asarray([row["features"] for row in features], dtype=float)
    if not np.allclose(matrix, FEATURES, rtol=0.0, atol=1e-12):
        raise ValueError(f"feature mismatch in {path}")
    proxy = np.asarray([row["proxy_delta"] for row in features], dtype=float)
    summary = read_json(path / "summary.json")
    decisions = read_json(path / "decisions_frozen.json")
    selection = {}
    audit_deployment = {}
    audit_proxy = {}
    for adaptation in ADAPTATIONS:
        matches = [
            row for row in decisions
            if row["method"] == "all_hf_reference" and int(row["adaptation"]) == adaptation
        ]
        if len(matches) != 1:
            raise ValueError(f"missing all-HF bank for seed={seed}, h={adaptation}")
        selection[adaptation] = np.asarray(
            [matches[0]["observed"][identifier] for identifier in CANDIDATE_IDS], dtype=float
        )
        audit_deployment[adaptation] = _matrix_by_block(
            summary, adaptation, "deployment_delta_audit"
        )
        audit_proxy[adaptation] = _matrix_by_block(summary, adaptation, "proxy_delta_audit")
    return {
        "seed": seed,
        "path": path,
        "proxy": proxy,
        "selection": selection,
        "audit_deployment": audit_deployment,
        "audit_proxy": audit_proxy,
        # Compatibility view used only to calculate the v2 diagnostic.
        "audit": audit_deployment,
    }


def second_moment(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values.T @ values / len(values)


def psd_projection(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    values, vectors = np.linalg.eigh(symmetric)
    projected = (vectors * np.maximum(values, 0.0)) @ vectors.T
    return 0.5 * (projected + projected.T), values


def shrink_covariance(matrix: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fixed 25% off-diagonal shrinkage after PSD projection."""
    projected, raw_eigenvalues = psd_projection(matrix)
    weight = COVARIANCE_DIAGONAL_SHRINKAGE
    shrunk = (1.0 - weight) * projected + weight * np.diag(np.diag(projected))
    scale = max(1.0, float(np.mean(np.diag(shrunk))))
    floor = NUMERICAL_FLOOR * scale
    shrunk = 0.5 * (shrunk + shrunk.T) + floor * np.eye(len(shrunk))
    return shrunk, {
        "raw_eigenvalues": raw_eigenvalues.tolist(),
        "projected_eigenvalues": np.linalg.eigvalsh(projected).tolist(),
        "shrunk_eigenvalues": np.linalg.eigvalsh(shrunk).tolist(),
        "diagonal_shrinkage": weight,
        "numerical_floor": floor,
    }


def shrink_query_noise(raw: np.ndarray) -> tuple[np.ndarray, dict]:
    clipped = np.maximum(np.asarray(raw, dtype=float), 0.0)
    pooled = float(np.mean(clipped))
    pooled_target = (1.0 - QUERY_NOISE_POOLING) * clipped + QUERY_NOISE_POOLING * pooled
    shrunk = np.maximum(clipped, pooled_target)
    floor = NUMERICAL_FLOOR * max(1.0, pooled)
    shrunk = np.maximum(shrunk, floor)
    return shrunk, {
        "raw": np.asarray(raw, dtype=float).tolist(),
        "nonnegative_raw": clipped.tolist(),
        "pooled_nonnegative_mean": pooled,
        "pooling_weight_for_low_estimates": QUERY_NOISE_POOLING,
        "shrunk": shrunk.tolist(),
    }


def fit_model(roots: list[dict], adaptation: int) -> dict:
    """Fit v3 moments using only the supplied complete roots."""
    if len(roots) < 4:
        raise ValueError("at least four complete roots are required")
    proxy_observed = np.asarray([root["proxy"] for root in roots])
    deployment_a = np.asarray([root["audit_deployment"][adaptation]["A"] for root in roots])
    deployment_b = np.asarray([root["audit_deployment"][adaptation]["B"] for root in roots])
    proxy_a = np.asarray([root["audit_proxy"][adaptation]["A"] for root in roots])
    proxy_b = np.asarray([root["audit_proxy"][adaptation]["B"] for root in roots])
    selection = np.asarray([root["selection"][adaptation] for root in roots])

    deployment_mean = 0.5 * (deployment_a + deployment_b)
    proxy_audit_mean = 0.5 * (proxy_a + proxy_b)
    gap_a = deployment_a - proxy_a
    gap_b = deployment_b - proxy_b
    gap_mean = 0.5 * (gap_a + gap_b)

    r_deployment_mean, dep_noise_info = shrink_covariance(
        second_moment(deployment_a - deployment_b) / 4.0
    )
    r_proxy_mean, proxy_noise_info = shrink_covariance(second_moment(proxy_a - proxy_b) / 4.0)
    r_gap_mean, gap_noise_info = shrink_covariance(second_moment(gap_a - gap_b) / 4.0)

    correction_mean = gap_mean.mean(axis=0)
    mu_beta = FEATURE_PINV @ correction_mean
    delta = correction_mean - FEATURES @ mu_beta

    beta_by_root = gap_mean @ FEATURE_PINV.T
    beta_measurement = FEATURE_PINV @ r_gap_mean @ FEATURE_PINV.T
    sigma_beta_raw = np.cov(beta_by_root, rowvar=False, ddof=1) - beta_measurement
    sigma_beta, sigma_beta_info = shrink_covariance(sigma_beta_raw)

    centered_gap = gap_mean - correction_mean
    mis_by_root = centered_gap @ RESIDUAL_PROJECTOR.T
    mis_noise = RESIDUAL_PROJECTOR @ r_gap_mean @ RESIDUAL_PROJECTOR.T
    k_mis_raw = np.cov(mis_by_root, rowvar=False, ddof=1) - mis_noise
    k_mis, k_mis_info = shrink_covariance(k_mis_raw)

    proxy_error = proxy_audit_mean - proxy_observed
    k_proxy_raw = np.cov(proxy_error, rowvar=False, ddof=1) - r_proxy_mean
    k_proxy, k_proxy_info = shrink_covariance(k_proxy_raw)

    # Sampling uncertainty for the full fitted correction mean, including
    # both feature and off-feature directions and finite audit measurement.
    k_mean_raw = np.cov(gap_mean, rowvar=False, ddof=1) / len(roots)
    k_mean, k_mean_info = shrink_covariance(k_mean_raw)

    k_beta = FEATURES @ sigma_beta @ FEATURES.T
    latent_covariance = k_beta + k_mean + k_proxy + k_mis
    latent_covariance = 0.5 * (latent_covariance + latent_covariance.T)
    if np.min(np.linalg.eigvalsh(latent_covariance)) < -1e-8:
        raise AssertionError("sum of v3 covariance components is not PSD")

    raw_r_hf = np.mean((selection - deployment_a) * (selection - deployment_b), axis=0)
    r_hf, r_hf_info = shrink_query_noise(raw_r_hf)
    return {
        "training_seeds": [root["seed"] for root in roots],
        "mu_beta": mu_beta,
        "delta": delta,
        "correction_mean": correction_mean,
        "sigma_beta": sigma_beta,
        "k_beta": k_beta,
        "k_mean": k_mean,
        "k_proxy": k_proxy,
        "k_mis": k_mis,
        "latent_covariance": latent_covariance,
        "r_hf": r_hf,
        "r_audit_mean": r_deployment_mean,
        "component_info": {
            "sigma_beta": sigma_beta_info,
            "k_mean": k_mean_info,
            "k_proxy": k_proxy_info,
            "k_mis": k_mis_info,
            "r_hf": r_hf_info,
            "r_deployment_audit_mean": dep_noise_info,
            "r_proxy_audit_mean": proxy_noise_info,
            "r_gap_audit_mean": gap_noise_info,
        },
    }


def posterior_for_proxy(proxy: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build runtime posterior from the public proxy vector only."""
    proxy = np.asarray(proxy, dtype=float)
    if proxy.shape != (len(GRID),) or not np.all(np.isfinite(proxy)):
        raise ValueError("runtime proxy must be a finite candidate vector")
    candidate_mean = proxy + FEATURES @ model["mu_beta"] + model["delta"]
    mean = np.r_[candidate_mean, 0.0]
    covariance = np.zeros((len(ALL_IDS), len(ALL_IDS)))
    covariance[:-1, :-1] = model["latent_covariance"]
    observation_variance = np.r_[model["r_hf"], 0.0]
    return mean, covariance, observation_variance


def gaussian_scores(target: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> dict:
    dimension = len(target)
    covariance = 0.5 * (covariance + covariance.T)
    scale = max(1.0, float(np.trace(covariance) / dimension))
    floor = NLL_FLOOR * scale
    values, vectors = np.linalg.eigh(covariance)
    stabilized_values = np.maximum(values, floor)
    inverse = (vectors * (1.0 / stabilized_values)) @ vectors.T
    residual = target - mean
    logdet = float(np.log(stabilized_values).sum())
    multivariate_nll = 0.5 * (
        dimension * math.log(2.0 * math.pi) + logdet + float(residual @ inverse @ residual)
    )
    variance = np.maximum(np.diag(covariance), floor)
    marginal_nll = 0.5 * (
        math.log(2.0 * math.pi) + np.log(variance) + residual**2 / variance
    )
    z = np.abs(residual) / np.sqrt(variance)
    return {
        "multivariate_nll": float(multivariate_nll),
        "marginal_nll": marginal_nll,
        "covered_80": z <= 1.2815515655446004,
        "covered_95": z <= 1.959963984540054,
        "whitened_quadratic_per_dimension": float(residual @ inverse @ residual / dimension),
        "scoring_floor": floor,
    }


def bootstrap_mean(values: Iterable[float], seed: int) -> dict:
    array = np.asarray(list(values), dtype=float)
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(len(array), size=(10000, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "n_root_seeds": int(len(array)),
    }


def summarize_calibration(records: list[dict]) -> list[dict]:
    output = []
    for adaptation in ADAPTATIONS:
        for version in ("v2", "v3"):
            rows = [r for r in records if r["adaptation"] == adaptation and r["version"] == version]
            output.append({
                "adaptation": adaptation,
                "posterior": version,
                "marginal_nll": bootstrap_mean(
                    [float(np.mean(r["marginal_nll"])) for r in rows],
                    810000 + adaptation * 10 + (version == "v3"),
                ),
                "multivariate_nll_per_root": bootstrap_mean(
                    [r["multivariate_nll"] for r in rows],
                    820000 + adaptation * 10 + (version == "v3"),
                ),
                "coverage_80": float(np.mean([x for r in rows for x in r["covered_80"]])),
                "coverage_95": float(np.mean([x for r in rows for x in r["covered_95"]])),
                "whitened_quadratic_per_dimension": bootstrap_mean(
                    [r["whitened_quadratic_per_dimension"] for r in rows],
                    830000 + adaptation * 10 + (version == "v3"),
                ),
            })
    return output


def compare_calibration(records: list[dict]) -> list[dict]:
    """Paired root-level v2-minus-v3 NLL differences; positive favors v3."""
    output = []
    for adaptation in ADAPTATIONS:
        indexed = {
            (row["seed"], row["version"]): row
            for row in records if row["adaptation"] == adaptation
        }
        seeds = sorted({seed for seed, _ in indexed})
        marginal_difference = [
            float(np.mean(indexed[seed, "v2"]["marginal_nll"]))
            - float(np.mean(indexed[seed, "v3"]["marginal_nll"]))
            for seed in seeds
        ]
        multivariate_difference = [
            indexed[seed, "v2"]["multivariate_nll"]
            - indexed[seed, "v3"]["multivariate_nll"]
            for seed in seeds
        ]
        v2_marginal = float(np.mean([
            np.mean(indexed[seed, "v2"]["marginal_nll"]) for seed in seeds
        ]))
        v2_multivariate = float(np.mean([
            indexed[seed, "v2"]["multivariate_nll"] for seed in seeds
        ]))
        output.append({
            "adaptation": adaptation,
            "positive_favors_v3": True,
            "paired_marginal_nll_reduction": bootstrap_mean(
                marginal_difference, 840000 + adaptation
            ),
            "paired_multivariate_nll_reduction": bootstrap_mean(
                multivariate_difference, 850000 + adaptation
            ),
            "relative_marginal_nll_reduction": float(np.mean(marginal_difference) / v2_marginal),
            "relative_multivariate_nll_reduction": float(
                np.mean(multivariate_difference) / v2_multivariate
            ),
        })
    return output


def run_replay(roots: list[dict]) -> tuple[list[dict], dict]:
    decisions = []
    calibration_records = []
    fold_models = {}
    method_seed_offset = {
        "joint_gaussian_voi": 11,
        "uniform_random": 23,
        "global_ivr": 37,
        "posterior_lucb": 53,
    }
    for held in roots:
        training = [root for root in roots if root["seed"] != held["seed"]]
        fold_models[str(held["seed"])] = {}
        for adaptation in ADAPTATIONS:
            model = fit_model(training, adaptation)
            mean, covariance, observation_variance = posterior_for_proxy(held["proxy"], model)
            fold_models[str(held["seed"])][str(adaptation)] = {
                "training_seeds": model["training_seeds"],
                "mu_beta": model["mu_beta"].tolist(),
                "delta": model["delta"].tolist(),
                "r_hf": model["r_hf"].tolist(),
                "component_trace": {
                    name: float(np.trace(model[name]))
                    for name in ("k_beta", "k_mean", "k_proxy", "k_mis", "latent_covariance")
                },
                "component_info": model["component_info"],
            }
            audit = 0.5 * (
                held["audit_deployment"][adaptation]["A"]
                + held["audit_deployment"][adaptation]["B"]
            )
            v3_scores = gaussian_scores(
                audit,
                mean[:-1],
                covariance[:-1, :-1] + model["r_audit_mean"],
            )
            calibration_records.append({"seed": held["seed"], "adaptation": adaptation,
                                        "version": "v3", **v3_scores})

            # Same held roots and scoring target for the previous development posterior.
            v2 = fit_v2_model(training, adaptation)
            v2_mean = held["proxy"] + FEATURES @ v2["coefficient_mean"]
            v2_scores = gaussian_scores(
                audit, v2_mean, v2["latent_covariance"] + v2["audit_mean_noise_covariance"]
            )
            calibration_records.append({"seed": held["seed"], "adaptation": adaptation,
                                        "version": "v2", **v2_scores})

            audit_all = np.r_[audit, 0.0]
            for cap in CAPS:
                query_cost = float(adaptation + 32)
                setup_cost = float(adaptation)
                selector_budget = max(0.0, float(cap) - setup_cost)
                costs = np.r_[np.full(len(CANDIDATE_IDS), query_cost), 0.0]
                queryable = [True] * len(CANDIDATE_IDS) + [False]
                selection_values = held["selection"][adaptation].copy()
                for method in METHODS:
                    def query(request, values=selection_values):
                        index = int(request["index"])
                        if index >= len(CANDIDATE_IDS):
                            raise AssertionError("incumbent is not queryable")
                        return {"observation": float(values[index]), "cost": float(request["cost"])}

                    result = run_joint_gaussian_selector(
                        ALL_IDS, mean, covariance, observation_variance, costs, query,
                        method=method,
                        budget=selector_budget,
                        seed=held["seed"] + method_seed_offset[method],
                        queryable=queryable,
                        incumbent_id="incumbent",
                        stop_on_zero_voi=False,
                    )
                    selected = ALL_IDS.index(result["selected_id"])
                    total_cost = result["spent_cost"] + (setup_cost if result["query_count"] else 0.0)
                    if total_cost > cap + 1e-9:
                        raise AssertionError("HF cap exceeded")
                    decisions.append({
                        "seed": held["seed"], "adaptation": adaptation, "budget_cap": cap,
                        "method": method, "display_name": DISPLAY_NAMES[method],
                        "selected_id": result["selected_id"],
                        "audit_gain": float(audit_all[selected]),
                        "audit_regret": float(np.max(audit_all) - audit_all[selected]),
                        "query_count": result["query_count"], "hf_episode_cost": total_cost,
                        "queried_ids": result["queried_ids"],
                        "outer_training_seeds": model["training_seeds"],
                    })
                for method, estimate in (
                    ("calibrated_no_hf", mean), ("proxy_only", np.r_[held["proxy"], 0.0])
                ):
                    selected = int(np.argmax(estimate))
                    decisions.append({
                        "seed": held["seed"], "adaptation": adaptation, "budget_cap": cap,
                        "method": method, "display_name": DISPLAY_NAMES[method],
                        "selected_id": ALL_IDS[selected], "audit_gain": float(audit_all[selected]),
                        "audit_regret": float(np.max(audit_all) - audit_all[selected]),
                        "query_count": 0, "hf_episode_cost": 0.0, "queried_ids": [],
                        "outer_training_seeds": model["training_seeds"],
                    })

    method_table = []
    for adaptation in ADAPTATIONS:
        for cap in CAPS:
            for method in METHODS + ("calibrated_no_hf", "proxy_only"):
                rows = [r for r in decisions if r["adaptation"] == adaptation
                        and r["budget_cap"] == cap and r["method"] == method]
                method_table.append({
                    "adaptation": adaptation, "budget_cap": cap, "method": method,
                    "display_name": DISPLAY_NAMES[method],
                    "audit_regret": bootstrap_mean(
                        [r["audit_regret"] for r in rows], 910000 + adaptation * 100 + cap + len(method)
                    ),
                    "audit_gain": bootstrap_mean(
                        [r["audit_gain"] for r in rows], 920000 + adaptation * 100 + cap + len(method)
                    ),
                    "mean_query_count": float(np.mean([r["query_count"] for r in rows])),
                    "mean_hf_episode_cost": float(np.mean([r["hf_episode_cost"] for r in rows])),
                    "selection_accuracy": float(np.mean([r["audit_regret"] <= 1e-12 for r in rows])),
                })

    contrasts = []
    for adaptation in ADAPTATIONS:
        for cap in CAPS:
            pivot = {r["seed"]: r for r in decisions if r["adaptation"] == adaptation
                     and r["budget_cap"] == cap and r["method"] == "joint_gaussian_voi"}
            for comparator in ("uniform_random", "global_ivr", "posterior_lucb", "calibrated_no_hf"):
                other = {r["seed"]: r for r in decisions if r["adaptation"] == adaptation
                         and r["budget_cap"] == cap and r["method"] == comparator}
                difference = [other[s]["audit_regret"] - pivot[s]["audit_regret"] for s in sorted(pivot)]
                contrasts.append({
                    "adaptation": adaptation, "budget_cap": cap,
                    "contrast": f"PIVOT-CG-v3 regret reduction vs {comparator}",
                    "positive_favors_pivot": True,
                    "regret_reduction": bootstrap_mean(
                        difference, 930000 + adaptation * 100 + cap + len(comparator)
                    ),
                })

    calibration = summarize_calibration(calibration_records)
    calibration_comparisons = compare_calibration(calibration_records)
    primary = next(x for x in contrasts if x["adaptation"] == 32 and x["budget_cap"] == 192
                   and x["contrast"].endswith("uniform_random"))
    return decisions, {
        "status": "complete_development_replay",
        "version": VERSION,
        "n_development_roots": len(roots),
        "development_seeds": [r["seed"] for r in roots],
        "method_table": method_table,
        "contrasts": contrasts,
        "held_out_calibration": calibration,
        "held_out_calibration_comparisons": calibration_comparisons,
        "fold_models": fold_models,
        "diagnostics": {
            "primary_development_contrast": primary,
            "all_root_folds_exclude_held_root": all(
                row["seed"] not in row["outer_training_seeds"] for row in decisions
            ),
            "audit_labels_passed_to_selector": False,
            "formal_confirmation_status": "NOT_RUN",
        },
        "shrinkage_rule": {
            "covariance_diagonal_weight": COVARIANCE_DIAGONAL_SHRINKAGE,
            "query_noise_lower_tail_pooling_weight": QUERY_NOISE_POOLING,
            "numerical_floor": NUMERICAL_FLOOR,
            "nll_floor": NLL_FLOOR,
            "tuned_against_regret": False,
        },
        "claim_boundary": (
            "All 44000--44029 outcomes are development-only. This replay can diagnose and "
            "freeze a posterior, but cannot establish a paper-level v3 advantage."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(path for path in args.development_root.glob("seed_44*") if path.is_dir())
    roots = [load_root(path) for path in paths]
    if [root["seed"] for root in roots] != list(range(44000, 44030)):
        raise ValueError("development root must contain exactly seeds 44000--44029")
    args.output.mkdir(parents=True, exist_ok=False)
    decisions, summary = run_replay(roots)
    sources = [Path(__file__), Path(__file__).with_name("POSTERIOR_SPEC.md"),
               Path(__file__).parents[1] / "joint_gaussian_selector_v2.py"]
    summary["source_sha256"] = {str(path): sha256(path) for path in sources}
    summary["input_sha256"] = {
        str(root["path"] / name): sha256(root["path"] / name)
        for root in roots for name in ("features.json", "decisions_frozen.json", "summary.json")
    }
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "decisions.json", decisions)
    with (args.output / "decisions.csv").open("w", newline="") as handle:
        fields = ["seed", "adaptation", "budget_cap", "method", "display_name",
                  "selected_id", "audit_gain", "audit_regret", "query_count", "hf_episode_cost"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in decisions:
            writer.writerow({field: row[field] for field in fields})
    print(json.dumps({
        "status": summary["status"],
        "primary_development_contrast": summary["diagnostics"]["primary_development_contrast"],
        "held_out_calibration": summary["held_out_calibration"],
    }, indent=2))


if __name__ == "__main__":
    main()
