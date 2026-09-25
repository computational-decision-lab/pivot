"""Root-cross-fitted development replay for PIVOT-CG-v2.

This script consumes the already-inspected 44000--44029 Melting Pot cohort.
Those roots are development data after the frozen v1 null result; no number
produced here is fresh confirmation evidence.  For every scored root, all
calibration parameters are fitted on the other roots.  The selector can read
only proxy values and the historical selection-query bank.  Independent A/B
audit values are opened only after a decision has been returned.
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


VERSION = "melting_pivot_cg_v2_root_crossfit_development_replay"
GRID = np.asarray([0.125, 0.225, 0.325, 0.425, 0.575, 0.675, 0.775, 0.875])
FEATURES = np.c_[(GRID - 0.5) / 0.25, np.abs(GRID - 0.5) / 0.25]
CANDIDATE_IDS = tuple(str(index) for index in range(len(GRID)))
ALL_IDS = CANDIDATE_IDS + ("incumbent",)
ADAPTATIONS = (4, 32)
CAPS = (96, 192, 384)
SELECTOR_METHODS = (
    "joint_gaussian_voi",
    "uniform_random",
    "global_ivr",
    "posterior_lucb",
)
DISPLAY_NAMES = {
    "joint_gaussian_voi": "PIVOT-CG-v2 (ours, development)",
    "uniform_random": "Uniform/Random HF (matched)",
    "global_ivr": "Global-IVR (matched)",
    "posterior_lucb": "Posterior LUCB heuristic (matched)",
    "calibrated_no_hf": "Calibrated, no HF",
    "proxy_only": "Proxy only",
}


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


def _matrix_by_block(summary: dict, adaptation: int) -> dict[str, np.ndarray]:
    result = {"A": np.full(len(GRID), np.nan), "B": np.full(len(GRID), np.nan)}
    for row in summary["mechanism_audit"]:
        if int(row["adaptation"]) != adaptation:
            continue
        index = int(row["candidate"])
        result[str(row["block"])][index] = float(row["deployment_delta_audit"])
    for block, values in result.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"incomplete adaptation={adaptation} audit block={block}")
    return result


def load_root(path: Path) -> dict:
    seed = int(path.name.split("_")[-1])
    feature_rows = read_json(path / "features.json")
    if [str(row["id"]) for row in feature_rows] != list(CANDIDATE_IDS):
        raise ValueError(f"candidate order mismatch in {path}")
    matrix = np.asarray([row["features"] for row in feature_rows], dtype=float)
    if not np.allclose(matrix, FEATURES, rtol=0.0, atol=1e-12):
        raise ValueError(f"feature grid mismatch in {path}")
    proxy = np.asarray([row["proxy_delta"] for row in feature_rows], dtype=float)
    summary = read_json(path / "summary.json")
    decisions = read_json(path / "decisions_frozen.json")
    selection = {}
    for adaptation in ADAPTATIONS:
        matches = [
            row
            for row in decisions
            if row["method"] == "all_hf_reference" and int(row["adaptation"]) == adaptation
        ]
        if len(matches) != 1:
            raise ValueError(f"missing all-HF selection bank for seed {seed}, h={adaptation}")
        selection[adaptation] = np.asarray(
            [matches[0]["observed"][identifier] for identifier in CANDIDATE_IDS], dtype=float
        )
    audits = {adaptation: _matrix_by_block(summary, adaptation) for adaptation in ADAPTATIONS}
    for adaptation in ADAPTATIONS:
        expected = np.asarray(
            [summary["audit_gains"][str(adaptation)][identifier] for identifier in CANDIDATE_IDS]
        )
        actual = 0.5 * (audits[adaptation]["A"] + audits[adaptation]["B"])
        if not np.allclose(expected, actual, rtol=0.0, atol=1e-10):
            raise ValueError(f"audit reconstruction mismatch in seed {seed}, h={adaptation}")
    return {
        "seed": seed,
        "path": path,
        "proxy": proxy,
        "selection": selection,
        "audit": audits,
    }


def _psd_projection(matrix: np.ndarray) -> tuple[np.ndarray, list[float]]:
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    projected = (vectors * np.maximum(values, 0.0)) @ vectors.T
    return 0.5 * (projected + projected.T), values.tolist()


def base_moments(roots: list[dict], adaptation: int) -> dict:
    if len(roots) < 3:
        raise ValueError("at least three roots are required")
    proxy = np.asarray([root["proxy"] for root in roots])
    audit_a = np.asarray([root["audit"][adaptation]["A"] for root in roots])
    audit_b = np.asarray([root["audit"][adaptation]["B"] for root in roots])
    audit_mean = 0.5 * (audit_a + audit_b)
    selection = np.asarray([root["selection"][adaptation] for root in roots])
    correction = audit_mean - proxy
    feature_pinv = np.linalg.pinv(FEATURES)
    root_coefficients = correction @ feature_pinv.T
    coefficient_mean = root_coefficients.mean(axis=0)
    coefficient_covariance = np.cov(root_coefficients, rowvar=False, ddof=1)
    coefficient_mean_covariance = coefficient_covariance / len(roots)

    # A and B each contain 16 replicas per native stratum.  Their difference
    # has covariance 2 R_16, and their mean has covariance Cov(A-B)/4.
    audit_difference_covariance = np.cov(audit_a - audit_b, rowvar=False, ddof=1)
    audit_mean_noise_covariance = audit_difference_covariance / 4.0
    # The historical selection block is independent of A/B and has eight
    # replicas per stratum.  Estimate its actual marginal noise directly:
    # Var(selection - audit_mean) = R_selection + R_audit_mean.  This avoids
    # assuming ideal 1/n scaling across two differently keyed native blocks.
    selection_minus_audit = selection - audit_mean
    selection_difference_covariance = np.cov(
        selection_minus_audit, rowvar=False, ddof=1
    )
    raw_selection_noise = selection_difference_covariance - audit_mean_noise_covariance
    selection_noise_covariance, selection_noise_raw_eigenvalues = _psd_projection(
        raw_selection_noise
    )
    selection_observation_variance = np.maximum(np.diag(selection_noise_covariance), 1e-8)
    return {
        "coefficient_mean": coefficient_mean,
        "coefficient_mean_covariance": coefficient_mean_covariance,
        "audit_mean_noise_covariance": audit_mean_noise_covariance,
        "selection_observation_variance": selection_observation_variance,
        "selection_noise_covariance": selection_noise_covariance,
        "selection_noise_raw_eigenvalues": selection_noise_raw_eigenvalues,
        "root_coefficients": root_coefficients,
    }


def exchangeable_discrepancy(
    residuals: np.ndarray, known_covariances: np.ndarray
) -> tuple[np.ndarray, dict]:
    if residuals.ndim != 2 or len(residuals) < 3:
        raise ValueError("cross-fitted residuals must be a root-by-candidate matrix")
    count, candidates = residuals.shape
    if known_covariances.shape != (count, candidates, candidates):
        raise ValueError("known covariance shape mismatch")
    raw = residuals.T @ residuals / count - known_covariances.mean(axis=0)
    raw = 0.5 * (raw + raw.T)
    unit = np.ones(candidates) / math.sqrt(candidates)
    common_projector = np.outer(unit, unit)
    contrast_projector = np.eye(candidates) - common_projector
    common = max(0.0, float(unit @ raw @ unit))
    contrast = max(0.0, float(np.trace(contrast_projector @ raw) / (candidates - 1)))
    discrepancy = common * common_projector + contrast * contrast_projector
    return discrepancy, {
        "common_direction_eigenvalue": common,
        "contrast_direction_eigenvalue": contrast,
        "raw_after_known_terms": raw.tolist(),
    }


def fit_model(roots: list[dict], adaptation: int) -> dict:
    """Fit only from supplied roots, with an inner root-level cross-fit."""
    residuals = []
    known = []
    inner_folds = []
    for held in roots:
        training = [root for root in roots if root["seed"] != held["seed"]]
        moments = base_moments(training, adaptation)
        audit_mean = 0.5 * (
            held["audit"][adaptation]["A"] + held["audit"][adaptation]["B"]
        )
        predicted = held["proxy"] + FEATURES @ moments["coefficient_mean"]
        residuals.append(audit_mean - predicted)
        known.append(
            FEATURES @ moments["coefficient_mean_covariance"] @ FEATURES.T
            + moments["audit_mean_noise_covariance"]
        )
        inner_folds.append({
            "held_seed": held["seed"],
            "training_seeds": [root["seed"] for root in training],
        })
    discrepancy, discrepancy_info = exchangeable_discrepancy(
        np.asarray(residuals), np.asarray(known)
    )
    moments = base_moments(roots, adaptation)
    latent_covariance = (
        FEATURES @ moments["coefficient_mean_covariance"] @ FEATURES.T + discrepancy
    )
    latent_covariance, pre_projection_eigenvalues = _psd_projection(latent_covariance)
    return {
        **moments,
        "discrepancy_covariance": discrepancy,
        "discrepancy_info": discrepancy_info,
        "latent_covariance": latent_covariance,
        "latent_covariance_pre_projection_eigenvalues": pre_projection_eigenvalues,
        "inner_folds": inner_folds,
    }


def posterior_for_root(root: dict, model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_mean = root["proxy"] + FEATURES @ model["coefficient_mean"]
    mean = np.r_[candidate_mean, 0.0]
    covariance = np.zeros((len(ALL_IDS), len(ALL_IDS)))
    covariance[:-1, :-1] = model["latent_covariance"]
    observation_variance = np.r_[model["selection_observation_variance"], 0.0]
    return mean, covariance, observation_variance


def bootstrap_mean(values: Iterable[float], seed: int) -> dict:
    array = np.asarray(list(values), dtype=float)
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(len(array), size=(10000, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "n_root_seeds": int(len(array)),
    }


def run_replay(roots: list[dict]) -> tuple[list[dict], dict]:
    rows = []
    coverage = {adaptation: [] for adaptation in ADAPTATIONS}
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
            mean, covariance, observation_variance = posterior_for_root(held, model)
            fold_models[str(held["seed"])][str(adaptation)] = {
                "training_seeds": [root["seed"] for root in training],
                "coefficient_mean": model["coefficient_mean"].tolist(),
                "coefficient_mean_covariance": model["coefficient_mean_covariance"].tolist(),
                "selection_observation_variance": model["selection_observation_variance"].tolist(),
                "discrepancy_info": model["discrepancy_info"],
            }
            audit = 0.5 * (
                held["audit"][adaptation]["A"] + held["audit"][adaptation]["B"]
            )
            audit_all = np.r_[audit, 0.0]
            audit_noise = np.zeros_like(covariance)
            audit_noise[:-1, :-1] = model["audit_mean_noise_covariance"]
            predictive_sd = np.sqrt(np.maximum(np.diag(covariance + audit_noise), 1e-12))
            coverage[adaptation].extend(
                (np.abs(audit_all[:-1] - mean[:-1]) <= 1.96 * predictive_sd[:-1]).tolist()
            )
            for cap in CAPS:
                query_cost = float(adaptation + 32)
                setup_cost = float(adaptation)
                selector_budget = max(0.0, float(cap) - setup_cost)
                costs = np.r_[np.full(len(CANDIDATE_IDS), query_cost), 0.0]
                queryable = [True] * len(CANDIDATE_IDS) + [False]

                # The callback closes only over the selection split.  Audit is
                # intentionally scored after run_joint_gaussian_selector returns.
                selection_values = held["selection"][adaptation].copy()
                for method in SELECTOR_METHODS:
                    def query(request, values=selection_values):
                        index = int(request["index"])
                        if index >= len(CANDIDATE_IDS):
                            raise AssertionError("incumbent is not queryable")
                        return {"observation": float(values[index]), "cost": float(request["cost"])}

                    result = run_joint_gaussian_selector(
                        ALL_IDS,
                        mean,
                        covariance,
                        observation_variance,
                        costs,
                        query,
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
                        raise AssertionError("HF episode cap exceeded")
                    zero_ties = 0
                    if method == "joint_gaussian_voi":
                        for step in result["steps"]:
                            scores = list(step["evsi"].values())
                            zero_ties += int(scores and max(scores) <= 1e-12)
                    rows.append({
                        "seed": held["seed"],
                        "adaptation": adaptation,
                        "budget_cap": cap,
                        "method": method,
                        "display_name": DISPLAY_NAMES[method],
                        "selected_id": result["selected_id"],
                        "audit_gain": float(audit_all[selected]),
                        "audit_regret": float(np.max(audit_all) - audit_all[selected]),
                        "query_count": result["query_count"],
                        "hf_episode_cost": total_cost,
                        "queried_ids": result["queried_ids"],
                        "zero_evsi_tie_steps": zero_ties,
                        "outer_training_seeds": [root["seed"] for root in training],
                    })

                for method, estimates in (
                    ("calibrated_no_hf", mean),
                    ("proxy_only", np.r_[held["proxy"], 0.0]),
                ):
                    selected = int(np.argmax(estimates))
                    rows.append({
                        "seed": held["seed"],
                        "adaptation": adaptation,
                        "budget_cap": cap,
                        "method": method,
                        "display_name": DISPLAY_NAMES[method],
                        "selected_id": ALL_IDS[selected],
                        "audit_gain": float(audit_all[selected]),
                        "audit_regret": float(np.max(audit_all) - audit_all[selected]),
                        "query_count": 0,
                        "hf_episode_cost": 0.0,
                        "queried_ids": [],
                        "zero_evsi_tie_steps": 0,
                        "outer_training_seeds": [root["seed"] for root in training],
                    })

    grouped = {}
    for row in rows:
        key = (row["adaptation"], row["budget_cap"], row["method"])
        grouped.setdefault(key, []).append(row)
    table = []
    for (adaptation, cap, method), group in sorted(grouped.items()):
        table.append({
            "adaptation": adaptation,
            "budget_cap": cap,
            "method": method,
            "display_name": DISPLAY_NAMES[method],
            "audit_regret": bootstrap_mean(
                [row["audit_regret"] for row in group], 910000 + adaptation * 100 + cap
            ),
            "audit_gain": bootstrap_mean(
                [row["audit_gain"] for row in group], 920000 + adaptation * 100 + cap
            ),
            "mean_query_count": float(np.mean([row["query_count"] for row in group])),
            "mean_hf_episode_cost": float(np.mean([row["hf_episode_cost"] for row in group])),
            "selection_accuracy": float(np.mean([row["audit_regret"] <= 1e-12 for row in group])),
            "zero_evsi_tie_steps": int(sum(row["zero_evsi_tie_steps"] for row in group)),
        })

    contrasts = []
    for adaptation in ADAPTATIONS:
        for cap in CAPS:
            pivot = {
                row["seed"]: row
                for row in rows
                if row["adaptation"] == adaptation
                and row["budget_cap"] == cap
                and row["method"] == "joint_gaussian_voi"
            }
            for comparator in ("uniform_random", "global_ivr", "posterior_lucb", "calibrated_no_hf"):
                other = {
                    row["seed"]: row
                    for row in rows
                    if row["adaptation"] == adaptation
                    and row["budget_cap"] == cap
                    and row["method"] == comparator
                }
                differences = [
                    other[seed]["audit_regret"] - pivot[seed]["audit_regret"]
                    for seed in sorted(pivot)
                ]
                contrasts.append({
                    "adaptation": adaptation,
                    "budget_cap": cap,
                    "contrast": f"PIVOT-CG-v2 regret reduction vs {comparator}",
                    "positive_favors_pivot": True,
                    "regret_reduction": bootstrap_mean(
                        differences, 930000 + adaptation * 100 + cap + len(comparator)
                    ),
                })

    primary = next(
        row
        for row in contrasts
        if row["adaptation"] == 32
        and row["budget_cap"] == 192
        and row["contrast"].endswith("uniform_random")
    )
    diagnostics = {
        "observation_predictive_95_coverage": {
            str(adaptation): float(np.mean(coverage[adaptation])) for adaptation in ADAPTATIONS
        },
        "primary_development_contrast": primary,
        "all_root_folds_exclude_held_root": all(
            row["seed"] not in row["outer_training_seeds"] for row in rows
        ),
        "audit_labels_passed_to_selector": False,
        "formal_confirmation_status": "NOT_RUN",
    }
    return rows, {
        "status": "complete_development_replay",
        "version": VERSION,
        "n_development_roots": len(roots),
        "development_seeds": [root["seed"] for root in roots],
        "adaptations": list(ADAPTATIONS),
        "budget_caps": list(CAPS),
        "method_table": table,
        "contrasts": contrasts,
        "diagnostics": diagnostics,
        "fold_models": fold_models,
        "claim_boundary": (
            "44000--44029 were inspected before v2 was designed and are now development data. "
            "This replay can justify freezing a method, but cannot support a paper-level v2 advantage claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root_paths = sorted(args.development_root.glob("seed_44*"))
    roots = [load_root(path) for path in root_paths if path.is_dir()]
    expected = list(range(44000, 44030))
    if [root["seed"] for root in roots] != expected:
        raise ValueError("development root must contain exactly seeds 44000--44029")
    args.output.mkdir(parents=True, exist_ok=False)
    rows, summary = run_replay(roots)
    source_files = [Path(__file__), Path(__file__).parents[1] / "joint_gaussian_selector_v2.py"]
    summary["source_sha256"] = {str(path): sha256(path) for path in source_files}
    summary["input_sha256"] = {
        str(root["path"] / name): sha256(root["path"] / name)
        for root in roots
        for name in ("features.json", "decisions_frozen.json", "summary.json")
    }
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "decisions.json", rows)
    with (args.output / "decisions.csv").open("w", newline="") as handle:
        fields = [
            "seed", "adaptation", "budget_cap", "method", "display_name", "selected_id",
            "audit_gain", "audit_regret", "query_count", "hf_episode_cost", "zero_evsi_tie_steps",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    print(json.dumps({
        "status": summary["status"],
        "primary_development_contrast": summary["diagnostics"]["primary_development_contrast"],
        "coverage": summary["diagnostics"]["observation_predictive_95_coverage"],
    }, indent=2))


if __name__ == "__main__":
    main()
