"""Development-only decision-opportunity diagnostics for Melting Pot.

This analysis treats each of roots 44000--44029 as one independent grouped
unit.  For a held root, every calibrated posterior quantity is reconstructed
from the other 29 roots with the same cross-fitting code used by the v2
development replay.  The independent A/B audit mean is opened only for
diagnosis and scoring.

The script does not modify a selector, protocol, or any frozen result.  Its
purpose is to distinguish three different facts that raw gap magnitude alone
conflates:

1. whether deployment changes candidate values;
2. whether it changes the identity/ranking of the best update; and
3. whether meaningful posterior mass remains near an argmax boundary after
   calibration, so an additional high-fidelity query can improve selection.

All outputs are development evidence and cannot support a confirmation claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, norm, spearmanr


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from colin_pivot_cloud.joint_gaussian_selector_v2 import exact_affine_evsi
from colin_pivot_cloud.method_v2_draft.development_crossfit_replay import (
    ADAPTATIONS,
    ALL_IDS,
    CANDIDATE_IDS,
    FEATURES,
    fit_model,
    load_root,
    posterior_for_root,
)


VERSION = "melting_decision_opportunity_development_v1"
N_POSTERIOR_DRAWS = 100_000
N_BOOTSTRAP = 20_000
PALETTE = {
    "blue": "#2166AC",
    "orange": "#D6604D",
    "gold": "#B8860B",
    "charcoal": "#2B2B2B",
    "grey": "#8A8A8A",
    "light_grey": "#D9D9D9",
    "pale_blue": "#D8E7F3",
    "pale_orange": "#F4DDD8",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def psd_square_root(covariance: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    return vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))


def top_two_margin(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    return float(ordered[-1] - ordered[-2])


def pairwise_order_agreement(left: np.ndarray, right: np.ndarray) -> float:
    comparisons = []
    for i in range(len(left)):
        for j in range(i):
            comparisons.append(bool((left[i] > left[j]) == (right[i] > right[j])))
    return float(np.mean(comparisons))


def posterior_diagnostics(
    mean: np.ndarray,
    covariance: np.ndarray,
    observation_variance: np.ndarray,
    *,
    seed: int,
) -> dict:
    winner = int(np.argmax(mean))
    order = np.argsort(mean)
    runner = int(order[-2])
    difference_variance = float(
        covariance[winner, winner]
        + covariance[runner, runner]
        - 2.0 * covariance[winner, runner]
    )
    difference_sd = math.sqrt(max(difference_variance, 0.0))
    margin = float(mean[winner] - mean[runner])
    runner_beats_winner = (
        0.0 if difference_sd <= 1e-15 else float(norm.cdf(-margin / difference_sd))
    )

    pairwise_crossing = {}
    for challenger in range(len(mean)):
        if challenger == winner:
            continue
        variance = float(
            covariance[winner, winner]
            + covariance[challenger, challenger]
            - 2.0 * covariance[winner, challenger]
        )
        sd = math.sqrt(max(variance, 0.0))
        crossing = (
            float(mean[challenger] > mean[winner])
            if sd <= 1e-15
            else float(norm.cdf((mean[challenger] - mean[winner]) / sd))
        )
        pairwise_crossing[ALL_IDS[challenger]] = crossing

    rng = np.random.default_rng(seed)
    standard = rng.standard_normal((N_POSTERIOR_DRAWS, len(mean)))
    samples = mean + standard @ psd_square_root(covariance).T
    best = np.argmax(samples, axis=1)
    probabilities = np.bincount(best, minlength=len(mean)) / N_POSTERIOR_DRAWS
    positive = probabilities[probabilities > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    evpi = float(np.mean(np.max(samples, axis=1)) - np.max(mean))

    evsi = np.asarray(
        [
            exact_affine_evsi(mean, covariance, observation_variance, index)
            for index in range(len(CANDIDATE_IDS))
        ]
    )
    best_query = int(np.argmax(evsi))
    return {
        "posterior_mean_winner": ALL_IDS[winner],
        "posterior_mean_runner_up": ALL_IDS[runner],
        "posterior_mean_top_two_margin": margin,
        "runner_up_crossing_probability": runner_beats_winner,
        "max_pairwise_crossing_probability": float(max(pairwise_crossing.values())),
        "pairwise_crossing_probability": pairwise_crossing,
        "probability_mean_winner_is_optimal_mc": float(probabilities[winner]),
        "probability_any_other_is_optimal_mc": float(1.0 - probabilities[winner]),
        "optimal_candidate_probability_mc": {
            identifier: float(probabilities[index])
            for index, identifier in enumerate(ALL_IDS)
        },
        "optimal_candidate_entropy_nats_mc": entropy,
        "expected_value_perfect_information_mc": evpi,
        "best_one_query_id": CANDIDATE_IDS[best_query],
        "best_one_query_evsi": float(evsi[best_query]),
        "all_one_query_evsi": {
            identifier: float(evsi[index])
            for index, identifier in enumerate(CANDIDATE_IDS)
        },
        "best_one_query_evsi_fraction_of_evpi": (
            0.0 if evpi <= 1e-15 else float(evsi[best_query] / evpi)
        ),
        "best_query_observation_noise_variance": float(observation_variance[best_query]),
        "best_query_latent_variance": float(covariance[best_query, best_query]),
        "best_query_noise_to_latent_variance_ratio": (
            math.inf
            if covariance[best_query, best_query] <= 1e-15
            else float(observation_variance[best_query] / covariance[best_query, best_query])
        ),
    }


def bootstrap_mean(values: list[float], seed: int) -> dict:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(len(array), size=(N_BOOTSTRAP, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "n_roots": int(len(array)),
    }


def wilson_interval(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total**2))
        / denominator
    )
    return [center - radius, center + radius]


def summarize_rate(values: list[bool]) -> dict:
    successes = int(sum(values))
    return {
        "count": successes,
        "n_roots": len(values),
        "rate": successes / len(values),
        "wilson_ci95": wilson_interval(successes, len(values)),
    }


def rank_counter(rows: list[dict], key: str, adaptation: int) -> dict:
    counter = Counter(row[key] for row in rows if row["adaptation"] == adaptation)
    return {identifier: int(counter.get(identifier, 0)) for identifier in ALL_IDS}


def aggregate_rows(rows: list[dict]) -> dict:
    aggregate = {}
    numeric_keys = (
        "raw_gap_rms",
        "raw_gap_centered_rms",
        "proxy_deployment_spearman",
        "proxy_deployment_kendall",
        "proxy_deployment_pairwise_order_agreement",
        "proxy_top_two_margin",
        "deployment_top_two_margin",
        "calibrated_top_two_margin",
        "proxy_regret",
        "calibrated_regret",
        "probability_mean_winner_is_optimal_mc",
        "probability_any_other_is_optimal_mc",
        "runner_up_crossing_probability",
        "max_pairwise_crossing_probability",
        "optimal_candidate_entropy_nats_mc",
        "expected_value_perfect_information_mc",
        "best_one_query_evsi",
        "best_one_query_evsi_fraction_of_evpi",
        "best_query_noise_to_latent_variance_ratio",
        "posterior_optimality_calibration_residual",
        "posterior_evpi_minus_realized_calibrated_regret",
        "posterior_optimality_brier_score",
        "posterior_probability_of_true_deployment_winner",
    )
    for adaptation in ADAPTATIONS:
        subset = [row for row in rows if row["adaptation"] == adaptation]
        block = {
            key: bootstrap_mean(
                [float(row[key]) for row in subset],
                seed=2026091700 + adaptation * 100 + index,
            )
            for index, key in enumerate(numeric_keys)
        }
        block.update(
            {
                "proxy_winner_frequency": rank_counter(rows, "proxy_winner", adaptation),
                "deployment_winner_frequency": rank_counter(
                    rows, "deployment_winner", adaptation
                ),
                "calibrated_winner_frequency": rank_counter(
                    rows, "calibrated_winner", adaptation
                ),
                "proxy_deployment_argmax_match": summarize_rate(
                    [row["proxy_deployment_argmax_match"] for row in subset]
                ),
                "calibrated_deployment_argmax_match": summarize_rate(
                    [row["calibrated_deployment_argmax_match"] for row in subset]
                ),
                "proxy_deployment_argmax_changed": summarize_rate(
                    [not row["proxy_deployment_argmax_match"] for row in subset]
                ),
            }
        )
        aggregate[str(adaptation)] = block

    paired = {}
    for key in numeric_keys:
        short = {
            row["seed"]: float(row[key]) for row in rows if row["adaptation"] == 4
        }
        long = {
            row["seed"]: float(row[key]) for row in rows if row["adaptation"] == 32
        }
        paired[key + "_long_minus_short"] = bootstrap_mean(
            [long[seed] - short[seed] for seed in sorted(short)],
            seed=2026091790 + len(paired),
        )
    aggregate["paired_long_minus_short"] = paired
    return aggregate


def method_rows(decisions_path: Path, *, cap: int = 192) -> list[dict]:
    decisions = json.loads(decisions_path.read_text())
    wanted = {
        "proxy_only": "Proxy",
        "calibrated_no_hf": "Calibrated\n(no HF)",
        "joint_gaussian_voi": "PIVOT-CG-v2",
        "uniform_random": "Uniform/Random",
        "global_ivr": "Global-IVR",
        "posterior_lucb": "Posterior LUCB",
    }
    result = []
    for adaptation in ADAPTATIONS:
        for method, label in wanted.items():
            subset = [
                row
                for row in decisions
                if int(row["adaptation"]) == adaptation
                and int(row["budget_cap"]) == cap
                and row["method"] == method
            ]
            values = [float(row["audit_regret"]) for row in subset]
            interval = bootstrap_mean(
                values, seed=2026092300 + adaptation * 10 + len(result)
            )
            result.append(
                {
                    "adaptation": adaptation,
                    "budget_cap": cap,
                    "method": method,
                    "label": label,
                    "audit_regret": interval,
                }
            )
    return result


def method_contrasts(decisions_path: Path, *, cap: int = 192) -> list[dict]:
    decisions = json.loads(decisions_path.read_text())
    result = []
    for adaptation in ADAPTATIONS:
        selected = {
            (int(row["seed"]), row["method"]): float(row["audit_regret"])
            for row in decisions
            if int(row["adaptation"]) == adaptation and int(row["budget_cap"]) == cap
        }
        for comparator in ("calibrated_no_hf", "uniform_random", "global_ivr"):
            values = [
                selected[(seed, comparator)] - selected[(seed, "joint_gaussian_voi")]
                for seed in range(44000, 44030)
            ]
            result.append(
                {
                    "adaptation": adaptation,
                    "budget_cap": cap,
                    "contrast": f"PIVOT-CG-v2 regret reduction vs {comparator}",
                    "positive_favors_pivot": True,
                    "regret_reduction": bootstrap_mean(
                        values, seed=2026092600 + adaptation + len(result)
                    ),
                }
            )
    return result


def transition_matrix(rows: list[dict], adaptation: int, left: str, right: str) -> np.ndarray:
    matrix = np.zeros((len(ALL_IDS), len(ALL_IDS)), dtype=int)
    index = {identifier: position for position, identifier in enumerate(ALL_IDS)}
    for row in rows:
        if row["adaptation"] == adaptation:
            matrix[index[row[left]], index[row[right]]] += 1
    return matrix


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#777777")
    axis.spines["bottom"].set_color("#777777")
    axis.tick_params(colors="#333333", labelsize=8)
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.8, zorder=0)


def save_figures(
    rows: list[dict],
    aggregate: dict,
    methods: list[dict],
    contrasts: list[dict],
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )

    # Figure 1: mechanism-to-decision bridge.
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), constrained_layout=True)
    short = {row["seed"]: row for row in rows if row["adaptation"] == 4}
    long = {row["seed"]: row for row in rows if row["adaptation"] == 32}
    x = np.array([0.0, 1.0])
    for seed in sorted(short):
        axes[0, 0].plot(
            x,
            [short[seed]["raw_gap_rms"], long[seed]["raw_gap_rms"]],
            color=PALETTE["light_grey"],
            linewidth=0.7,
            zorder=1,
        )
    means = [
        aggregate["4"]["raw_gap_rms"]["mean"],
        aggregate["32"]["raw_gap_rms"]["mean"],
    ]
    axes[0, 0].plot(x, means, color=PALETTE["blue"], marker="o", linewidth=2.2, zorder=3)
    axes[0, 0].set_xticks(x, ["Short (4)", "Long (32)"])
    axes[0, 0].set_ylabel("RMS(proxy − deployment), return")
    axes[0, 0].set_title("A. Response strengthens the value gap")
    style_axis(axes[0, 0])

    positions = np.arange(len(ALL_IDS))
    width = 0.36
    for offset, adaptation, color, label in (
        (-width / 2, 4, PALETTE["grey"], "Short (4)"),
        (width / 2, 32, PALETTE["blue"], "Long (32)"),
    ):
        counts = [
            aggregate[str(adaptation)]["deployment_winner_frequency"][identifier]
            for identifier in ALL_IDS
        ]
        axes[0, 1].bar(
            positions + offset,
            np.asarray(counts) / 30.0,
            width,
            color=color,
            label=label,
            zorder=2,
        )
    axes[0, 1].set_xticks(positions, list(ALL_IDS), rotation=0)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_ylabel("Fraction of roots")
    axes[0, 1].set_title("B. Long-response winner concentrates on candidate 7")
    axes[0, 1].legend(frameon=False, fontsize=8)
    style_axis(axes[0, 1])

    metric_labels = ["Argmax\nmatch", "Spearman\nrank ρ", "Pairwise\norder"]
    short_values = [
        aggregate["4"]["proxy_deployment_argmax_match"]["rate"],
        aggregate["4"]["proxy_deployment_spearman"]["mean"],
        aggregate["4"]["proxy_deployment_pairwise_order_agreement"]["mean"],
    ]
    long_values = [
        aggregate["32"]["proxy_deployment_argmax_match"]["rate"],
        aggregate["32"]["proxy_deployment_spearman"]["mean"],
        aggregate["32"]["proxy_deployment_pairwise_order_agreement"]["mean"],
    ]
    metric_x = np.arange(len(metric_labels))
    axes[1, 0].bar(metric_x - width / 2, short_values, width, color=PALETTE["grey"], label="Short")
    axes[1, 0].bar(metric_x + width / 2, long_values, width, color=PALETTE["blue"], label="Long")
    axes[1, 0].axhline(0, color=PALETTE["charcoal"], linewidth=0.8)
    axes[1, 0].set_xticks(metric_x, metric_labels)
    axes[1, 0].set_ylim(-0.55, 1.0)
    axes[1, 0].set_ylabel("Agreement (rank ρ may be negative)")
    axes[1, 0].set_title("C. Raw proxy ranking fails under long response")
    axes[1, 0].legend(frameon=False, fontsize=8)
    style_axis(axes[1, 0])

    for seed in sorted(short):
        axes[1, 1].plot(
            x,
            [short[seed]["deployment_top_two_margin"], long[seed]["deployment_top_two_margin"]],
            color=PALETTE["light_grey"],
            linewidth=0.7,
            zorder=1,
        )
    margin_means = [
        aggregate["4"]["deployment_top_two_margin"]["mean"],
        aggregate["32"]["deployment_top_two_margin"]["mean"],
    ]
    axes[1, 1].plot(x, margin_means, color=PALETTE["orange"], marker="o", linewidth=2.2, zorder=3)
    axes[1, 1].set_xticks(x, ["Short (4)", "Long (32)"])
    axes[1, 1].set_ylabel("Deployment top-two margin, return")
    axes[1, 1].set_title("D. Long response widens the winning margin")
    style_axis(axes[1, 1])

    figure.suptitle(
        "Melting Pot decision opportunity (30 development roots)",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5, -0.015,
        "Development-only • statistical unit: root seed (n=30); grey lines pair the same root",
        ha="center", fontsize=7, color="#555555",
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"figure_1_mechanism_to_decision.{suffix}", bbox_inches="tight")
    plt.close(figure)

    # Figure 2: calibration and finite-budget method opportunity.
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.3), constrained_layout=True)
    for adaptation, axis, title in (
        (4, axes[0, 0], "A. Short response: proxy → deployment winner"),
        (32, axes[0, 1], "B. Long response: proxy → deployment winner"),
    ):
        matrix = transition_matrix(rows, adaptation, "proxy_winner", "deployment_winner")
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
        for i in range(len(ALL_IDS)):
            for j in range(len(ALL_IDS)):
                if matrix[i, j]:
                    axis.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7,
                              color="white" if matrix[i, j] > matrix.max() / 2 else PALETTE["charcoal"])
        axis.set_xticks(range(len(ALL_IDS)), ALL_IDS, fontsize=7)
        axis.set_yticks(range(len(ALL_IDS)), ALL_IDS, fontsize=7)
        axis.set_xlabel("Deployment winner")
        axis.set_ylabel("Proxy winner")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.75, label="Roots")

    chosen_methods = ["proxy_only", "calibrated_no_hf", "joint_gaussian_voi", "uniform_random", "global_ivr"]
    method_labels = {
        "proxy_only": "Proxy",
        "calibrated_no_hf": "Calibrated\n(no HF)",
        "joint_gaussian_voi": "PIVOT-CG-v2",
        "uniform_random": "Uniform/Random",
        "global_ivr": "Global-IVR",
    }
    for adaptation, axis, title in (
        (4, axes[1, 0], "C. Short response, fixed cap = 192 episodes"),
        (32, axes[1, 1], "D. Long response, fixed cap = 192 episodes"),
    ):
        subset = {
            row["method"]: row
            for row in methods
            if row["adaptation"] == adaptation and row["method"] in chosen_methods
        }
        estimates = np.asarray([subset[method]["audit_regret"]["mean"] for method in chosen_methods])
        lows = np.asarray([subset[method]["audit_regret"]["ci95"][0] for method in chosen_methods])
        highs = np.asarray([subset[method]["audit_regret"]["ci95"][1] for method in chosen_methods])
        colors = [PALETTE["grey"], PALETTE["gold"], PALETTE["blue"], PALETTE["orange"], "#6A8E3A"]
        axis.bar(np.arange(len(chosen_methods)), estimates, color=colors, zorder=2)
        axis.errorbar(
            np.arange(len(chosen_methods)),
            estimates,
            yerr=np.vstack([estimates - lows, highs - estimates]),
            fmt="none",
            ecolor=PALETTE["charcoal"],
            capsize=3,
            linewidth=1,
            zorder=3,
        )
        axis.set_xticks(
            np.arange(len(chosen_methods)),
            [method_labels[method] for method in chosen_methods],
            rotation=18,
            ha="right",
        )
        axis.set_ylabel("Deployment selection regret")
        axis.set_title(title)
        style_axis(axis)

    figure.suptitle(
        "Large response gap does not guarantee value from extra HF queries",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5, -0.015,
        "Development-only • statistical unit: root seed (n=30); error bars are root-bootstrap 95% intervals",
        ha="center", fontsize=7, color="#555555",
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"figure_2_method_opportunity.{suffix}", bbox_inches="tight")
    plt.close(figure)

    # Figure 3: model-implied posterior opportunity versus empirical decision opportunity.
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.55), constrained_layout=True)
    metrics = [
        ("probability_any_other_is_optimal_mc", "Posterior P(any challenger optimal)", (0, 1)),
        ("expected_value_perfect_information_mc", "Model EVPI, return", None),
        ("best_one_query_evsi", "Best one-query EVSI, return", None),
    ]
    for axis, (key, label, ylim) in zip(axes, metrics):
        for seed in sorted(short):
            axis.plot(x, [short[seed][key], long[seed][key]], color=PALETTE["light_grey"], linewidth=0.7)
        mean_values = [aggregate["4"][key]["mean"], aggregate["32"][key]["mean"]]
        axis.plot(x, mean_values, color=PALETTE["blue"], marker="o", linewidth=2.2, zorder=3)
        axis.set_xticks(x, ["Short (4)", "Long (32)"])
        axis.set_ylabel(label)
        if ylim:
            axis.set_ylim(*ylim)
        style_axis(axis)
    axes[0].set_title("A. Model boundary uncertainty")
    axes[1].set_title("B. Perfect-information opportunity")
    axes[2].set_title("C. Single-query opportunity")
    figure.suptitle(
        "Cross-fitted model-implied opportunity (development diagnostics)",
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.5, -0.04,
        "Development-only • statistical unit: root seed (n=30); grey lines pair the same held root",
        ha="center", fontsize=7, color="#555555",
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"figure_3_posterior_opportunity.{suffix}", bbox_inches="tight")
    plt.close(figure)

    # Figure 4: model-implied opportunity against held-root realized outcomes.
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.75), constrained_layout=True)
    adaptation_x = np.arange(2)
    labels = ["Short (4)", "Long (32)"]
    width = 0.34

    predicted_error = np.asarray(
        [
            aggregate[str(adaptation)]["probability_any_other_is_optimal_mc"]["mean"]
            for adaptation in ADAPTATIONS
        ]
    )
    observed_error = np.asarray(
        [
            1.0 - aggregate[str(adaptation)]["calibrated_deployment_argmax_match"]["rate"]
            for adaptation in ADAPTATIONS
        ]
    )
    axes[0].bar(adaptation_x - width / 2, predicted_error, width, color=PALETTE["blue"], label="Model-implied")
    axes[0].bar(adaptation_x + width / 2, observed_error, width, color=PALETTE["gold"], label="Held-root observed")
    axes[0].set_xticks(adaptation_x, labels)
    axes[0].set_ylim(0, 0.8)
    axes[0].set_ylabel("Probability chosen arm is not optimal")
    axes[0].set_title("A. Best-arm uncertainty")
    axes[0].legend(frameon=False, fontsize=8)
    style_axis(axes[0])

    model_evpi = np.asarray(
        [
            aggregate[str(adaptation)]["expected_value_perfect_information_mc"]["mean"]
            for adaptation in ADAPTATIONS
        ]
    )
    realized_regret = np.asarray(
        [aggregate[str(adaptation)]["calibrated_regret"]["mean"] for adaptation in ADAPTATIONS]
    )
    axes[1].bar(adaptation_x - width / 2, model_evpi, width, color=PALETTE["blue"], label="Model EVPI")
    axes[1].bar(adaptation_x + width / 2, realized_regret, width, color=PALETTE["gold"], label="Held-root regret")
    axes[1].set_xticks(adaptation_x, labels)
    axes[1].set_ylabel("Return")
    axes[1].set_title("B. Perfect-info opportunity")
    axes[1].legend(frameon=False, fontsize=8)
    style_axis(axes[1])

    best_evsi = np.asarray(
        [aggregate[str(adaptation)]["best_one_query_evsi"]["mean"] for adaptation in ADAPTATIONS]
    )
    contrast_by_adaptation = {
        row["adaptation"]: row["regret_reduction"]["mean"]
        for row in contrasts
        if row["contrast"].endswith("calibrated_no_hf")
    }
    realized_gain = np.asarray([contrast_by_adaptation[adaptation] for adaptation in ADAPTATIONS])
    axes[2].bar(adaptation_x - width / 2, best_evsi, width, color=PALETTE["blue"], label="Best one-query EVSI")
    axes[2].bar(adaptation_x + width / 2, realized_gain, width, color=PALETTE["orange"], label="Observed PIVOT gain†")
    axes[2].axhline(0, color=PALETTE["charcoal"], linewidth=0.8)
    axes[2].set_xticks(adaptation_x, labels)
    axes[2].set_ylabel("Regret reduction / return")
    axes[2].set_title("C. Query value")
    axes[2].legend(frameon=False, fontsize=8)
    style_axis(axes[2])
    axes[2].text(
        0.0,
        -0.28,
        "†Fixed cap 192 vs calibrated no-HF; positive favors PIVOT",
        transform=axes[2].transAxes,
        fontsize=7,
        color="#555555",
    )

    figure.suptitle(
        "The Gaussian model overstates long-response decision opportunity",
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.5, -0.055,
        "Development-only • statistical unit: held-out root seed (n=30)",
        ha="center", fontsize=7, color="#555555",
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"figure_4_model_vs_realized_opportunity.{suffix}", bbox_inches="tight")
    plt.close(figure)


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "seed",
        "adaptation",
        "proxy_winner",
        "deployment_winner",
        "calibrated_winner",
        "proxy_deployment_argmax_match",
        "calibrated_deployment_argmax_match",
        "raw_gap_rms",
        "raw_gap_centered_rms",
        "proxy_deployment_spearman",
        "proxy_deployment_kendall",
        "proxy_deployment_pairwise_order_agreement",
        "proxy_top_two_margin",
        "deployment_top_two_margin",
        "calibrated_top_two_margin",
        "proxy_regret",
        "calibrated_regret",
        "probability_mean_winner_is_optimal_mc",
        "probability_any_other_is_optimal_mc",
        "runner_up_crossing_probability",
        "max_pairwise_crossing_probability",
        "optimal_candidate_entropy_nats_mc",
        "expected_value_perfect_information_mc",
        "best_one_query_id",
        "best_one_query_evsi",
        "best_one_query_evsi_fraction_of_evpi",
        "best_query_noise_to_latent_variance_ratio",
        "posterior_optimality_calibration_residual",
        "posterior_evpi_minus_realized_calibrated_regret",
        "posterior_optimality_brier_score",
        "posterior_probability_of_true_deployment_winner",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def build_report(summary: dict) -> str:
    short = summary["aggregate"]["4"]
    long = summary["aggregate"]["32"]
    paired = summary["aggregate"]["paired_long_minus_short"]
    methods = {
        (row["adaptation"], row["method"]): row["audit_regret"]
        for row in summary["method_results_cap_192"]
    }
    contrasts = {
        (row["adaptation"], row["contrast"].split(" vs ")[-1]): row["regret_reduction"]
        for row in summary["method_contrasts_cap_192"]
    }
    pct = lambda value: f"{100.0 * value:.1f}%"
    ci = lambda value: f"{value['mean']:.3f} [{value['ci95'][0]:.3f}, {value['ci95'][1]:.3f}]"
    lines = [
        "# Melting Pot: response gap versus decision opportunity",
        "",
        "**Status: development-only diagnostic.** Roots 44000–44029 had already been inspected before PIVOT-CG-v2 was designed. Every calibrated quantity below uses an outer leave-one-root-out fit; the held root's A/B audit mean is used only after prediction for diagnosis and scoring. This material can motivate a frozen protocol or theory refinement, but it cannot establish a new method advantage.",
        "",
        "## Main finding",
        "",
        (
            "Stronger opponent response creates a much larger proxy/deployment discrepancy and often changes the raw proxy winner, "
            "but it does not create a hard selection problem in this candidate grid. Under long adaptation, candidate 7 is the true "
            f"deployment winner in {long['deployment_winner_frequency']['7']}/30 roots ({pct(long['deployment_winner_frequency']['7']/30)}), "
            f"and the cross-fitted calibrated no-HF rule selects the true winner in {long['calibrated_deployment_argmax_match']['count']}/30 roots "
            f"({pct(long['calibrated_deployment_argmax_match']['rate'])}). Its mean regret is only {methods[(32, 'calibrated_no_hf')]['mean']:.3f}. "
            "Extra fixed-budget noisy queries therefore have little real decision opportunity and can perturb an already good choice."
        ),
        "",
        "## Concrete results",
        "",
        f"- RMS proxy/deployment gap: short {ci(short['raw_gap_rms'])}; long {ci(long['raw_gap_rms'])}. The paired long-minus-short increase is {ci(paired['raw_gap_rms_long_minus_short'])}.",
        f"- Raw proxy argmax changes at deployment in {short['proxy_deployment_argmax_changed']['count']}/30 short roots ({pct(short['proxy_deployment_argmax_changed']['rate'])}) and {long['proxy_deployment_argmax_changed']['count']}/30 long roots ({pct(long['proxy_deployment_argmax_changed']['rate'])}).",
        f"- Mean proxy/deployment Spearman rank correlation is {short['proxy_deployment_spearman']['mean']:.3f} short and {long['proxy_deployment_spearman']['mean']:.3f} long. Long response therefore changes more than the scale of returns; it reverses much of the raw ordering.",
        f"- Deployment top-two margin grows from {ci(short['deployment_top_two_margin'])} short to {ci(long['deployment_top_two_margin'])} long; paired increase {ci(paired['deployment_top_two_margin_long_minus_short'])}. The best candidate becomes easier, rather than harder, to distinguish across roots.",
        f"- Long, cap 192: proxy regret {methods[(32, 'proxy_only')]['mean']:.3f}; calibrated no-HF {methods[(32, 'calibrated_no_hf')]['mean']:.3f}; PIVOT-CG-v2 {methods[(32, 'joint_gaussian_voi')]['mean']:.3f}; Uniform/Random {methods[(32, 'uniform_random')]['mean']:.3f}; Global-IVR {methods[(32, 'global_ivr')]['mean']:.3f}.",
        f"- Cross-fitted model-implied probability that some challenger beats the posterior-mean winner averages {pct(short['probability_any_other_is_optimal_mc']['mean'])} short and {pct(long['probability_any_other_is_optimal_mc']['mean'])} long. Model EVPI is {short['expected_value_perfect_information_mc']['mean']:.3f} short and {long['expected_value_perfect_information_mc']['mean']:.3f} long; best one-query EVSI is {short['best_one_query_evsi']['mean']:.3f} and {long['best_one_query_evsi']['mean']:.3f}. These are model-implied development diagnostics, not observed method gains.",
        f"- In the long condition, the model assigns the posterior-mean winner only {pct(long['probability_mean_winner_is_optimal_mc']['mean'])} optimality probability, while that choice is actually optimal in {pct(long['calibrated_deployment_argmax_match']['rate'])} of held roots. The held-root minus model probability gap is {100.0 * long['posterior_optimality_calibration_residual']['mean']:.1f} percentage points [{100.0 * long['posterior_optimality_calibration_residual']['ci95'][0]:.1f}, {100.0 * long['posterior_optimality_calibration_residual']['ci95'][1]:.1f}].",
        f"- Long model EVPI exceeds realized calibrated regret by {ci(long['posterior_evpi_minus_realized_calibrated_regret'])}. At cap 192, PIVOT's paired regret reduction versus calibrated no-HF is {ci(contrasts[(32, 'calibrated_no_hf')])}; negative means the forced queries hurt on these development roots.",
        "",
        "## Interpretation for the paper",
        "",
        "The experiment strongly supports the mechanism claim that adaptive deployment can invalidate a proxy ranking. It does not support the stronger claim that a larger raw proxy/deployment gap must make PIVOT more valuable. The missing condition is **decision-relevant boundary uncertainty**: after calibration, posterior mass must remain near competing argmax regions, and an affordable HF observation must be informative enough to move the final choice. A large, systematic response shift can be predictable and can even widen the true top-two margin. In that case the gap is real, while the value of additional selection queries is small.",
        "",
        "This also separates marginal uncertainty calibration from decision calibration. A model can have reasonable candidate-wise predictive coverage yet still put too much posterior mass across the best-arm boundary. Here the long-response Gaussian posterior overstates challenger probability and EVPI relative to held-root winner frequency and regret. That mismatch is directly relevant to a VOI acquisition rule, because it can make low-value queries look useful.",
        "",
        "The honest paper-level use is a boundary result and theory refinement: response strength is necessary for proxy failure, but raw gap magnitude is insufficient to predict VOI. Method-value claims still require fresh confirmation on a benchmark where calibrated candidate rankings remain unstable near the decision boundary.",
        "",
        "## Files",
        "",
        "- `diagnostics.json`: full root-level and aggregate diagnostics, including Monte Carlo posterior winner probabilities and exact one-query EVSI.",
        "- `root_diagnostics.csv`: one row per root × adaptation condition.",
        "- `method_results_cap_192.csv`: root-bootstrap regret summaries for the fixed cap used in the main development comparison.",
        "- `figure_1_mechanism_to_decision.*`: gap, winner frequency, rank agreement, and top-two margin.",
        "- `figure_2_method_opportunity.*`: proxy→deployment winner transitions and cap-192 regret.",
        "- `figure_3_posterior_opportunity.*`: cross-fitted posterior boundary probability, EVPI, and one-query EVSI.",
        "- `figure_4_model_vs_realized_opportunity.*`: model-implied versus held-root best-arm uncertainty and query value.",
        "",
        "## Statistical notes",
        "",
        "- Unit of resampling: root seed (n=30), never candidate or episode.",
        "- Intervals for means: percentile root bootstrap, 20,000 draws. Binary rates use Wilson 95% intervals.",
        f"- Posterior optimum probabilities and EVPI: {N_POSTERIOR_DRAWS:,} deterministic Monte Carlo draws per outer fold. One-query EVSI uses the selector's exact affine-Gaussian upper-envelope calculation.",
        "- Deployment truth: mean of independent audit blocks A and B. The incumbent has gain 0.",
        "- All v1 and formal result files remain unchanged.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--replay-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root_paths = sorted(path for path in args.development_root.glob("seed_44*") if path.is_dir())
    roots = [load_root(path) for path in root_paths]
    if [root["seed"] for root in roots] != list(range(44000, 44030)):
        raise ValueError("development root must contain exactly roots 44000--44029")
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for held in roots:
        training = [root for root in roots if root["seed"] != held["seed"]]
        for adaptation in ADAPTATIONS:
            model = fit_model(training, adaptation)
            mean, covariance, observation_variance = posterior_for_root(held, model)
            proxy = np.r_[held["proxy"], 0.0]
            deployment = np.r_[
                0.5 * (
                    held["audit"][adaptation]["A"]
                    + held["audit"][adaptation]["B"]
                ),
                0.0,
            ]
            proxy_winner = int(np.argmax(proxy))
            deployment_winner = int(np.argmax(deployment))
            calibrated_winner = int(np.argmax(mean))
            correction = deployment[:-1] - proxy[:-1]
            posterior = posterior_diagnostics(
                mean,
                covariance,
                observation_variance,
                seed=held["seed"] * 100 + adaptation,
            )
            optimal_probabilities = posterior["optimal_candidate_probability_mc"]
            observed_optimum = np.zeros(len(ALL_IDS), dtype=float)
            observed_optimum[deployment_winner] = 1.0
            predicted_optimum = np.asarray(
                [optimal_probabilities[identifier] for identifier in ALL_IDS]
            )
            calibrated_regret = float(
                deployment.max() - deployment[calibrated_winner]
            )
            rows.append(
                {
                    "seed": held["seed"],
                    "adaptation": adaptation,
                    "proxy_winner": ALL_IDS[proxy_winner],
                    "deployment_winner": ALL_IDS[deployment_winner],
                    "calibrated_winner": ALL_IDS[calibrated_winner],
                    "proxy_deployment_argmax_match": proxy_winner == deployment_winner,
                    "calibrated_deployment_argmax_match": calibrated_winner == deployment_winner,
                    "raw_gap_rms": float(np.sqrt(np.mean(correction**2))),
                    "raw_gap_centered_rms": float(
                        np.sqrt(np.mean((correction - correction.mean()) ** 2))
                    ),
                    "proxy_deployment_spearman": float(spearmanr(proxy, deployment).statistic),
                    "proxy_deployment_kendall": float(kendalltau(proxy, deployment).statistic),
                    "proxy_deployment_pairwise_order_agreement": pairwise_order_agreement(
                        proxy, deployment
                    ),
                    "proxy_top_two_margin": top_two_margin(proxy),
                    "deployment_top_two_margin": top_two_margin(deployment),
                    "calibrated_top_two_margin": top_two_margin(mean),
                    "proxy_regret": float(deployment.max() - deployment[proxy_winner]),
                    "calibrated_regret": calibrated_regret,
                    "posterior_optimality_calibration_residual": float(
                        (calibrated_winner == deployment_winner)
                        - posterior["probability_mean_winner_is_optimal_mc"]
                    ),
                    "posterior_evpi_minus_realized_calibrated_regret": float(
                        posterior["expected_value_perfect_information_mc"]
                        - calibrated_regret
                    ),
                    "posterior_optimality_brier_score": float(
                        np.sum((predicted_optimum - observed_optimum) ** 2)
                    ),
                    "posterior_probability_of_true_deployment_winner": float(
                        predicted_optimum[deployment_winner]
                    ),
                    **posterior,
                }
            )

    aggregate = aggregate_rows(rows)
    methods = method_rows(args.replay_decisions)
    contrasts = method_contrasts(args.replay_decisions)

    replay = json.loads(args.replay_decisions.read_text())
    replay_lookup = {
        (int(row["seed"]), int(row["adaptation"]), int(row["budget_cap"]), row["method"]): row
        for row in replay
    }
    for row in rows:
        for method, key in (("proxy_only", "proxy_regret"), ("calibrated_no_hf", "calibrated_regret")):
            expected = float(
                replay_lookup[(row["seed"], row["adaptation"], 192, method)]["audit_regret"]
            )
            if not math.isclose(float(row[key]), expected, rel_tol=0.0, abs_tol=1e-10):
                raise AssertionError(f"diagnostic/replay regret mismatch for {row['seed']} {row['adaptation']} {method}")
        probability_sum = sum(row["optimal_candidate_probability_mc"].values())
        if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError("posterior optimum probabilities do not sum to one")
    summary = {
        "status": "complete_development_diagnostic",
        "version": VERSION,
        "claim_boundary": (
            "Roots 44000--44029 were inspected before v2 design. These diagnostics "
            "are development-only and cannot establish a confirmatory method advantage."
        ),
        "independent_unit": "root seed",
        "n_roots": len(roots),
        "development_seeds": [root["seed"] for root in roots],
        "deployment_truth": "mean of independent A/B audit blocks; incumbent gain = 0",
        "posterior_monte_carlo_draws_per_fold": N_POSTERIOR_DRAWS,
        "root_bootstrap_draws": N_BOOTSTRAP,
        "aggregate": aggregate,
        "method_results_cap_192": methods,
        "method_contrasts_cap_192": contrasts,
        "root_rows": rows,
        "source_sha256": {
            str(Path(__file__)): sha256(Path(__file__)),
            str(args.replay_decisions): sha256(args.replay_decisions),
        },
    }
    write_json(args.output / "diagnostics.json", summary)
    write_rows_csv(args.output / "root_diagnostics.csv", rows)
    with (args.output / "method_results_cap_192.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["adaptation", "budget_cap", "method", "mean_regret", "ci95_low", "ci95_high", "n_roots"])
        for row in methods:
            result = row["audit_regret"]
            writer.writerow(
                [
                    row["adaptation"],
                    row["budget_cap"],
                    row["method"],
                    result["mean"],
                    result["ci95"][0],
                    result["ci95"][1],
                    result["n_roots"],
                ]
            )
    save_figures(rows, aggregate, methods, contrasts, args.output)
    (args.output / "REPORT.md").write_text(build_report(summary))
    print(json.dumps({
        "status": summary["status"],
        "output": str(args.output),
        "long_deployment_winner_frequency": aggregate["32"]["deployment_winner_frequency"],
        "long_raw_argmax_changed": aggregate["32"]["proxy_deployment_argmax_changed"],
        "long_calibrated_match": aggregate["32"]["calibrated_deployment_argmax_match"],
        "long_calibrated_regret": next(
            row["audit_regret"] for row in methods
            if row["adaptation"] == 32 and row["method"] == "calibrated_no_hf"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
