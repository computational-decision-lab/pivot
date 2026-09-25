#!/usr/bin/env python3
"""Create paper-ready figures from the compact sealed OpenSpiel result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)

COLORS = {
    "short": "#4C78A8",
    "long": "#E07B39",
    "pivot_cg_v2": "#2563A6",
    "uniform_random": "#6B7280",
    "posterior_lucb": "#7C3AED",
    "global_ivr": "#0F766E",
    "calibrated_no_hf": "#111827",
    "proxy_only": "#9CA3AF",
    "author_global_voi_e5c_adapter": "#C2410C",
    "all_hf_raw_reference": "#D97706",
}

LABELS = {
    "pivot_cg_v2": "PIVOT-CG-v2",
    "uniform_random": "Uniform / Random HF",
    "posterior_lucb": "Posterior LUCB",
    "global_ivr": "Global-IVR",
    "calibrated_no_hf": "Calibrated no-HF",
    "proxy_only": "Proxy only",
    "author_global_voi_e5c_adapter": "Author Global-VOI adapter",
    "all_hf_raw_reference": "All-HF raw reference",
}


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def bootstrap_mean(values: np.ndarray, *parts: object, draws: int = 5000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(stable_seed("figure", *parts))
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10,
            "axes.edgecolor": "#9CA3AF",
            "axes.linewidth": 0.8,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "text.color": "#111827",
            "axes.titleweight": "bold",
            "figure.titlesize": 14,
            "figure.titleweight": "bold",
            "legend.frameon": False,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.7,
        }
    )


def mechanism_figure(mechanism: pd.DataFrame, compact: dict) -> None:
    short = mechanism[mechanism.horizon == "short"].sort_values("root_seed")
    long = mechanism[mechanism.horizon == "long"].sort_values("root_seed")
    assert np.array_equal(short.root_seed.to_numpy(), long.root_seed.to_numpy())

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    fig.suptitle("Stronger opponent response breaks proxy-based update selection", y=1.04)

    ax = axes[0]
    rng = np.random.default_rng(17)
    for x, (name, data) in enumerate((("Short (T=1)", short), ("Long (T=8)", long))):
        values = data.mean_abs_proxy_deployment_gap.to_numpy()
        jitter = rng.uniform(-0.09, 0.09, len(values))
        ax.scatter(np.full(len(values), x) + jitter, values, s=13, alpha=0.23, color=COLORS[name.split()[0].lower()], edgecolors="none")
        mean, low, high = bootstrap_mean(values, "gap", name)
        ax.errorbar(x, mean, yerr=[[mean - low], [high - mean]], fmt="o", ms=7, lw=2.2, capsize=4, color="#111827", zorder=5)
        ax.text(x, high + 0.008, f"mean {mean:.3f}", ha="center", va="bottom", fontsize=9)
    diff = compact["mechanism"]["long_minus_short"]["mean_abs_gap"]
    ax.set_xticks([0, 1], ["Short\nT=1", "Long\nT=8"])
    ax.set_ylabel("Mean |proxy − deployment| per root")
    ax.set_title("A. Proxy/deployment gap grows 6.46×")
    ax.grid(axis="y")
    ax.text(0.02, 0.98, f"Paired Δ={diff['mean']:.3f}\n95% CI [{diff['ci_low']:.3f}, {diff['ci_high']:.3f}]", transform=ax.transAxes, va="top", fontsize=8.8, bbox=dict(boxstyle="round,pad=.3", fc="white", ec="#D1D5DB"))

    ax = axes[1]
    rates = np.array([short.winner_mismatch.mean(), long.winner_mismatch.mean()])
    lowers, uppers = [], []
    for name, values in (("short", short.winner_mismatch.to_numpy()), ("long", long.winner_mismatch.to_numpy())):
        mean, low, high = bootstrap_mean(values, "mismatch", name)
        lowers.append(mean - low)
        uppers.append(high - mean)
    bars = ax.bar([0, 1], rates, color=[COLORS["short"], COLORS["long"]], width=0.62)
    ax.errorbar([0, 1], rates, yerr=[lowers, uppers], fmt="none", color="#111827", capsize=4, lw=1.5)
    for bar, value in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.018, f"{value:.0%}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1], ["Short\nT=1", "Long\nT=8"])
    ax.set_ylim(0, 0.42)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Proxy winner differs from deployment winner")
    ax.set_title("B. Winner mismatch rises from 1% to 30%")
    ax.grid(axis="y")

    ax = axes[2]
    values_short = short.proxy_best_selection_regret.to_numpy()
    values_long = long.proxy_best_selection_regret.to_numpy()
    means = []
    errors = [[], []]
    for name, values in (("short", values_short), ("long", values_long)):
        mean, low, high = bootstrap_mean(values, "proxy_regret", name)
        means.append(mean)
        errors[0].append(mean-low); errors[1].append(high-mean)
    bars = ax.bar([0, 1], means, color=[COLORS["short"], COLORS["long"]], width=0.62)
    ax.errorbar([0, 1], means, yerr=errors, fmt="none", color="#111827", capsize=4, lw=1.5)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.001, f"{value:.4f}", ha="center", fontsize=9, fontweight="bold")
    diff = compact["mechanism"]["long_minus_short"]["proxy_best_selection_regret"]
    ax.set_xticks([0, 1], ["Short\nT=1", "Long\nT=8"])
    ax.set_ylabel("Regret of proxy-selected update")
    ax.set_title("C. Proxy selection becomes costly")
    ax.grid(axis="y")
    ax.text(0.02, 0.98, f"Paired Δ={diff['mean']:.4f}\n95% CI [{diff['ci_low']:.4f}, {diff['ci_high']:.4f}]", transform=ax.transAxes, va="top", fontsize=8.8, bbox=dict(boxstyle="round,pad=.3", fc="white", ec="#D1D5DB"))
    fig.text(0.01, -0.04, "100 fresh roots (71000–71099). Error bars: 95% root-bootstrap intervals; paired differences use the frozen confirmatory audit.", fontsize=8.5, color="#4B5563")
    fig.tight_layout()
    save(fig, "figure_1_mechanism_chain")


def method_figure(summary: pd.DataFrame) -> None:
    long = summary[summary.horizon == "long"].copy()
    budgets = [0, 64, 128, 256, 512]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.35))
    fig.suptitle("Finite HF helps only when the acquisition and estimator use it well", y=1.04)

    ax = axes[0]
    core = ["pivot_cg_v2", "uniform_random", "posterior_lucb", "global_ivr", "calibrated_no_hf"]
    styles = {
        "pivot_cg_v2": ("o", "-", 2.4),
        "uniform_random": ("s", "--", 1.8),
        "posterior_lucb": ("^", "-.", 1.7),
        "global_ivr": ("D", ":", 1.8),
        "calibrated_no_hf": (None, "--", 1.5),
    }
    for method in core:
        data = long[long.method == method].sort_values("budget_paired_hands")
        marker, linestyle, width = styles[method]
        x = data.budget_paired_hands.to_numpy()
        y = data.mean_selection_regret.to_numpy()
        low = data.regret_ci_low.to_numpy(); high = data.regret_ci_high.to_numpy()
        ax.plot(x, y, marker=marker, ls=linestyle, lw=width, ms=5, color=COLORS[method], label=LABELS[method])
        if method in {"pivot_cg_v2", "uniform_random"}:
            ax.fill_between(x, low, high, color=COLORS[method], alpha=0.10, linewidth=0)
    ax.axvline(128, color="#CBD5E1", lw=1, zorder=0)
    ax.text(128, 0.0047, "primary budget", ha="center", va="top", fontsize=8, color="#64748B", rotation=90)
    ax.set_xticks(budgets)
    ax.set_xlabel("HF budget (paired hands)")
    ax.set_ylabel("Mean selection regret")
    ax.set_ylim(0, 0.005)
    ax.set_title("A. Core selectors, long response (zoom)")
    ax.grid(axis="y")
    ax.legend(fontsize=8, loc="upper left", ncol=2)

    ax = axes[1]
    methods = ["pivot_cg_v2", "proxy_only", "author_global_voi_e5c_adapter", "all_hf_raw_reference"]
    for method in methods:
        data = long[long.method == method].sort_values("budget_paired_hands")
        if data.empty:
            continue
        x = data.budget_paired_hands.to_numpy(); y = data.mean_selection_regret.to_numpy()
        low = data.regret_ci_low.to_numpy(); high = data.regret_ci_high.to_numpy()
        if method in {"proxy_only"}:
            ax.plot(x, y, ls="--", lw=1.7, color=COLORS[method], label=LABELS[method])
        elif method == "all_hf_raw_reference":
            ax.errorbar(x, y, yerr=[y-low, high-y], fmt="X", ms=8, capsize=4, color=COLORS[method], label=LABELS[method])
        else:
            ax.plot(x, y, marker="o", lw=2 if method=="pivot_cg_v2" else 1.8, color=COLORS[method], label=LABELS[method])
            ax.fill_between(x, low, high, color=COLORS[method], alpha=0.08, linewidth=0)
    ax.set_xticks(budgets)
    ax.set_xlabel("HF budget (paired hands)")
    ax.set_ylabel("Mean selection regret")
    ax.set_title("B. Raw/noisy HF can degrade selection")
    ax.grid(axis="y")
    ax.legend(fontsize=8, loc="upper left")
    fig.text(0.01, -0.04, "Long-response setting, 100 fresh roots. Shading/error bars: descriptive 95% root-bootstrap intervals. Random allocations are averaged within root.", fontsize=8.5, color="#4B5563")
    fig.tight_layout()
    save(fig, "figure_2_method_budget")


def contrast_figure(compact: dict) -> None:
    contrasts = compact["predeclared_contrasts"]
    rows = [
        ("Short: Random − PIVOT", contrasts["random_minus_pivot_short_budget_128"]),
        ("Long: Random − PIVOT", contrasts["random_minus_pivot_long_budget_128"]),
        ("Long-minus-short interaction", contrasts["pivot_advantage_long_minus_short_interaction"]),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 4.1))
    y = np.arange(len(rows))[::-1]
    for yi, (label, row) in zip(y, rows):
        mean, low, high = row["mean"], row["ci_low"], row["ci_high"]
        ax.errorbar(mean, yi, xerr=[[mean-low], [high-mean]], fmt="o", ms=7, capsize=4, lw=2, color="#2563A6")
        ax.text(high + 0.00010, yi, f"{mean:+.6f}  [{low:+.6f}, {high:+.6f}]", va="center", fontsize=9)
    ax.axvline(0, color="#111827", lw=1)
    ax.set_yticks(y, [label for label, _ in rows])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 1_000:.1f}"))
    ax.set_xticks([-0.0005, 0.0, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030])
    ax.set_xlabel("Paired regret difference ×10⁻³ (positive favors PIVOT)")
    ax.set_title("Registered primary contrast is directionally positive but inconclusive")
    ax.set_xlim(-0.0008, 0.0032)
    ax.grid(axis="x")
    ax.text(0.02, -0.28, "Budget = 128 paired hands; 100 fresh roots; 95% paired root-bootstrap intervals. All intervals touching/crossing zero fail the frozen support rule.", transform=ax.transAxes, fontsize=8.5, color="#4B5563")
    fig.tight_layout()
    save(fig, "figure_3_primary_contrast")


def diagnostic_figure(method_audit: dict) -> None:
    diag = method_audit["acquisition_diagnostics"]
    budgets = [64, 128, 256, 512]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    fig.suptitle("Why the formal PIVOT advantage did not materialize", y=1.04)

    ax = axes[0]
    for horizon in ("short", "long"):
        rates = [diag[f"{horizon}_budget_{budget}"]["all_evsi_zero_step_rate"] for budget in budgets]
        ax.plot(budgets, rates, marker="o", lw=2, color=COLORS[horizon], label=f"{horizon.title()} response")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xticks(budgets)
    ax.set_xlabel("HF budget (paired hands)")
    ax.set_ylabel("PIVOT query steps with max EVSI ≈ 0")
    ax.set_title("A. Decision-relevant VOI often vanishes")
    ax.grid(axis="y")
    ax.legend(fontsize=8)

    ax = axes[1]
    noise = method_audit["calibrated_noise_scale"]
    ratios = [noise["short"]["median_query_sd_over_median_latent_sd"], noise["long"]["median_query_sd_over_median_latent_sd"]]
    bars = ax.bar([0, 1], ratios, width=.62, color=[COLORS["short"], COLORS["long"]])
    for bar, value in zip(bars, ratios):
        ax.text(bar.get_x()+bar.get_width()/2, value+1, f"{value:.1f}×", ha="center", fontweight="bold")
    ax.set_xticks([0, 1], ["Short\nT=1", "Long\nT=8"])
    ax.set_ylabel("Median query SD / median latent SD")
    ax.set_title("B. One HF query is noisy")
    ax.grid(axis="y")

    ax = axes[2]
    same_path = [diag[f"long_budget_{budget}"]["pivot_global_ivr_identical_query_path_rate"] for budget in budgets]
    same_choice = [diag[f"long_budget_{budget}"]["pivot_global_ivr_identical_selection_rate"] for budget in budgets]
    ax.plot(budgets, same_path, marker="s", ls="--", lw=1.8, color="#6B7280", label="Identical query path")
    ax.plot(budgets, same_choice, marker="o", lw=2.2, color=COLORS["pivot_cg_v2"], label="Identical final selection")
    ax.set_ylim(0.78, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xticks(budgets)
    ax.set_xlabel("HF budget (paired hands)")
    ax.set_ylabel("PIVOT vs Global-IVR agreement")
    ax.set_title("C. Different paths, identical decisions")
    ax.grid(axis="y")
    ax.legend(fontsize=8, loc="lower left")
    fig.text(0.01, -0.04, "Post-audit diagnostics; these explain the frozen null result and are not additional confirmatory tests.", fontsize=8.5, color="#4B5563")
    fig.tight_layout()
    save(fig, "figure_4_method_diagnostics")


def main() -> None:
    style()
    mechanism = pd.read_csv(HERE / "mechanism_by_root.csv")
    summary = pd.read_csv(HERE / "method_budget_summary.csv")
    compact = json.loads((HERE / "formal_compact_summary.json").read_text())
    method_audit = json.loads((HERE / "method_audit.json").read_text())
    mechanism_figure(mechanism, compact)
    method_figure(summary)
    contrast_figure(compact)
    diagnostic_figure(method_audit)
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
