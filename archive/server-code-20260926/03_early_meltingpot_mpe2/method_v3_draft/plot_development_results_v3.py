"""Create compact development-only plots from a completed v3 summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    colors = {"v2": "#9aa0a6", "v3": "#2878b5"}
    width = 0.34
    x = np.arange(2)
    for offset, version in ((-width / 2, "v2"), (width / 2, "v3")):
        rows = [row for row in data["held_out_calibration"] if row["posterior"] == version]
        rows.sort(key=lambda row: row["adaptation"])
        means = [row["multivariate_nll_per_root"]["mean"] for row in rows]
        ci = [row["multivariate_nll_per_root"]["ci95"] for row in rows]
        lower = [mean - interval[0] for mean, interval in zip(means, ci)]
        upper = [interval[1] - mean for mean, interval in zip(means, ci)]
        axes[0].bar(x + offset, means, width, label=version.upper(), color=colors[version],
                    yerr=np.asarray([lower, upper]), capsize=3)
    axes[0].set_xticks(x, ["short (4)", "long (32)"])
    axes[0].set_ylabel("held-out multivariate NLL / root")
    axes[0].set_title("Root-LOO predictive calibration")
    axes[0].legend(frameon=False)

    methods = ["joint_gaussian_voi", "uniform_random", "global_ivr",
               "posterior_lucb", "calibrated_no_hf"]
    labels = ["PIVOT-CG-v3", "Uniform", "Global-IVR", "LUCB", "No HF"]
    palette = ["#2878b5", "#e07a2f", "#59a14f", "#b07aa1", "#777777"]
    short = {row["method"]: row for row in data["method_table"]
             if row["adaptation"] == 4 and row["budget_cap"] == 192}
    long = {row["method"]: row for row in data["method_table"]
            if row["adaptation"] == 32 and row["budget_cap"] == 192}
    positions = np.arange(len(methods))
    for shift, table, hatch, label in ((-0.18, short, "", "short (4)"),
                                        (0.18, long, "//", "long (32)")):
        means = [table[method]["audit_regret"]["mean"] for method in methods]
        ci = [table[method]["audit_regret"]["ci95"] for method in methods]
        lower = [mean - interval[0] for mean, interval in zip(means, ci)]
        upper = [interval[1] - mean for mean, interval in zip(means, ci)]
        axes[1].bar(positions + shift, means, 0.34, color=palette, alpha=0.9,
                    hatch=hatch, yerr=np.asarray([lower, upper]), capsize=2,
                    label=label)
    axes[1].set_xticks(positions, labels, rotation=25, ha="right")
    axes[1].set_ylabel("audit simple regret (lower is better)")
    axes[1].set_title("Fixed 192-episode cap")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Melting Pot PIVOT-CG-v3 — development replay only", fontsize=12)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
