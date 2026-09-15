"""Build V9 publication figures and their source-data contracts."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper.figures.style import PALETTE, STYLE_VERSION, apply_publication_style, finalize_figure
from pivot.v9.artifacts import read_jsonl_gz, sha256, write_json


def build(root: Path) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    apply_publication_style()
    output = root / "figures/v9"
    output.mkdir(parents=True, exist_ok=True)
    built: list[dict[str, Any]] = []
    e2c = _preferred_run(root, "e2c")
    e3c = _preferred_run(root, "e3c")
    e4c = _preferred_run(root, "e4c")
    e5c = _preferred_run(root, "e5c")
    e7c = _preferred_run(root, "e7c")
    if e2c:
        summary = json.loads((e2c / "operator_shift_summary.json").read_text(encoding="utf-8"))
        built.append(_e2c(root, output, summary, e2c))
        rows = read_jsonl_gz(e2c / "transition_rows.jsonl.gz")
        built.append(_reversal(root, output, rows, e2c))
    if e3c:
        trajectories = json.loads((e3c / "trajectory_metrics.json").read_text(encoding="utf-8"))
        built.append(_e3c(root, output, trajectories, e3c))
    if e4c:
        reports = json.loads((e4c / "ood_reports.json").read_text(encoding="utf-8"))
        built.append(_e4c(root, output, reports, e4c))
    if e5c:
        frontier = _read_csv(e5c / "efficiency_frontier.csv")
        built.append(_e5c(root, output, frontier, e5c))
        calibration = json.loads((e5c / "calibration_robustness.json").read_text(encoding="utf-8"))
        built.append(_e5c_calibration(root, output, calibration, e5c))
    if e7c:
        summary = json.loads((e7c / "strategic_summary.json").read_text(encoding="utf-8"))
        built.append(_e7c(root, output, summary["by_mode"], e7c))
    write_json(output / "figure_manifest.json", {"style_version": STYLE_VERSION, "figures": built})
    plt.close("all")
    return built


def _preferred_run(root: Path, experiment: str) -> Path | None:
    candidates = [root / f"results/v9/{experiment}-confirmatory", root / f"results/v9/{experiment}-development", root / f"results/v9/{experiment}-smoke"]
    return next((path for path in candidates if path.is_dir()), None)


def _e2c(root: Path, output: Path, summary: dict[str, Any], run: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    rows = list(summary.get("cells", []))
    figure, axes = plt.subplots(1, 3, figsize=(7.3, 2.35))
    for environment in sorted({str(row["environment_id"]) for row in rows}):
        subset = [row for row in rows if str(row["environment_id"]) == environment]
        axes[0].plot([float(row["chi_square_shift"]) for row in subset], [float(row["improvement_fidelity"]) for row in subset], ".", ms=2.8, label=environment.replace("_", " "))
    axes[0].set_xscale("symlog", linthresh=0.01)
    axes[0].set_xlabel(r"operator shift $\chi^2$")
    axes[0].set_ylabel("IDE")
    axes[0].legend(fontsize=6)
    grouped = _mean_by(rows, "operator_shift", "global_policy_spearman")
    axes[1].plot(grouped[0], grouped[1], color=PALETTE["blue_main"], marker="o", ms=3)
    axes[1].set_xlabel("operator shift")
    axes[1].set_ylabel("global rank")
    grouped_irr = _mean_by(rows, "operator_shift", "IRR")
    axes[2].plot(grouped_irr[0], grouped_irr[1], color=PALETTE["red_strong"], marker="o", ms=3)
    axes[2].set_xlabel("operator shift")
    axes[2].set_ylabel("IRR")
    figure.suptitle("E2C: operator shift changes update fidelity", fontsize=10)
    figure.tight_layout(pad=0.7)
    return _save_bundle(root, output, "fig2_e2c_operator_shift", figure, rows, run, "E2C", main_text=True)


def _reversal(root: Path, output: Path, rows: list[dict[str, Any]], run: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    selected = rows[:: max(1, len(rows) // 8000)]
    environments = sorted({str(row["environment_id"]) for row in selected})
    figure, axes = plt.subplots(1, max(len(environments), 1), figsize=(7.1, 2.55), squeeze=False)
    axes_flat = list(axes[0])
    for axis, environment in zip(axes_flat, environments):
        subset = [row for row in selected if str(row["environment_id"]) == environment]
        proxy = np.asarray([float(row["delta_proxy"]) for row in subset])
        true = np.asarray([float(row["delta_true"]) for row in subset])
        reversal = (proxy > 0) & (true < 0)
        axis.scatter(proxy[~reversal], true[~reversal], s=4, alpha=0.22, color=PALETTE["blue_secondary"], label="other")
        axis.scatter(proxy[reversal], true[reversal], s=8, alpha=0.65, color=PALETTE["red_strong"], label="reversal")
        lower = float(min(proxy.min(), true.min()))
        upper = float(max(proxy.max(), true.max()))
        span = max(upper - lower, 1e-6)
        axis.set_xlim(lower - 0.04 * span, upper + 0.04 * span)
        axis.set_ylim(lower - 0.04 * span, upper + 0.04 * span)
        axis.plot([lower, upper], [lower, upper], color=PALETTE["neutral"], lw=0.8)
        axis.axhline(0, color=PALETTE["text"], lw=0.5)
        axis.axvline(0, color=PALETTE["text"], lw=0.5)
        axis.set_title(environment.replace("_", " "), fontsize=8)
        axis.set_xlabel(r"proxy $\Delta_V$")
        axis.grid(alpha=0.12)
    axes_flat[0].set_ylabel(r"deployment $\Delta_*$")
    axes_flat[0].legend(fontsize=6, loc="upper left")
    figure.suptitle("E2C: verifier-approved updates can reverse after deployment", fontsize=10)
    figure.tight_layout(pad=0.7)
    return _save_bundle(root, output, "fig1_v9_improvement_reversal", figure, selected, run, "E2C", main_text=True)


def _e3c(root: Path, output: Path, trajectories: list[dict[str, Any]], run: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    # The native MPE2 result is deliberately frozen from V7 rather than
    # rerun: the V9 protocol says to preserve that powered external null.  It
    # is joined only at the figure layer, with provenance fields that prevent
    # it being mistaken for a new V9 confirmatory estimate.
    frozen_path = root / "results/v7/e3b-confirmatory/trajectory_metrics.json"
    frozen_rows: list[dict[str, Any]] = []
    if frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if isinstance(frozen, list):
            for method_name, method_id in (("Proxy Only", "proxy_only"), ("PIVOT-VOI", "pivot_voi")):
                subset = [row for row in frozen if str(row.get("trajectory_id", "")).endswith(f"method={method_name}")]
                if subset:
                    frozen_rows.append(
                        {
                            "environment_id": "mpe2_frozen_external_null",
                            "method": method_id,
                            "CTI": float(np.mean([float(row["cti"]) for row in subset])),
                            "CISR": float(np.mean([float(row["cisr"]) for row in subset])),
                            "external_null": True,
                            "source_status": "HYPOTHESIS_NOT_SUPPORTED",
                            "source_trajectories": len(subset),
                        }
                    )
    display_rows = [dict(row, external_null=False) for row in trajectories] + frozen_rows
    environments = ["mpe2_frozen_external_null", "performative_control", "congestion_resource"]
    methods = sorted({str(row["method"]) for row in display_rows})
    colors = {method: (PALETTE["blue_main"] if method == "pivot_voi" else PALETTE["neutral"]) for method in methods}
    figure, axes = plt.subplots(len(environments), 2, figsize=(7.0, 5.35), sharex="col")
    for row_index, environment in enumerate(environments):
        for method in methods:
            subset = [row for row in display_rows if str(row["environment_id"]) == environment and str(row["method"]) == method]
            if not subset:
                continue
            marker = "D" if bool(subset[0].get("external_null")) else "o"
            axes[row_index, 0].scatter(method, np.mean([float(row["CTI"]) for row in subset]), color=colors[method], marker=marker, s=18, zorder=3)
            axes[row_index, 1].scatter(method, np.mean([float(row["CISR"]) for row in subset]), color=colors[method], marker=marker, s=18, zorder=3)
        axes[row_index, 0].set_ylabel(f"{environment.replace('_', ' ')}\nCTI", fontsize=7)
        axes[row_index, 1].set_ylabel("CISR", fontsize=7)
        for axis in axes[row_index]:
            axis.grid(axis="y", alpha=0.2)
            axis.tick_params(axis="x", labelrotation=70, labelsize=5)
    axes[-1, 0].set_xlabel("method")
    axes[-1, 1].set_xlabel("method")
    figure.suptitle("E3C: closed-loop outcomes at matched query budgets", fontsize=10)
    figure.tight_layout(pad=0.7, h_pad=0.8)
    return _save_bundle(
        root,
        output,
        "fig4_e3c_closed_loop",
        figure,
        display_rows,
        run,
        "E3C",
        main_text=True,
        extra_inputs=[frozen_path] if frozen_path.is_file() else [],
    )


def _e4c(root: Path, output: Path, reports: list[dict[str, Any]], run: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(6.3, 2.35))
    splits = sorted({str(row["split"]) for row in reports})
    families = sorted({str(row["family"]) for row in reports})
    x = np.arange(len(splits))
    width = 0.8 / max(len(families), 1)
    for index, family in enumerate(families):
        subset = {str(row["split"]): row for row in reports if str(row["family"]) == family}
        axes[0].bar(x + index * width, [float(subset[split]["transition_ISC"]) for split in splits], width, label=family)
        axes[1].bar(x + index * width, [float(subset[split]["transition_IDE"]) for split in splits], width, label=family)
    for axis, label in zip(axes, ("transition ISC", "transition IDE")):
        axis.set_xticks(x + width * max(len(families) - 1, 0) / 2, splits, rotation=35, ha="right", fontsize=7)
        axis.set_ylabel(label)
    axes[0].legend(fontsize=6)
    figure.suptitle("E4C: differential evaluator OOD diagnostics", fontsize=10)
    figure.tight_layout(pad=0.65)
    return _save_bundle(root, output, "fig6_e4c_ood", figure, reports, run, "E4C", main_text=False)


def _e5c(root: Path, output: Path, frontier: list[dict[str, Any]], run: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(3.8, 2.8))
    for method in sorted({str(row["method"]) for row in frontier}):
        subset = sorted([row for row in frontier if str(row["method"]) == method], key=lambda row: float(row["mean_hf_cost"]))
        axis.plot([float(row["mean_hf_cost"]) for row in subset], [float(row["mean_CISR"]) for row in subset], marker="o", ms=2.5, lw=0.8, label=method)
    axis.set_xlabel("mean HF cost")
    axis.set_ylabel("CISR")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=5, ncol=2)
    figure.suptitle("E5C: evidence-efficiency frontier", fontsize=10)
    figure.tight_layout(pad=0.65)
    return _save_bundle(root, output, "fig5_e5c_budget_frontier", figure, frontier, run, "E5C", main_text=True)


def _e5c_calibration(root: Path, output: Path, calibration: dict[str, Any], run: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    rows = [{"section": "posterior", **row} for row in calibration.get("posterior_sample_stability", [])]
    rows.extend({"section": "cost", **row} for row in calibration.get("cost_misspecification", []))
    figure, axes = plt.subplots(1, 2, figsize=(6.1, 2.35))
    posterior_rows = [row for row in rows if row["section"] == "posterior"]
    cost_rows = [row for row in rows if row["section"] == "cost"]
    axes[0].plot([float(row["posterior_samples"]) for row in posterior_rows], [float(row["selected_set_jaccard"]) for row in posterior_rows], marker="o", color=PALETTE["teal"])
    axes[0].set_xlabel("posterior samples")
    axes[0].set_ylabel("query-set Jaccard")
    axes[1].plot([float(row["cost_multiplier"]) for row in cost_rows], [float(row["selected_query_jaccard"]) for row in cost_rows], marker="o", color=PALETTE["violet"])
    axes[1].set_xlabel("cost multiplier")
    axes[1].set_ylabel("query-set Jaccard")
    figure.suptitle("E5C calibration robustness", fontsize=10)
    figure.tight_layout(pad=0.65)
    return _save_bundle(root, output, "fig8_e5c_calibration", figure, rows, run, "E5C", main_text=False)


def _e7c(root: Path, output: Path, rows: list[dict[str, Any]], run: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    ordered = ["fixed", "reactive", "best_response", "gradient_adaptive", "rl_evolutionary"]
    lookup = {str(row["opponent_mode"]): row for row in rows}
    figure, axis = plt.subplots(figsize=(4.2, 2.8))
    means = [float(lookup[mode]["mean_strategic_effect"]) for mode in ordered if mode in lookup]
    lows = [float(lookup[mode]["strategic_effect_ci_low"]) for mode in ordered if mode in lookup]
    highs = [float(lookup[mode]["strategic_effect_ci_high"]) for mode in ordered if mode in lookup]
    labels = [mode for mode in ordered if mode in lookup]
    axis.errorbar(np.arange(len(labels)), means, yerr=[np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)], fmt="o", color=PALETTE["red_strong"], capsize=3)
    axis.axhline(0, color=PALETTE["neutral"], lw=0.8)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right", fontsize=7)
    axis.set_ylabel(r"$\Delta_{strategic}-\Delta_{actor}$")
    axis.set_title("E7C: strategic response layer")
    figure.tight_layout(pad=0.65)
    return _save_bundle(root, output, "fig7_e7c_strategic", figure, rows, run, "E7C", main_text=False)


def _mean_by(rows: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    grouped: dict[float, list[float]] = {}
    for row in rows:
        grouped.setdefault(float(row[x_key]), []).append(float(row[y_key]))
    keys = sorted(grouped)
    return keys, [float(np.mean(grouped[key])) for key in keys]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _save_bundle(
    root: Path,
    output: Path,
    stem: str,
    figure: Any,
    rows: list[dict[str, Any]],
    run: Path,
    experiment: str,
    *,
    main_text: bool,
    extra_inputs: list[Path] | None = None,
) -> dict[str, Any]:
    source = output / f"{stem}.csv"
    _write_csv(source, rows)
    parquet = output / f"{stem}.parquet"
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist([{str(key): value for key, value in row.items()} for row in rows])
        pq.write_table(table, parquet)
    except (ImportError, ValueError, TypeError):
        parquet.write_text("parquet unavailable; use adjacent CSV source\n", encoding="utf-8")
    paths = finalize_figure(figure, output / stem, formats=("pdf", "svg", "png"), dpi=300)
    script = Path(__file__).resolve()
    input_paths = [path for path in sorted(run.glob("*")) if path.is_file()]
    input_paths.extend(path for path in (extra_inputs or []) if path.is_file())
    input_paths = sorted(set(input_paths))
    metadata = {
        "figure_id": stem,
        "experiment_ids": [experiment],
        "main_text": main_text,
        "input_files": [str(path.relative_to(root)) for path in input_paths],
        "input_sha256": {str(path.relative_to(root)): sha256(path) for path in input_paths},
        "analysis_script": str(script.relative_to(root)),
        "analysis_script_sha256": sha256(script),
        "style_version": STYLE_VERSION,
        "git_commit": _git_commit(root),
        "source_csv": source.name,
        "source_parquet": parquet.name,
        "output_files": [path.name for path in paths],
    }
    write_json(output / f"{stem}.meta.json", metadata)
    return metadata


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PIVOT V9 figures")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    built = build(args.root.resolve())
    print(json.dumps({"figures": [item["figure_id"] for item in built], "count": len(built)}, sort_keys=True))


if __name__ == "__main__":
    main()
