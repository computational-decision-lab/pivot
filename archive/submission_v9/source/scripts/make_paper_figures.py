#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.validation import validate_figure_bundle

try:
    from scripts.figure_style import (
        DEFAULT_COLORS,
        PALETTE,
        FigureStyle,
        apply_publication_style,
        finalize_figure,
        reversal_cmap,
    )
except ModuleNotFoundError:  # Direct ``python scripts/make_paper_figures.py`` entry.
    from figure_style import (  # type: ignore[no-redef]
        DEFAULT_COLORS,
        PALETTE,
        FigureStyle,
        apply_publication_style,
        finalize_figure,
        reversal_cmap,
    )

STEMS = (
    "fig1_when_better_gets_worse",
    "fig2_reversal_phase_diagram",
    "fig3_optimizing_wrong_world",
    "fig4_policy_vs_improvement_fidelity",
    "fig5_pivot_budget_frontier",
    "fig6_observer_actor_strategic",
    "fig7_strategic_reversal",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical PIVOT paper figure artifacts")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contrast",
        type=Path,
        help="Optional E4 value-versus-improvement comparison.json used for Figure 4",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    apply_publication_style(FigureStyle(font_size=10, axes_linewidth=1.2))
    _build_scatter_figure(args.input, args.output)
    _build_heatmap_figure(args.input, args.output)
    _build_overoptimization(args.input, args.output)
    _build_global_vs_local(args.input, args.output, contrast=args.contrast)
    _build_budget_frontier(args.input, args.output)
    _build_strategic_layers(args.input, args.output)
    _build_competition(args.input, args.output)
    validation = validate_figure_bundle(args.output, STEMS)
    (args.output / "figure_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    if not validation["valid"]:
        errors = cast(list[str], validation["errors"])
        raise SystemExit("figure validation failed: " + "; ".join(errors))
    checked = cast(list[object], validation["checked"])
    print(f"validated figures={len(checked)} output={args.output}")


def _find(input_dir: Path, filename: str) -> Path | None:
    direct = input_dir / filename
    if direct.exists():
        return direct
    matches = sorted(input_dir.rglob(filename))
    return matches[0] if matches else None


def _copy_or_unavailable(input_dir: Path, output_dir: Path, stem: str, source_stem: str) -> None:
    png = _find(input_dir, f"{source_stem}.png")
    csv_path = _find(input_dir, f"{source_stem}.csv")
    if png is None or csv_path is None:
        (output_dir / f"{stem}.unavailable").write_text(
            f"Source artifacts {source_stem}.png/.csv were not found.\n", encoding="utf-8"
        )
        return
    shutil.copyfile(png, output_dir / f"{stem}.png")
    shutil.copyfile(csv_path, output_dir / f"{stem}.csv")


def _build_scatter_figure(input_dir: Path, output_dir: Path) -> None:
    """Regenerate Figure 1 from its source table with semantic labels."""

    source = _find(input_dir, "proxy_vs_true_scatter.csv")
    if source is None:
        _copy_or_unavailable(input_dir, output_dir, "fig1_when_better_gets_worse", "proxy_vs_true_scatter")
        return
    rows = _read_csv(source)
    shutil.copyfile(source, output_dir / "fig1_when_better_gets_worse.csv")
    _semantic_scatter_png(output_dir / "fig1_when_better_gets_worse.png", rows)


def _build_heatmap_figure(input_dir: Path, output_dir: Path) -> None:
    """Regenerate Figure 2 as a continuous heatmap with a boundary contour."""

    source = _find(input_dir, "response_footprint_heatmap.csv")
    if source is None:
        _copy_or_unavailable(input_dir, output_dir, "fig2_reversal_phase_diagram", "response_footprint_heatmap")
        return
    rows = _read_csv(source)
    shutil.copyfile(source, output_dir / "fig2_reversal_phase_diagram.csv")
    _semantic_heatmap_png(output_dir / "fig2_reversal_phase_diagram.png", rows)


def _build_overoptimization(input_dir: Path, output_dir: Path) -> None:
    source = _find(input_dir, "overoptimization.json")
    if source is None:
        (output_dir / "fig3_optimizing_wrong_world.unavailable").write_text(
            "E3 overoptimization output was not found.\n", encoding="utf-8"
        )
        return
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        (output_dir / "fig3_optimizing_wrong_world.unavailable").write_text(
            "E3 output contained no rows.\n", encoding="utf-8"
        )
        return
    csv_path = output_dir / "fig3_optimizing_wrong_world.csv"
    _write_csv(csv_path, rows)
    _line_png(output_dir / "fig3_optimizing_wrong_world.png", rows)


def _build_global_vs_local(input_dir: Path, output_dir: Path, *, contrast: Path | None = None) -> None:
    source = _find(input_dir, "local_predictions.csv")
    stem = "fig4_policy_vs_improvement_fidelity"
    if contrast is not None:
        _build_contrast_figure(contrast, output_dir, stem)
        return
    if source is None:
        _mark_unavailable(output_dir, stem, "E4 local_predictions.csv was not found.")
        return
    rows = _read_csv(source)
    shutil.copyfile(source, output_dir / f"{stem}.csv")
    _scatter_png(
        output_dir / f"{stem}.png",
        rows,
        x_key="delta_true",
        y_keys=("delta_proxy", "predicted_delta"),
        title="Policy versus improvement fidelity",
    )


def _build_contrast_figure(contrast: Path, output_dir: Path, stem: str) -> None:
    """Render the controlled evaluator contrast as the central E4 figure."""

    payload = json.loads(contrast.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
        raise TypeError(f"invalid evaluator contrast payload: {contrast}")
    metrics = payload["metrics"]
    rows = []
    for evaluator in ("value_fidelity", "transition_fidelity"):
        entry = metrics.get(evaluator)
        if not isinstance(entry, dict):
            raise TypeError(f"missing evaluator metrics: {evaluator}")
        rows.append(
            {
                "evaluator": evaluator,
                "policy_value_mae": entry["policy_value_mae"],
                "policy_rank_correlation": entry["policy_rank_correlation"],
                "improvement_differential_error": entry["improvement_differential_error"],
                "improvement_sign_consistency": entry["improvement_sign_consistency"],
                "improvement_reversal_rate": entry["improvement_reversal_rate"],
                "update_selection_regret": entry["update_selection_regret"],
            }
        )
    _write_csv(output_dir / f"{stem}.csv", rows)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _fallback_png(output_dir / f"{stem}.png")
        return
    labels = ["value fidelity\n(A)", "transition fidelity\n(B)"]
    colors = [PALETTE["red_strong"], PALETTE["blue_main"]]
    metrics_to_plot = (
        ("policy_value_mae", "Policy value MAE", "lower is better"),
        ("policy_rank_correlation", "Policy rank", "higher is better"),
        ("improvement_differential_error", "Transition IDE", "lower is better"),
        ("improvement_sign_consistency", "Transition ISC", "higher is better"),
        ("improvement_reversal_rate", "Reversal rate", "lower is better"),
        ("update_selection_regret", "Selection regret", "lower is better"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.5))
    for axis, (key, title, subtitle) in zip(axes.flat, metrics_to_plot):
        values = [float(row[key]) if row[key] is not None else 0.0 for row in rows]
        axis.bar(labels, values, color=colors, width=0.62)
        axis.set_title(title, fontsize=9)
        axis.text(0.5, -0.28, subtitle, transform=axis.transAxes, ha="center", fontsize=7)
        axis.grid(axis="y", alpha=0.2, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", labelsize=7)
    figure.suptitle("Value fidelity is not improvement fidelity", fontsize=11, y=1.01)
    figure.tight_layout()
    finalize_figure(figure, output_dir / stem, formats=("png",), dpi=300)


def _semantic_scatter_png(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _fallback_png(path)
        return
    positive = [row for row in rows if str(row.get("reversal", "")).casefold() not in {"true", "1"}]
    reversal = [row for row in rows if str(row.get("reversal", "")).casefold() in {"true", "1"}]
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.axvline(0.0, color="#333333", linewidth=0.8)
    if positive:
        axis.scatter(
            [float(row["delta_proxy"]) for row in positive],
            [float(row["delta_true"]) for row in positive],
            color=PALETTE["blue_secondary"],
            alpha=0.72,
            label="other transitions",
        )
    if reversal:
        axis.scatter(
            [float(row["delta_proxy"]) for row in reversal],
            [float(row["delta_true"]) for row in reversal],
            color=PALETTE["red_strong"],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.95,
            label="improvement reversal",
        )
    _, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()
    axis.fill_betweenx([0.0, ymax], 0.0, xmax, color="#eaf4ea", alpha=0.35, zorder=-2)
    axis.fill_betweenx([ymin, 0.0], 0.0, xmax, color="#fff0f0", alpha=0.35, zorder=-2)
    axis.text(0.03, 0.96, "hidden gain", transform=axis.transAxes, fontsize=8, va="top")
    axis.text(0.62, 0.96, "correct improvement", transform=axis.transAxes, fontsize=8, va="top")
    axis.text(
        0.62,
        0.06,
        "false improvement\n(improvement reversal)",
        transform=axis.transAxes,
        fontsize=7.5,
        va="bottom",
    )
    axis.text(0.03, 0.06, "correct failure", transform=axis.transAxes, fontsize=8, va="bottom")
    axis.set(xlabel="Proxy improvement", ylabel="Deployment improvement", title="When better gets worse")
    axis.legend(loc="best", fontsize=8, frameon=False)
    figure.tight_layout()
    finalize_figure(figure, path, formats=("png",), dpi=300)


def _semantic_heatmap_png(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _fallback_png(path)
        return
    responses = sorted({float(row["response_strength"]) for row in rows})
    footprints = sorted({float(row["update_footprint"]) for row in rows})
    lookup = {
        (float(row["response_strength"]), float(row["update_footprint"])): float(row["irr"])
        for row in rows
    }
    matrix = [
        [lookup.get((response, footprint), float("nan")) for footprint in footprints]
        for response in responses
    ]
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    x_edges = _grid_edges(footprints)
    y_edges = _grid_edges(responses)
    image = axis.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        interpolation="bilinear",
        vmin=0.0,
        vmax=1.0,
        cmap=reversal_cmap(),
    )
    if len(responses) > 1 and len(footprints) > 1:
        axis.contour(footprints, responses, matrix, levels=[0.5], colors="#f5f5f5", linewidths=1.2)
    axis.set_xticks(footprints, [f"{value:.2f}" for value in footprints], rotation=45, ha="right")
    axis.set_yticks(responses, [f"{value:.2f}" for value in responses])
    axis.set(
        xlabel="Update footprint",
        ylabel="Response strength",
        title="Improvement reversal phase diagram",
    )
    axis.text(
        0.02,
        0.02,
        "bilinear display; light contour: zero-to-positive boundary",
        transform=axis.transAxes,
        fontsize=7,
        color="white",
    )
    figure.colorbar(image, ax=axis, label="IRR (continuous scale)")
    figure.tight_layout()
    finalize_figure(figure, path, formats=("png",), dpi=300)


def _grid_edges(values: list[float]) -> list[float]:
    if not values:
        return [0.0, 1.0]
    if len(values) == 1:
        width = max(0.05, abs(values[0]) * 0.25)
        return [values[0] - width, values[0] + width]
    interior = [(left + right) / 2.0 for left, right in pairwise(values)]
    return [values[0] - (interior[0] - values[0]), *interior, values[-1] + (values[-1] - interior[-1])]


def _build_budget_frontier(input_dir: Path, output_dir: Path) -> None:
    source = _find(input_dir, "budget_frontier.csv")
    stem = "fig5_pivot_budget_frontier"
    if source is None:
        _mark_unavailable(output_dir, stem, "E5 budget_frontier.csv was not found.")
        return
    rows = _read_csv(source)
    shutil.copyfile(source, output_dir / f"{stem}.csv")
    _grouped_line_png(
        output_dir / f"{stem}.png",
        rows,
        group_key="method",
        x_key="mean_queries",
        y_key="mean_cti",
        title="PIVOT budget frontier",
    )


def _build_strategic_layers(input_dir: Path, output_dir: Path) -> None:
    source = _find(input_dir, "strategic_reversal.json")
    stem = "fig6_observer_actor_strategic"
    if source is None:
        _mark_unavailable(output_dir, stem, "E7 strategic_reversal.json was not found.")
        return
    rows = _load_json_rows(source)
    _write_csv(output_dir / f"{stem}.csv", rows)
    _scatter_png(
        output_dir / f"{stem}.png",
        rows,
        x_key="seed",
        y_keys=("delta_proxy", "delta_actor", "delta_strategic"),
        title="Observer, actor, strategic",
    )


def _build_competition(input_dir: Path, output_dir: Path) -> None:
    source = _find(input_dir, "competition.json")
    stem = "fig7_strategic_reversal"
    if source is None:
        _mark_unavailable(output_dir, stem, "E8 competition.json was not found.")
        return
    rows = _load_json_rows(source)
    grouped: dict[tuple[str, float], list[float]] = {}
    for row in rows:
        key = (str(row["mode"]), float(row["strategic_sensitivity"]))
        grouped.setdefault(key, []).append(float(row["competition_effect"]))
    summary = [
        {
            "mode": mode,
            "strategic_sensitivity": sensitivity,
            "mean_competition_effect": sum(values) / len(values),
            "n": len(values),
        }
        for (mode, sensitivity), values in sorted(grouped.items())
    ]
    _write_csv(output_dir / f"{stem}.csv", summary)
    _grouped_line_png(
        output_dir / f"{stem}.png",
        summary,
        group_key="mode",
        x_key="strategic_sensitivity",
        y_key="mean_competition_effect",
        title="Strategic response strength",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected a list of records in {path}")
    return rows


def _mark_unavailable(output_dir: Path, stem: str, reason: str) -> None:
    (output_dir / f"{stem}.unavailable").write_text(reason + "\n", encoding="utf-8")


def _line_png(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        # Reuse the valid fallback renderer from the controlled output when a
        # minimal environment has no Matplotlib.
        from pivot.environments.performative.proxy import _canvas, _line, _save_png

        canvas = _canvas(640, 420)
        values = [float(row["true_value"]) for row in rows]
        for index in range(1, len(values)):
            _line(canvas, 50 + (index - 1) * 560 // max(1, len(values) - 1), 380 - int(values[index - 1]), 50 + index * 560 // max(1, len(values) - 1), 380 - int(values[index]), [38, 104, 161])
        _save_png(path, canvas)
        return
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot([row["round_id"] for row in rows], [row["proxy_value"] for row in rows], label="proxy")
    axis.plot([row["round_id"] for row in rows], [row["true_value"] for row in rows], label="true")
    axis.set(xlabel="round", ylabel="value", title="Optimizing the wrong world")
    axis.legend()
    figure.tight_layout()
    finalize_figure(figure, path, formats=("png",), dpi=300)


def _scatter_png(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_keys: tuple[str, ...],
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _fallback_png(path)
        return
    figure, axis = plt.subplots(figsize=(6, 4))
    x = [float(row[x_key]) for row in rows]
    for index, key in enumerate(y_keys):
        axis.scatter(
            x,
            [float(row[key]) for row in rows],
            label=key,
            alpha=0.75,
            color=DEFAULT_COLORS[index % len(DEFAULT_COLORS)],
        )
    axis.axhline(0.0, color=PALETTE["text"], linewidth=0.7)
    axis.set(xlabel=x_key, ylabel="value", title=title)
    axis.legend()
    figure.tight_layout()
    finalize_figure(figure, path, formats=("png",), dpi=300)


def _grouped_line_png(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    x_key: str,
    y_key: str,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _fallback_png(path)
        return
    figure, axis = plt.subplots(figsize=(6, 4))
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[group_key]), []).append(row)
    for index, (name, group) in enumerate(sorted(groups.items())):
        ordered = sorted(group, key=lambda row: float(row[x_key]))
        axis.plot(
            [float(row[x_key]) for row in ordered],
            [float(row[y_key]) for row in ordered],
            marker="o",
            label=name,
            color=DEFAULT_COLORS[index % len(DEFAULT_COLORS)],
        )
    axis.set(xlabel=x_key, ylabel=y_key, title=title)
    axis.legend(fontsize=8)
    figure.tight_layout()
    finalize_figure(figure, path, formats=("png",), dpi=300)


def _fallback_png(path: Path) -> None:
    from pivot.environments.performative.proxy import _canvas, _line, _save_png

    canvas = _canvas(640, 420)
    _line(canvas, 50, 370, 610, 30, [38, 104, 161])
    _save_png(path, canvas)


if __name__ == "__main__":
    main()
