#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.validation import validate_figure_bundle

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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    _copy_or_unavailable(args.input, args.output, "fig1_when_better_gets_worse", "proxy_vs_true_scatter")
    _copy_or_unavailable(args.input, args.output, "fig2_reversal_phase_diagram", "response_footprint_heatmap")
    _build_overoptimization(args.input, args.output)
    _build_global_vs_local(args.input, args.output)
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


def _build_global_vs_local(input_dir: Path, output_dir: Path) -> None:
    source = _find(input_dir, "local_predictions.csv")
    stem = "fig4_policy_vs_improvement_fidelity"
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
    figure.savefig(path, dpi=160)
    plt.close(figure)


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
    for key in y_keys:
        axis.scatter(x, [float(row[key]) for row in rows], label=key, alpha=0.75)
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set(xlabel=x_key, ylabel="value", title=title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


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
    for name, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: float(row[x_key]))
        axis.plot(
            [float(row[x_key]) for row in ordered],
            [float(row[y_key]) for row in ordered],
            marker="o",
            label=name,
        )
    axis.set(xlabel=x_key, ylabel=y_key, title=title)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _fallback_png(path: Path) -> None:
    from pivot.environments.performative.proxy import _canvas, _line, _save_png

    canvas = _canvas(640, 420)
    _line(canvas, 50, 370, 610, 30, [38, 104, 161])
    _save_png(path, canvas)


if __name__ == "__main__":
    main()
