from __future__ import annotations

import csv
import json
import math
import platform
import struct
import subprocess
import zlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

from pivot.core.policy import Policy
from pivot.core.result import RolloutContext
from pivot.core.transition import TRANSITION_COLUMNS, PolicyTransition
from pivot.evaluation.decomposition import decompose_effects
from pivot.evaluation.paired import PairedEvaluator
from pivot.evaluation.uncertainty import bootstrap_mean_ci
from pivot.footprint.generic import compute_update_footprint
from pivot.logging.transition_store import Manifest, TransitionStore
from pivot.metrics.improvement import compute_improvement_metrics

from .config import PerformativeConfig
from .world import PerformativeWorld

REQUIRED_TRANSITION_COLUMNS = TRANSITION_COLUMNS


def run_first_milestone(
    output_dir: Path,
    config: PerformativeConfig,
    seeds: Sequence[int],
    candidate_scales: Sequence[float],
    response_strengths: Sequence[float] | None = None,
    optimization_strengths: Sequence[float] | None = None,
) -> Manifest:
    """Run the registered controlled E1/E2 grid and persist all first outputs.

    The default remains a single response/optimization setting for the small
    integration smoke test.  The registered P2 config passes three or more
    values for each axis and therefore produces the response-by-footprint
    phase diagram used by the paper protocol.
    """

    output_dir = Path(output_dir)
    response_grid = _validate_grid(
        response_strengths if response_strengths is not None else [config.response_strength],
        "response_strengths",
    )
    optimization_grid = _validate_grid(
        optimization_strengths
        if optimization_strengths is not None
        else [config.optimization_strength],
        "optimization_strengths",
    )
    scales = _validate_grid(candidate_scales, "candidate_scales")
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("seeds must not be empty")
    if any(seed < 0 for seed in seed_values):
        raise ValueError("seeds must be non-negative")

    store = TransitionStore(output_dir, required_columns=REQUIRED_TRANSITION_COLUMNS)
    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    incumbent = Policy.from_mapping({"intensity": 0.2})
    states = (-1.0, -0.5, 0.0, 0.5, 1.0)
    for response_strength in response_grid:
        world_config = replace_config(config, response_strength=response_strength)
        world = PerformativeWorld(world_config)
        for optimization_strength in optimization_grid:
            for seed in seed_values:
                contexts = [RolloutContext(seed=seed, scenario_id=f"episode-{seed}")]
                for index, scale in enumerate(scales):
                    try:
                        candidate_value = min(
                            0.95, max(-0.95, 0.2 + scale * optimization_strength)
                        )
                        candidate = Policy.from_mapping({"intensity": candidate_value})
                        config_id = (
                            f"{config.config_id}:r={response_strength:g}:o={optimization_strength:g}"
                        )
                        transition = PolicyTransition(
                            incumbent=incumbent,
                            candidate=candidate,
                            round_id=0,
                            candidate_index=index,
                            improvement_operator="synthetic",
                            edit_type="intensity",
                            proxy_world_id=f"{config_id}:observer",
                            high_fidelity_world_id=f"{config_id}:actor",
                            response_strength=response_strength,
                            competition_strength=config.competition_strength,
                            optimization_strength=optimization_strength,
                            seed=seed,
                            config_id=config_id,
                        )
                        proxy = PairedEvaluator(world, mode="observer").evaluate(transition, contexts)
                        actor = PairedEvaluator(world, mode="actor").evaluate(transition, contexts)
                        footprint = compute_update_footprint(incumbent, candidate, states)
                        effects = decompose_effects(proxy.delta, actor.delta, None)
                        enriched = replace(
                            transition,
                            proxy_incumbent_value=proxy.incumbent_value,
                            proxy_candidate_value=proxy.candidate_value,
                            delta_proxy=proxy.delta,
                            actor_incumbent_value=actor.incumbent_value,
                            actor_candidate_value=actor.candidate_value,
                            delta_actor=actor.delta,
                            true_incumbent_value=actor.incumbent_value,
                            true_candidate_value=actor.candidate_value,
                            delta_true=actor.delta,
                            mechanical_effect=effects.mechanical_effect,
                            update_footprint=footprint.distance,
                            footprint_components=footprint.components,
                            paired_seed_ids=actor.paired_seed_ids,
                            hf_queried=True,
                            hf_query_reason="first_milestone_full_actor",
                            hf_query_cost=float(actor.environment_steps),
                            improvement_reversal=proxy.delta > 1e-9 and actor.delta < -1e-9,
                        )
                        row = enriched.to_record()
                        store.append(row)
                        all_rows.append(row)
                    except (ArithmeticError, TypeError, ValueError) as error:
                        failures.append(
                            {
                                "response_strength": response_strength,
                                "optimization_strength": optimization_strength,
                                "seed": seed,
                                "candidate_index": index,
                                "candidate_scale": scale,
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                        )
    manifest = store.finalize()
    metrics = compute_improvement_metrics(all_rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "failed_runs.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in failures), encoding="utf-8"
    )
    _write_provenance(
        output_dir,
        config=config,
        response_strengths=response_grid,
        optimization_strengths=optimization_grid,
        candidate_scales=scales,
        seeds=seed_values,
        manifest=manifest,
    )
    _write_first_milestone_outputs(output_dir, all_rows)
    return manifest


def replace_config(config: PerformativeConfig, **changes: float) -> PerformativeConfig:
    """Build a world config without mutating the frozen registered config."""

    values: dict[str, Any] = {
        "response_strength": config.response_strength,
        "competition_strength": config.competition_strength,
        "noise_scale": config.noise_scale,
        "horizon": config.horizon,
        "reward_bound": config.reward_bound,
        "decay": config.decay,
        "optimization_strength": config.optimization_strength,
        "config_id": config.config_id,
    }
    values.update(changes)
    return PerformativeConfig(**values)


def _validate_grid(values: Iterable[float], name: str) -> tuple[float, ...]:
    grid = tuple(float(value) for value in values)
    if not grid or any(value < 0 or not math.isfinite(value) for value in grid):
        raise ValueError(f"{name} must contain finite non-negative values")
    return grid


def _write_provenance(
    output_dir: Path,
    *,
    config: PerformativeConfig,
    response_strengths: Sequence[float],
    optimization_strengths: Sequence[float],
    candidate_scales: Sequence[float],
    seeds: Sequence[int],
    manifest: Manifest,
) -> None:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    payload = {
        "run_id": output_dir.name,
        "config_snapshot": config.__dict__,
        "response_strengths": list(response_strengths),
        "optimization_strengths": list(optimization_strengths),
        "candidate_scales": list(candidate_scales),
        "seeds": list(seeds),
        "git_commit": git_commit,
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment_version": "pivot-controlled-v1",
        "dataset_version": "synthetic-performative-v1",
        "dependency_versions": _dependency_versions(),
        "manifest": manifest.__dict__,
        "paired": True,
        "tau_sign": 1e-9,
        "tau_mtr": 1e-8,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )


def _write_first_milestone_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot generate first-milestone outputs without rows")
    metrics = compute_improvement_metrics(rows)
    scatter = [
        {
            "delta_proxy": float(row["delta_proxy"]),
            "delta_true": float(row["delta_true"]),
            "reversal": bool(row["improvement_reversal"]),
            "response_strength": float(row["response_strength"]),
            "update_footprint": float(row["update_footprint"]),
        }
        for row in rows
    ]
    response_groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
    footprint_groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
    heatmap_groups: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        response = _stable_float(row["response_strength"])
        footprint = _stable_float(row["update_footprint"])
        response_groups[response].append(row)
        footprint_groups[footprint].append(row)
        heatmap_groups[(response, footprint)].append(row)
    irr_by_response = _group_summary(response_groups, "response_strength")
    irr_by_footprint = _group_summary(footprint_groups, "update_footprint")
    heatmap = [
        {
            "response_strength": response,
            "update_footprint": footprint,
            **_summary(group),
        }
        for (response, footprint), group in sorted(heatmap_groups.items())
    ]
    _write_json(output_dir / "proxy_vs_true_scatter.json", scatter)
    _write_json(output_dir / "irr_vs_response.json", irr_by_response)
    _write_json(output_dir / "irr_vs_footprint.json", irr_by_footprint)
    _write_json(output_dir / "response_footprint_heatmap.json", heatmap)
    confidence_rows = [
        {"metric": "irr", "estimate": metrics["irr"], **_ci([float(row["improvement_reversal"]) for row in rows])},
        {"metric": "ide", "estimate": metrics["ide"], **_ci([abs(float(row["delta_proxy"]) - float(row["delta_true"])) for row in rows])},
    ]
    _write_json(output_dir / "confidence_intervals.json", confidence_rows)
    _write_csv(output_dir / "confidence_intervals.csv", confidence_rows)
    _write_csv(output_dir / "proxy_vs_true_scatter.csv", scatter)
    _write_csv(output_dir / "irr_vs_response.csv", irr_by_response)
    _write_csv(output_dir / "irr_vs_footprint.csv", irr_by_footprint)
    _write_csv(output_dir / "response_footprint_heatmap.csv", heatmap)
    _write_plots(output_dir, scatter, irr_by_response, irr_by_footprint, heatmap)


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    values = [float(row["improvement_reversal"]) for row in rows]
    return {"irr": sum(values) / len(values), "n": len(values), **_ci(values)}


def _group_summary(groups: dict[float, list[dict[str, Any]]], key: str) -> list[dict[str, Any]]:
    return [{key: value, **_summary(rows)} for value, rows in sorted(groups.items())]


def _ci(values: Sequence[float]) -> dict[str, float]:
    low, high = bootstrap_mean_ci(values, seed=20260819)
    return {"ci_low": low, "ci_high": high}


def _stable_float(value: object, digits: int = 12) -> float:
    if not isinstance(value, (float, int, str)):
        raise TypeError(f"expected a numeric value, got {type(value).__name__}")
    return round(float(value), digits)


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "yaml", "pyarrow", "matplotlib"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[name] = "unavailable"
    return versions


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(
    output_dir: Path,
    scatter: Sequence[dict[str, Any]],
    irr_by_response: Sequence[dict[str, Any]],
    irr_by_footprint: Sequence[dict[str, Any]],
    heatmap: Sequence[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _write_fallback_pngs(output_dir, scatter, irr_by_response, irr_by_footprint, heatmap)
        return

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.scatter(
        [row["delta_proxy"] for row in scatter if not row["reversal"]],
        [row["delta_true"] for row in scatter if not row["reversal"]],
        c=[row["response_strength"] for row in scatter if not row["reversal"]],
        cmap="viridis",
        alpha=0.72,
        label="other transitions",
    )
    axis.scatter(
        [row["delta_proxy"] for row in scatter if row["reversal"]],
        [row["delta_true"] for row in scatter if row["reversal"]],
        color="#c43d3d",
        edgecolor="white",
        linewidth=0.45,
        alpha=0.95,
        label="improvement reversal",
    )
    axis.axvspan(0.0, axis.get_xlim()[1], color="#eaf4ea", alpha=0.35, zorder=-2)
    axis.axhspan(axis.get_ylim()[0], 0.0, color="#fff0f0", alpha=0.35, zorder=-2)
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
    axis.set(xlabel="Proxy improvement", ylabel="Actor improvement", title="When better gets worse")
    axis.legend(loc="best", fontsize=8, frameon=True)
    figure.tight_layout()
    figure.savefig(output_dir / "proxy_vs_true_scatter.png", dpi=160)
    plt.close(figure)

    _line_plot(output_dir / "irr_vs_response.png", irr_by_response, "response_strength", "irr", "IRR vs response")
    _line_plot(output_dir / "irr_vs_footprint.png", irr_by_footprint, "update_footprint", "irr", "IRR vs footprint")

    responses = sorted({float(row["response_strength"]) for row in heatmap})
    footprints = sorted({float(row["update_footprint"]) for row in heatmap})
    lookup = {(float(row["response_strength"]), float(row["update_footprint"])): float(row["irr"]) for row in heatmap}
    matrix = [[lookup.get((response, footprint), float("nan")) for footprint in footprints] for response in responses]
    figure, axis = plt.subplots(figsize=(6, 4))
    x_edges = _edges(footprints)
    y_edges = _edges(responses)
    image = axis.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        interpolation="none",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    if len(responses) > 1 and len(footprints) > 1:
        axis.contour(footprints, responses, matrix, levels=[0.5], colors="#f5f5f5", linewidths=1.2)
    axis.set_xticks(footprints, [f"{value:.3f}" for value in footprints], rotation=45, ha="right")
    axis.set_yticks(responses, [f"{value:.2f}" for value in responses])
    axis.set(xlabel="Update footprint", ylabel="Response strength", title="Response x footprint")
    for response_index, response in enumerate(responses):
        for footprint_index, footprint in enumerate(footprints):
            value = matrix[response_index][footprint_index]
            if math.isfinite(value):
                axis.text(
                    footprint,
                    response,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="black" if value > 0.55 else "white",
                )
    axis.text(
        0.02,
        0.02,
        "raw sampled cells; labels show IRR; contour is descriptive only",
        transform=axis.transAxes,
        fontsize=7,
        color="white",
    )
    figure.colorbar(image, ax=axis, label="IRR (continuous scale)")
    figure.tight_layout()
    figure.savefig(output_dir / "response_footprint_heatmap.png", dpi=160)
    plt.close(figure)


def _edges(values: Sequence[float]) -> list[float]:
    """Return cell edges for an ordered numeric grid."""

    if not values:
        return [0.0, 1.0]
    if len(values) == 1:
        width = max(0.05, abs(values[0]) * 0.25)
        return [values[0] - width, values[0] + width]
    interior = [(left + right) / 2.0 for left, right in pairwise(values)]
    return [values[0] - (interior[0] - values[0]), *interior, values[-1] + (values[-1] - interior[-1])]


def _line_plot(path: Path, rows: Sequence[dict[str, Any]], x_key: str, y_key: str, title: str) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 4))
    x = [float(row[x_key]) for row in rows]
    y = [float(row[y_key]) for row in rows]
    low = [float(row["ci_low"]) for row in rows]
    high = [float(row["ci_high"]) for row in rows]
    axis.plot(x, y, marker="o")
    axis.fill_between(x, low, high, alpha=0.2)
    axis.set(xlabel=x_key, ylabel=y_key, title=title, ylim=(-0.02, 1.02))
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_fallback_pngs(
    output_dir: Path,
    scatter: Sequence[dict[str, Any]],
    irr_by_response: Sequence[dict[str, Any]],
    irr_by_footprint: Sequence[dict[str, Any]],
    heatmap: Sequence[dict[str, Any]],
) -> None:
    """Write small valid PNGs when optional Matplotlib is unavailable.

    The fallback keeps the artifact contract usable in a minimal Python
    environment.  Full publication styling is supplied by Matplotlib when
    the research environment is installed.
    """

    width, height = 640, 420
    _save_png(output_dir / "proxy_vs_true_scatter.png", _scatter_pixels(scatter, width, height))
    _save_png(output_dir / "irr_vs_response.png", _line_pixels(irr_by_response, "response_strength", width, height),)
    _save_png(output_dir / "irr_vs_footprint.png", _line_pixels(irr_by_footprint, "update_footprint", width, height),)
    _save_png(output_dir / "response_footprint_heatmap.png", _heatmap_pixels(heatmap, width, height))


def _canvas(width: int, height: int) -> list[list[list[int]]]:
    return [[[255, 255, 255] for _ in range(width)] for _ in range(height)]


def _pixel(canvas: list[list[list[int]]], x: int, y: int, color: list[int]) -> None:
    if 0 <= y < len(canvas) and 0 <= x < len(canvas[0]):
        canvas[y][x] = color


def _line(canvas: list[list[list[int]]], x0: int, y0: int, x1: int, y1: int, color: list[int]) -> None:
    dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
    dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        _pixel(canvas, x0, y0, color)
        if x0 == x1 and y0 == y1:
            return
        double = 2 * error
        if double >= dy:
            error += dy
            x0 += sx
        if double <= dx:
            error += dx
            y0 += sy


def _dot(canvas: list[list[list[int]]], x: int, y: int, color: list[int], radius: int = 4) -> None:
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                _pixel(canvas, x + dx, y + dy, color)


def _scatter_pixels(rows: Sequence[dict[str, Any]], width: int, height: int) -> list[list[list[int]]]:
    canvas = _canvas(width, height)
    left, right, top, bottom = 55, width - 20, 20, height - 45
    _line(canvas, left, bottom, right, bottom, [20, 20, 20])
    _line(canvas, left, top, left, bottom, [20, 20, 20])
    xs = [float(row["delta_proxy"]) for row in rows]
    ys = [float(row["delta_true"]) for row in rows]
    _draw_zero_axes(canvas, xs, ys, left, right, top, bottom)
    for x, y in zip(xs, ys):
        _dot(canvas, _scale(x, xs, left, right), _scale(y, ys, bottom, top), [38, 104, 161])
    return canvas


def _line_pixels(rows: Sequence[dict[str, Any]], x_key: str, width: int, height: int) -> list[list[list[int]]]:
    canvas = _canvas(width, height)
    left, right, top, bottom = 55, width - 20, 20, height - 45
    _line(canvas, left, bottom, right, bottom, [20, 20, 20])
    _line(canvas, left, top, left, bottom, [20, 20, 20])
    xs = [float(row[x_key]) for row in rows]
    ys = [float(row["irr"]) for row in rows]
    previous: tuple[int, int] | None = None
    for x, y in zip(xs, ys):
        point = (_scale(x, xs, left, right), _scale(y, [0.0, 1.0], bottom, top))
        if previous is not None:
            _line(canvas, previous[0], previous[1], point[0], point[1], [38, 104, 161])
        _dot(canvas, point[0], point[1], [38, 104, 161])
        previous = point
    return canvas


def _heatmap_pixels(rows: Sequence[dict[str, Any]], width: int, height: int) -> list[list[list[int]]]:
    canvas = _canvas(width, height)
    responses = sorted({float(row["response_strength"]) for row in rows})
    footprints = sorted({float(row["update_footprint"]) for row in rows})
    if not responses or not footprints:
        return canvas
    left, right, top, bottom = 55, width - 20, 20, height - 45
    cell_w = max(1, (right - left) // len(footprints))
    cell_h = max(1, (bottom - top) // len(responses))
    for row in rows:
        xi = footprints.index(float(row["update_footprint"]))
        yi = responses.index(float(row["response_strength"]))
        value = max(0.0, min(1.0, float(row["irr"])))
        color = [int(255 * value), int(80 * (1.0 - value)), int(180 * (1.0 - value))]
        x0 = left + xi * cell_w
        y0 = bottom - (yi + 1) * cell_h
        for x in range(x0, min(right, x0 + cell_w)):
            for y in range(y0, min(bottom, y0 + cell_h)):
                _pixel(canvas, x, y, color)
    return canvas


def _draw_zero_axes(canvas: list[list[list[int]]], xs: Sequence[float], ys: Sequence[float], left: int, right: int, top: int, bottom: int) -> None:
    if min(xs) <= 0 <= max(xs):
        x = _scale(0.0, xs, left, right)
        _line(canvas, x, top, x, bottom, [180, 180, 180])
    if min(ys) <= 0 <= max(ys):
        y = _scale(0.0, ys, bottom, top)
        _line(canvas, left, y, right, y, [180, 180, 180])


def _scale(value: float, values: Sequence[float], low: int, high: int) -> int:
    minimum, maximum = min(values), max(values)
    if maximum == minimum:
        return (low + high) // 2
    fraction = (value - minimum) / (maximum - minimum)
    return round(low + fraction * (high - low))


def _save_png(path: Path, canvas: list[list[list[int]]]) -> None:
    height, width = len(canvas), len(canvas[0])
    raw = b"".join(b"\x00" + bytes(channel for pixel in row for channel in pixel) for row in canvas)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, level=6))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)
