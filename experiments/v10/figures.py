"""Build the V10 publication figures from frozen V9 source evidence.

No scientific run is performed here.  The builder only transforms frozen
source rows into publication figures and traceable source tables.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from paper.figures.v10_style import COLORS, apply, method_style, save

MODE_ORDER = ["fixed", "reactive", "best_response", "gradient_adaptive", "rl_evolutionary"]
MODE_LABEL = {
    "fixed": "Fixed",
    "reactive": "Reactive",
    "best_response": "Best response",
    "gradient_adaptive": "Gradient adaptive",
    "rl_evolutionary": "RL / evolutionary",
}
SPLIT_ORDER = ["environment", "operator", "response_regime", "trajectory"]
SPLIT_LABEL = {
    "environment": "Environment",
    "operator": "Operator",
    "response_regime": "Response regime",
    "trajectory": "Trajectory",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit(root: Path) -> str | None:
    override = os.environ.get("PIVOT_BUILD_COMMIT", "").strip()
    if override:
        return override
    provenance = root / "configs/v15/build_provenance.json"
    try:
        payload = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    if isinstance(payload, dict):
        configured = payload.get("artifact_source_commit")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Publication bundles may combine summary rows and forest rows.  Coerce
    # mixed-type columns deterministically instead of silently dropping a
    # source row when Arrow infers incompatible schemas.
    columns = sorted({str(key) for row in rows for key in row})
    normalized: dict[str, list[Any]] = {}
    for column in columns:
        values = [row.get(column) for row in rows]
        kinds = {type(value) for value in values if value is not None}
        if len(kinds) > 1:
            values = [None if value is None else str(value) for value in values]
        normalized[column] = values
    pq.write_table(pa.table(normalized), path)


def _ci(values: Iterable[float], seed: int, draws: int = 10000) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return (float("nan"), float("nan"))
    if array.size == 1:
        return (float(array[0]), float(array[0]))
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(draws, array.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _safe_range(values: Iterable[float], pad: float = 0.08) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    lo, hi = float(np.nanmin(array)), float(np.nanmax(array))
    span = max(hi - lo, 1e-6)
    return lo - pad * span, hi + pad * span


def _bundle(
    root: Path,
    output: Path,
    stem: str,
    figure: Any,
    rows: list[dict[str, Any]],
    source_paths: list[Path],
    question: str,
    unit: str,
    *,
    appendix: bool = True,
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_csv = output / f"{stem}.csv"
    source_parquet = output / f"{stem}.parquet"
    _write_csv(source_csv, rows)
    _write_parquet(source_parquet, rows)
    outputs = save(figure, output / stem)
    import matplotlib.pyplot as plt

    plt.close(figure)
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    generated_at = (
        datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        if epoch
        else None
    )
    config_hashes = {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted((root / "configs/v9").glob("*.yaml"))
        if path.is_file()
    }
    metadata = {
        "figure_id": stem,
        "scientific_question": question,
        "unit_of_inference": unit,
        "appendix": appendix,
        "experiment_sources": sorted({str(path.relative_to(root)) for path in source_paths}),
        "source_hashes": {
            str(path.relative_to(root)): _sha256(path) for path in source_paths if path.is_file()
        },
        "analysis_script": "experiments/v10/figures.py",
        "style_version": "pivot-v10-publication-style-1",
        "generated_at": generated_at,
        "config_hashes": config_hashes,
        "git_commit": _commit(root),
        "source_csv": source_csv.name,
        "source_parquet": source_parquet.name,
        "outputs": [path.name for path in outputs],
        "raw_observations": True,
        "interval_definition": "bootstrap or descriptive span as stated in figure caption",
        "incomparable_conditions": "environment reward scales are not compared by raw magnitude",
        "oracle_reference": "all-HF is reference-only where shown; it is not a connected method line",
        "interpolation": "none; plotted points are observed cells",
        "grayscale_distinguishable": True,
        "display_policy": "source rows are retained; rendering may use deterministic display subsets",
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    (output / f"{stem}.meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _generated_at() -> str | None:
    """Return the release timestamp used by deterministic artifact builds."""

    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    if not epoch:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _config_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted((root / "configs/v9").glob("*.yaml"))
        if path.is_file()
    }


def _architecture_bundle(root: Path, output: Path) -> dict[str, Any]:
    """Materialize the architecture figure as a traceable V10 figure bundle.

    The editable OpenTikZ source is compiled by ``build.sh`` before this
    module runs.  Keeping a semantic node inventory beside the rendered files
    makes the architecture auditable in exactly the same way as data figures.
    """

    source = root / "paper/iclr2027/figures/fig3_pivot_architecture.tex"
    source_pdf = source.with_suffix(".pdf")
    source_svg = source.with_suffix(".svg")
    if not source.is_file() or not source_pdf.is_file() or not source_svg.is_file():
        raise FileNotFoundError(
            "compiled architecture source is required before V10 figure bundling"
        )

    stem = "fig3_pivot_voi"
    destination = output / stem
    shutil.copyfile(source_pdf, destination.with_suffix(".pdf"))
    shutil.copyfile(source_svg, destination.with_suffix(".svg"))
    shutil.copyfile(source, destination.with_suffix(".tex"))

    # Render a raster companion with the same standalone PDF, avoiding a
    # second layout implementation that could drift from the paper figure.
    png = destination.with_suffix(".png")
    subprocess.run(
        [
            "gs",
            "-q",
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=png16m",
            "-r320",
            f"-sOutputFile={png}",
            str(source_pdf),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    nodes = [
        {"node": "incumbent", "stage": "generation", "role": "input", "label": "pi_t incumbent"},
        {
            "node": "operator",
            "stage": "generation",
            "role": "process",
            "label": "self-improvement operator",
        },
        {
            "node": "candidate_batch",
            "stage": "generation",
            "role": "process",
            "label": "candidate batch",
        },
        {
            "node": "cheap_verifier",
            "stage": "generation",
            "role": "process",
            "label": "cheap verifier Delta_V",
        },
        {
            "node": "differential_posterior",
            "stage": "generation",
            "role": "process",
            "label": "posterior p(Delta* | D)",
        },
        {
            "node": "expected_regret",
            "stage": "generation",
            "role": "query",
            "label": "expected regret R(D)",
        },
        {
            "node": "update_footprint",
            "stage": "intervention",
            "role": "process",
            "label": "update footprint z_Delta",
        },
        {"node": "evsi", "stage": "intervention", "role": "query", "label": "EVSI"},
        {
            "node": "cost_aware_acquisition",
            "stage": "intervention",
            "role": "query",
            "label": "EVSI / cost",
        },
        {
            "node": "paired_hf_rollout",
            "stage": "intervention",
            "role": "evaluation",
            "label": "paired high-fidelity rollout",
        },
        {
            "node": "posterior_update_stop",
            "stage": "intervention",
            "role": "decision",
            "label": "posterior update and stop",
        },
    ]
    _write_csv(destination.with_suffix(".csv"), nodes)
    _write_parquet(destination.with_suffix(".parquet"), nodes)
    metadata = {
        "figure_id": stem,
        "alias_of": "fig3_pivot_architecture",
        "scientific_question": "How does PIVOT-VOI allocate paired interventional evaluations to preserve update decisions?",
        "unit_of_inference": "semantic architecture node and directed stage",
        "appendix": False,
        "experiment_sources": [str(source.relative_to(root))],
        "source_hashes": {
            str(path.relative_to(root)): _sha256(path) for path in (source, source_pdf, source_svg)
        },
        "analysis_script": "scripts/build_opentikz_architecture.py",
        "style_version": "pivot-v10-publication-style-1",
        "generated_at": _generated_at(),
        "config_hashes": _config_hashes(root),
        "git_commit": _commit(root),
        "source_csv": f"{stem}.csv",
        "source_parquet": f"{stem}.parquet",
        "outputs": [
            f"{stem}.pdf",
            f"{stem}.svg",
            f"{stem}.png",
            f"{stem}.tex",
            f"{stem}.csv",
            f"{stem}.parquet",
        ],
        "semantic_nodes": len(nodes),
        "raw_observations": True,
        "interval_definition": "not applicable; semantic node inventory",
        "incomparable_conditions": "not applicable",
        "oracle_reference": "not applicable",
        "interpolation": "none",
        "grayscale_distinguishable": True,
        "display_policy": "all semantic nodes retained",
    }
    (output / f"{stem}.meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _cluster_rows(
    rows: list[dict[str, Any]], keys: tuple[str, ...], value: str
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for key, subset in sorted(groups.items()):
        record: dict[str, Any] = {name: value_ for name, value_ in zip(keys, key)}
        values = np.asarray([float(row[value]) for row in subset], dtype=float)
        record[value] = float(np.mean(values))
        record["cluster_n"] = len(subset)
        output.append(record)
    return output


def build(root: Path) -> list[dict[str, Any]]:
    """Build all V10 figures and mirror them into the submission tree."""

    apply()
    root = root.resolve()
    output = root / "figures/v10"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "e2": [
            root / "results/v9/e2c-confirmatory/transition_rows.jsonl.gz",
            root / "tables/v9/e2c_cells.csv",
        ],
        "e3": [
            root / "results/v9/e3c-confirmatory/trajectory_metrics.json",
            root / "results/v9/e3c-confirmatory/mpe2_frozen_reference.json",
            root / "results/v7/e3b-confirmatory/trajectory_metrics.json",
            root / "results/v9/e3c-confirmatory/transition_rows.jsonl.gz",
        ],
        "e4": [
            root / "results/v9/e4c-confirmatory/ood_reports.csv",
            root / "results/v9/e4c-confirmatory/ood_summary.json",
        ],
        "e5": [
            root / "results/v9/e5c-confirmatory/calibration_robustness.json",
            root / "results/v9/e5c-confirmatory/efficiency_frontier.csv",
        ],
        "e7": [
            root / "results/v9/e7c-confirmatory/strategic_rows.jsonl.gz",
            root / "results/v9/e7c-confirmatory/strategic_summary.json",
        ],
        "finance": [
            root / "results/raw/e6-public-calibration/public_finance_rows.json",
            root / "results/raw/e6-public-calibration/summary.json",
        ],
    }
    built = [
        _figure1(root, output, source_paths["e2"]),
        _figure2(root, output, source_paths["e2"]),
        _figure4(root, output, source_paths["e5"]),
        _figure5(root, output, source_paths["e3"]),
        _figure6(root, output, source_paths["e7"]),
        _figure7(root, output, source_paths["e4"]),
        _figure8(root, output, source_paths["e5"]),
        _figure9(root, output, source_paths["e7"]),
        _figure_finance(root, output, source_paths["finance"]),
    ]
    # The architecture is compiled by the paper build just before this
    # module.  Bundle it under the method-oriented stem required by the V10
    # release contract while retaining the editable OpenTikZ source beside it.
    built.insert(2, _architecture_bundle(root, output))
    # Appendix numbering is stable for the manuscript, while the descriptive
    # stems remain convenient for scripts and reviewers who refer to Figures
    # 6--9.  Keep both names byte-identical and point their metadata to the
    # same source rows.
    aliases = {
        "figA_response_footprint": "fig6_world_response_decomposition",
        "figB_learned_ood_null": "fig7_learned_ood_null",
        "figC_posterior_robustness": "fig8_voi_robustness",
        "figD_strategic_distribution": "fig9_strategic_generalization",
        "figE_finance_boundary": "figE_finance_boundary",
    }
    for alias, target in aliases.items():
        if alias == target:
            continue
        for suffix in ("pdf", "svg", "png", "csv", "parquet", "meta.json"):
            shutil.copyfile(output / f"{target}.{suffix}", output / f"{alias}.{suffix}")
        metadata = json.loads((output / f"{target}.meta.json").read_text(encoding="utf-8"))
        metadata["figure_id"] = alias
        metadata["alias_of"] = target
        metadata["source_csv"] = f"{alias}.csv"
        metadata["source_parquet"] = f"{alias}.parquet"
        metadata["outputs"] = [f"{alias}.{suffix}" for suffix in ("pdf", "svg", "png")]
        (output / f"{alias}.meta.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        built.append(metadata)
    manifest = {
        "style_version": "pivot-v10-publication-style-1",
        "generated_at": built[0].get("generated_at") if built else None,
        "config_hashes": built[0].get("config_hashes", {}) if built else {},
        "figures": built,
        "source_commit": _commit(root),
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paper_output = root / "paper/iclr2027/figures/v10"
    if paper_output.exists():
        shutil.rmtree(paper_output)
    shutil.copytree(output, paper_output)
    return built


def _figure1(root: Path, output: Path, source_paths: list[Path]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    rows = _jsonl_gz(source_paths[0])
    # The frozen MPE2 artifact is an external null reference for the
    # closed-loop comparison.  Keep the main phenomenon figure to the two
    # independently implemented controlled worlds, where each intervention is
    # directly interpretable.
    environments = [
        environment
        for environment in ("performative_control", "congestion_resource")
        if any(str(row["environment_id"]) == environment for row in rows)
    ]
    figure, axes = plt.subplots(1, len(environments), figsize=(7.15, 2.35), squeeze=False)
    axes_flat = list(axes[0])
    plot_rows: list[dict[str, Any]] = []
    for axis, environment in zip(axes_flat, environments):
        all_subset = [row for row in rows if str(row["environment_id"]) == environment]
        # Deterministic display subsample; all source rows remain in the CSV.
        stride = max(1, len(all_subset) // 4200)
        subset = all_subset[::stride]
        proxy = np.asarray([float(row["delta_proxy"]) for row in subset])
        true = np.asarray([float(row["delta_true"]) for row in subset])
        reversal = (proxy > 0) & (true < 0)
        all_proxy = np.asarray([float(row["delta_proxy"]) for row in all_subset])
        all_true = np.asarray([float(row["delta_true"]) for row in all_subset])
        all_reversal = (all_proxy > 0) & (all_true < 0)
        axis.scatter(
            proxy[~reversal],
            true[~reversal],
            s=4,
            alpha=0.20,
            color=COLORS["actor"],
            edgecolors="none",
            label="other",
        )
        axis.scatter(
            proxy[reversal],
            true[reversal],
            s=8,
            alpha=0.72,
            color=COLORS["negative"],
            edgecolors="none",
            label="reversal",
        )
        low, high = _safe_range([*proxy, *true], pad=0.08)
        axis.plot([low, high], [low, high], color=COLORS["grid"], ls=":", lw=0.9)
        axis.axhline(0, color=COLORS["text"], lw=0.55, ls="--")
        axis.axvline(0, color=COLORS["text"], lw=0.55, ls="--")
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_title(environment.replace("_", " ").title(), fontsize=8.3)
        axis.set_xlabel(r"proxy improvement $\Delta_V$")
        axis.text(
            0.04,
            0.05,
            f"display n={len(subset)}\nIRR={np.mean(all_reversal):.2f} (all)",
            transform=axis.transAxes,
            fontsize=6.3,
        )
        displayed_ids = {id(row) for row in subset}
        # Keep every source transition in the auditable table.  The explicit
        # flag records the deterministic display subsample used for rendering.
        plot_rows.extend(
            {
                "environment_id": environment,
                "delta_proxy": float(row["delta_proxy"]),
                "delta_true": float(row["delta_true"]),
                "reversal": bool(float(row["delta_proxy"]) > 0 and float(row["delta_true"]) < 0),
                "displayed": id(row) in displayed_ids,
            }
            for row in all_subset
        )
    axes_flat[0].set_ylabel(r"deployment improvement $\Delta_*$")
    axes_flat[0].legend(fontsize=6, loc="upper left")
    figure.suptitle("Improvement reversal: proxy gains can fail after deployment", fontsize=9.5)
    figure.tight_layout(w_pad=1.0)
    return _bundle(
        root,
        output,
        "fig1_improvement_reversal",
        figure,
        plot_rows,
        source_paths,
        "Can a verifier-approved improvement reverse after deployment?",
        "paired transition; full source rows with deterministic display flag",
        appendix=False,
        metadata_extra={
            "display_subsample": "stride=max(1, floor(n/4200))",
            "display_row_count": int(sum(bool(row["displayed"]) for row in plot_rows)),
            "source_row_count": len(plot_rows),
        },
    )


def _figure2(root: Path, output: Path, source_paths: list[Path]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    raw_rows = _csv(source_paths[1])
    raw_rows = [
        row
        for row in raw_rows
        if str(row["environment_id"]) in {"performative_control", "congestion_resource"}
    ]
    for row in raw_rows:
        row["shift_x"] = float(np.log1p(float(row["chi_square_shift"])))
        row["ide"] = float(row["improvement_fidelity"])
        row["global_rank"] = float(row["global_policy_spearman"])
        row["irr"] = float(row["IRR"])
        row["ide_low"] = float(row["if_ci_low"])
        row["ide_high"] = float(row["if_ci_high"])
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(str(row["environment_id"]), float(row["operator_shift"]))].append(row)
    rows: list[dict[str, Any]] = []
    for (environment, shift), subset in sorted(grouped.items()):
        ide_values = [float(item["ide"]) for item in subset]
        low, high = _ci(ide_values, 2200 + len(rows))
        means = {
            key: float(np.mean([float(item[key]) for item in subset]))
            for key in ("shift_x", "ide", "global_rank", "irr")
        }
        rows.append(
            {
                "environment_id": environment,
                "operator_shift": shift,
                **means,
                "ide_low": low,
                "ide_high": high,
                "cell_n": len(subset),
            }
        )
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.25), sharex=True)
    colors = {"performative_control": COLORS["actor"], "congestion_resource": COLORS["strategic"]}
    for environment in sorted({str(row["environment_id"]) for row in rows}):
        subset = sorted(
            [row for row in rows if str(row["environment_id"]) == environment],
            key=lambda row: float(row["shift_x"]),
        )
        x = np.asarray([row["shift_x"] for row in subset])
        color = colors.get(environment, COLORS["global"])
        axes[0].plot(
            x,
            [row["ide"] for row in subset],
            marker="o",
            ms=3.5,
            color=color,
            label=environment.replace("_", " "),
        )
        axes[0].fill_between(
            x,
            [row["ide_low"] for row in subset],
            [row["ide_high"] for row in subset],
            color=color,
            alpha=0.12,
        )
        axes[1].plot(x, [row["global_rank"] for row in subset], marker="s", ms=3.3, color=color)
        axes[2].plot(x, [row["irr"] for row in subset], marker="D", ms=3.2, color=color)
    axes[0].set_ylabel("IDE (operator-relative)")
    axes[1].set_ylabel("global Spearman")
    axes[2].set_ylabel("IRR")
    axes[0].set_xlabel(r"$\log(1+\chi^2)$")
    axes[1].set_xlabel(r"$\log(1+\chi^2)$")
    axes[2].set_xlabel(r"$\log(1+\chi^2)$")
    axes[0].set_title("A  update error")
    axes[1].set_title("B  global fidelity")
    axes[2].set_title("C  decision failure")
    axes[0].legend(fontsize=5.8, loc="upper left")
    figure.suptitle("Operator-relative shift changes update fidelity", fontsize=9.5)
    figure.tight_layout(w_pad=1.0)
    return _bundle(
        root,
        output,
        "fig2_operator_shift",
        figure,
        rows,
        source_paths,
        "Why can global policy fidelity remain stable while operator-relative fidelity worsens?",
        "environment-by-shift cell mean",
        appendix=False,
    )


def _figure4(root: Path, output: Path, source_paths: list[Path]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    group_path = root / "results/v9/e5c-confirmatory/group_metrics.jsonl.gz"
    raw_groups = _jsonl_gz(group_path)
    rows = _csv(source_paths[1])
    rows = [row for row in rows if int(row["candidate_count"]) == 8]
    methods = ["proxy_only", "paired_lucb", "global_voi", "pivot_voi"]
    labels = {
        "proxy_only": "Proxy Only",
        "paired_lucb": "Paired LUCB",
        "global_voi": "Global-VOI",
        "pivot_voi": "PIVOT-VOI",
    }
    figure = plt.figure(figsize=(7.15, 4.25), constrained_layout=False)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.15, 0.95], hspace=0.48, wspace=0.30)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    forest_axis = figure.add_subplot(grid[1, :])
    source_rows: list[dict[str, Any]] = []
    for axis, environment in zip(axes, ["performative_control", "congestion_resource"]):
        # The oracle is a reference, not an acquisition curve; keep it out of
        # the connected method lines and add it as a thin horizontal guide.
        oracle_rows = [
            row
            for row in rows
            if row["environment_id"] == environment and row["method"] == "all_hf"
        ]
        if oracle_rows:
            oracle_mean = float(np.mean([float(row["mean_CISR"]) for row in oracle_rows]))
            axis.axhline(
                oracle_mean, color=COLORS["oracle"], lw=0.7, ls="--", label="All-HF oracle"
            )
            source_rows.extend(
                {**row, "figure_panel": environment, "K_primary": 8, "oracle_reference": True}
                for row in oracle_rows
            )
        for method in methods:
            subset = sorted(
                [
                    row
                    for row in rows
                    if row["environment_id"] == environment and row["method"] == method
                ],
                key=lambda row: float(row["mean_hf_cost"]),
            )
            if not subset:
                continue
            x = np.asarray([float(row["mean_hf_cost"]) for row in subset])
            y = np.asarray([float(row["mean_CISR"]) for row in subset])
            lo = np.asarray([float(row["CISR_ci_low"]) for row in subset])
            hi = np.asarray([float(row["CISR_ci_high"]) for row in subset])
            style = method_style(method)
            if method == "proxy_only":
                axis.errorbar(
                    [0],
                    [y[0]],
                    yerr=[[y[0] - lo[0]], [hi[0] - y[0]]],
                    fmt=style["marker"],
                    color=style["color"],
                    ms=5,
                    capsize=2,
                    label=labels[method],
                )
            elif method == "all_hf":
                axis.axhline(
                    float(y[-1]), color=COLORS["oracle"], lw=0.7, ls="--", label="All-HF oracle"
                )
            else:
                axis.errorbar(
                    x,
                    y,
                    yerr=[y - lo, hi - y],
                    marker=style["marker"],
                    ls=style["ls"],
                    color=style["color"],
                    ms=3.3,
                    lw=(1.5 if method == "pivot_voi" else 1.0),
                    capsize=2,
                    label=labels[method],
                )
            source_rows.extend(
                {**row, "figure_panel": environment, "K_primary": 8} for row in subset
            )
        axis.set_title(environment.replace("_", " ").title(), fontsize=8.3)
        axis.set_xlabel("mean HF cost")
        axis.set_ylabel("CISR")
        axis.grid(alpha=0.65)
    handles, labels_ = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels_, fontsize=6, loc="upper right")
    # Lower panel: paired, fixed-budget effects across candidate counts.  Each
    # interval is bootstrapped over trajectory seeds; no heterogeneous cells
    # are connected by a line.
    fixed_budget = 4
    effect_rows: list[dict[str, Any]] = []
    for environment in ["performative_control", "congestion_resource"]:
        for k in [4, 8, 16]:
            by_method_seed: dict[tuple[str, int], float] = {}
            for row in raw_groups:
                if (
                    row["environment_id"] == environment
                    and int(row["candidate_count"]) == k
                    and int(row["budget"]) == fixed_budget
                    and row["method"] in {"proxy_only", "pivot_voi", "paired_lucb", "global_voi"}
                ):
                    by_method_seed[(str(row["method"]), int(row["seed"]))] = float(row["CISR"])
            for contrast, method in [
                ("Proxy - PIVOT-VOI", "pivot_voi"),
                ("Proxy - Paired LUCB", "paired_lucb"),
                ("Proxy - Global-VOI", "global_voi"),
            ]:
                seeds = sorted(
                    {
                        seed
                        for meth, seed in by_method_seed
                        if meth == "proxy_only" and (method, seed) in by_method_seed
                    }
                )
                differences = [
                    by_method_seed[("proxy_only", seed)] - by_method_seed[(method, seed)]
                    for seed in seeds
                ]
                low, high = _ci(differences, 4400 + len(effect_rows))
                effect_rows.append(
                    {
                        "environment_id": environment,
                        "K": k,
                        "budget": fixed_budget,
                        "contrast": contrast,
                        "estimate": float(np.mean(differences)) if differences else float("nan"),
                        "ci_low": low,
                        "ci_high": high,
                        "seed_n": len(differences),
                    }
                )
    # Only the registered primary PIVOT contrast is displayed in the main
    # forest; other contrasts remain traceable in the source table.
    primary = [row for row in effect_rows if row["contrast"] == "Proxy - PIVOT-VOI"]
    fy = np.arange(len(primary))
    fe = np.asarray([row["estimate"] for row in primary])
    flo = np.asarray([row["ci_low"] for row in primary])
    fhi = np.asarray([row["ci_high"] for row in primary])
    forest_axis.errorbar(
        fe,
        fy,
        xerr=[fe - flo, fhi - fe],
        fmt="D",
        color=COLORS["pivot"],
        ecolor=COLORS["pivot"],
        capsize=2.5,
        lw=1.2,
        ms=4.5,
    )
    forest_axis.axvline(0, color=COLORS["text"], lw=0.65, ls="--")
    forest_axis.set_yticks(
        fy,
        [f"{row['environment_id'].replace('_', ' ').title()}, K={row['K']}" for row in primary],
        fontsize=6.8,
    )
    forest_axis.invert_yaxis()
    forest_axis.set_xlabel("CISR reduction: Proxy Only - PIVOT-VOI")
    forest_axis.set_title(
        f"C  registered paired effect at fixed HF budget {fixed_budget}", fontsize=8.3
    )
    forest_axis.text(
        0.01,
        0.05,
        "95% bootstrap CI; trajectory-seed unit",
        transform=forest_axis.transAxes,
        fontsize=6.5,
    )
    source_rows.extend({**row, "figure_panel": "forest"} for row in effect_rows)
    figure.suptitle("Evidence efficiency varies by environment and candidate set", fontsize=9.5)
    figure.subplots_adjust(top=0.88, bottom=0.13, left=0.09, right=0.98)
    return _bundle(
        root,
        output,
        "fig4_evidence_efficiency",
        figure,
        source_rows,
        [*source_paths, group_path],
        "Under matched candidate sets, how much update-selection regret is removed per HF cost?",
        "trajectory seed at K=8; paired seed effects at fixed budget",
        appendix=False,
    )


def _figure5(root: Path, output: Path, source_paths: list[Path]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    trajectories = _json(source_paths[0])
    environments = ["mpe2_frozen_external_null", "performative_control", "congestion_resource"]
    frozen = _json(source_paths[1]) if source_paths[1].is_file() else {}
    external_rows = _json(source_paths[2]) if source_paths[2].is_file() else []
    transition_rows = (
        _jsonl_gz(source_paths[3]) if len(source_paths) > 3 and source_paths[3].is_file() else []
    )
    figure, axes = plt.subplots(2, 3, figsize=(7.15, 3.95), squeeze=False)
    methods = ["proxy_only", "global_voi", "pivot_voi", "all_hf"]
    source_rows: list[dict[str, Any]] = []
    for col, environment in enumerate(environments):
        if environment == "mpe2_frozen_external_null":
            ext = []
            for source_env in ["external_adaptive_a", "external_adaptive_b"]:
                for method_label, method_id in [
                    ("Proxy Only", "proxy_only"),
                    ("PIVOT-VOI", "pivot_voi"),
                ]:
                    subset = [
                        row
                        for row in external_rows
                        if str(row.get("trajectory_id", "")).startswith(source_env + "|")
                        and str(row.get("trajectory_id", "")).endswith("method=" + method_label)
                    ]
                    if subset:
                        ext.append(
                            (
                                method_id,
                                [float(row["cti"]) for row in subset],
                                [float(row["cisr"]) for row in subset],
                            )
                        )
            for method, cti_values, cisr_values in ext:
                style = method_style(method)
                for axis, values in [(axes[0, col], cti_values), (axes[1, col], cisr_values)]:
                    mean = float(np.mean(values))
                    low, high = _ci(values, 800 + col)
                    axis.errorbar(
                        [0],
                        [mean],
                        yerr=[[mean - low], [high - mean]],
                        fmt=style["marker"],
                        color=style["color"],
                        ms=5,
                        capsize=2,
                        label=style["label"],
                    )
                source_rows.append(
                    {
                        "environment_id": environment,
                        "method": method,
                        "CTI": float(np.mean(cti_values)),
                        "CISR": float(np.mean(cisr_values)),
                        "seed_n": len(cti_values),
                        "frozen_external_null": True,
                        "source_status": frozen.get("status", "null"),
                    }
                )
            axes[0, col].text(
                0.04, 0.05, "frozen external null", transform=axes[0, col].transAxes, fontsize=6.3
            )
            axes[0, col].set_title("MPE2 (frozen external null)", fontsize=8.3)
            axes[1, col].set_title("MPE2 (frozen external null)", fontsize=8.3)
            for axis in axes[:, col]:
                axis.set_xticks([0], ["endpoint"])
                axis.grid(axis="y", alpha=0.65)
            axes[1, col].set_ylim(-0.01, 0.01)
            axes[1, col].text(
                0.04,
                0.82,
                "CISR = 0 for both methods",
                transform=axes[1, col].transAxes,
                fontsize=6.2,
            )
            continue
        # The confirmatory transition stream retains one selected row per
        # seed/round.  Reconstruct cumulative trajectories from those rows,
        # then bootstrap the trajectory (not transition) means.
        grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in transition_rows:
            if (
                str(row.get("environment_id")) == environment
                and str(row.get("method")) in methods
                and int(row.get("round_id", -1)) < 40
            ):
                grouped[(str(row["method"]), int(row["seed"]), int(row["round_id"]))].append(row)
        for method in methods:
            method_rounds: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
            for (method_id, seed, round_id), subset in grouped.items():
                if method_id != method:
                    continue
                selected = next((row for row in subset if bool(row.get("selected"))), None)
                if selected is None:
                    continue
                method_rounds[seed][round_id] = (float(selected["CTI"]), float(selected["CISR"]))
            if not method_rounds:
                # Keep a traceable endpoint fallback if a future frozen source
                # omits its row stream.
                subset = [
                    row
                    for row in trajectories
                    if row["environment_id"] == environment and row["method"] == method
                ]
                if subset:
                    for axis, values in [
                        (axes[0, col], [float(row["CTI"]) for row in subset]),
                        (axes[1, col], [float(row["CISR"]) for row in subset]),
                    ]:
                        mean = float(np.mean(values))
                        low, high = _ci(values, 850 + col)
                        axis.errorbar(
                            [40],
                            [mean],
                            yerr=[[mean - low], [high - mean]],
                            fmt=method_style(method)["marker"],
                            color=method_style(method)["color"],
                            ms=4.5,
                            capsize=2,
                        )
                continue
            rounds = sorted(
                {round_id for per_seed in method_rounds.values() for round_id in per_seed}
            )
            seeds = sorted(method_rounds)
            cti_traces: dict[int, list[float]] = {}
            cisr_traces: dict[int, list[float]] = {}
            for seed in seeds:
                cti_total = 0.0
                cisr_total = 0.0
                for round_id in rounds:
                    if round_id not in method_rounds[seed]:
                        continue
                    cti_value, cisr_value = method_rounds[seed][round_id]
                    cti_total += cti_value
                    cisr_total += cisr_value
                    cti_traces.setdefault(round_id, []).append(cti_total)
                    cisr_traces.setdefault(round_id, []).append(cisr_total)
            style = method_style(method)
            # A deterministic handful of seed traces makes the spread visible
            # without turning 30 trajectories into an opaque ribbon.
            for seed in seeds[:5]:
                ordered = sorted(method_rounds[seed])
                cti_values = []
                cisr_values = []
                cti_total = 0.0
                cisr_total = 0.0
                for round_id in ordered:
                    cti_value, cisr_value = method_rounds[seed][round_id]
                    cti_total += cti_value
                    cisr_total += cisr_value
                    cti_values.append(cti_total)
                    cisr_values.append(cisr_total)
                axes[0, col].plot(ordered, cti_values, color=style["color"], alpha=0.10, lw=0.55)
                axes[1, col].plot(ordered, cisr_values, color=style["color"], alpha=0.10, lw=0.55)
            x = np.asarray(rounds, dtype=float)
            cti_mean = np.asarray([float(np.mean(cti_traces[r])) for r in rounds])
            cisr_mean = np.asarray([float(np.mean(cisr_traces[r])) for r in rounds])
            cti_low = np.asarray([_ci(cti_traces[r], 860 + col)[0] for r in rounds])
            cti_high = np.asarray([_ci(cti_traces[r], 860 + col)[1] for r in rounds])
            cisr_low = np.asarray([_ci(cisr_traces[r], 870 + col)[0] for r in rounds])
            cisr_high = np.asarray([_ci(cisr_traces[r], 870 + col)[1] for r in rounds])
            axes[0, col].plot(
                x,
                cti_mean,
                color=style["color"],
                marker=style["marker"],
                markevery=max(1, len(x) // 6),
                ms=3.0,
                lw=1.25,
                ls=style["ls"],
                label=style["label"],
            )
            axes[0, col].fill_between(x, cti_low, cti_high, color=style["color"], alpha=0.10)
            axes[1, col].plot(
                x,
                cisr_mean,
                color=style["color"],
                marker=style["marker"],
                markevery=max(1, len(x) // 6),
                ms=3.0,
                lw=1.25,
                ls=style["ls"],
                label=style["label"],
            )
            axes[1, col].fill_between(x, cisr_low, cisr_high, color=style["color"], alpha=0.10)
            for round_id, mean_cti, mean_cisr, low_cti, high_cti, low_cisr, high_cisr in zip(
                rounds, cti_mean, cisr_mean, cti_low, cti_high, cisr_low, cisr_high
            ):
                source_rows.append(
                    {
                        "environment_id": environment,
                        "method": method,
                        "round": int(round_id),
                        "CTI_cumulative_mean": float(mean_cti),
                        "CTI_ci_low": float(low_cti),
                        "CTI_ci_high": float(high_cti),
                        "CISR_cumulative_mean": float(mean_cisr),
                        "CISR_ci_low": float(low_cisr),
                        "CISR_ci_high": float(high_cisr),
                        "seed_n": len(seeds),
                        "frozen_external_null": False,
                    }
                )
        for axis in axes[:, col]:
            axis.set_xlabel("round")
            axis.set_title(environment.replace("_", " ").title(), fontsize=8.3)
            axis.grid(axis="y", alpha=0.65)
    axes[0, 0].set_ylabel("cumulative true improvement (CTI)")
    axes[1, 0].set_ylabel("cumulative selection regret (CISR)")
    handles, labels_ = axes[0, 2].get_legend_handles_labels()
    axes[0, 2].legend(handles, labels_, fontsize=5.8, loc="best")
    figure.suptitle("Closed-loop validator outcomes: trajectory-level evidence", fontsize=9.5)
    figure.tight_layout(h_pad=1.0, w_pad=0.9)
    return _bundle(
        root,
        output,
        "fig5_closed_loop",
        figure,
        source_rows,
        source_paths,
        "Does validator choice change repeated self-improvement outcomes?",
        "trajectory seed and round; frozen external null is endpoint-only",
        appendix=False,
    )


def _figure6(root: Path, output: Path, source_paths: list[Path]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    rows = _jsonl_gz(source_paths[0])
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["opponent_mode"]), str(row["opponent_seed"]))].append(row)
    cluster: list[dict[str, Any]] = []
    for (mode, seed), subset in sorted(grouped.items()):
        cluster.append(
            {
                "opponent_mode": mode,
                "opponent_seed": seed,
                "delta_direct": float(np.mean([r["delta_direct"] for r in subset])),
                "delta_actor": float(np.mean([r["delta_actor"] for r in subset])),
                "delta_strategic": float(np.mean([r["delta_strategic"] for r in subset])),
                "strategic_effect": float(np.mean([r["strategic_effect"] for r in subset])),
                "actor_positive": float(np.mean([bool(r["actor_positive"]) for r in subset])),
            }
        )
    # The five opponent-family traces reuse the same 30 seeds.  Preserve all
    # family-by-seed observations as raw visual evidence, but compute summary
    # estimates at the matched-seed unit rather than treating 150 traces as
    # independent replicates.
    matched_by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cluster:
        matched_by_seed[str(row["opponent_seed"])].append(row)
    matched_seed_summary = [
        {
            key: float(np.mean([row[key] for row in subset]))
            for key in ("delta_direct", "delta_actor", "delta_strategic", "strategic_effect")
        }
        for _, subset in sorted(matched_by_seed.items())
    ]
    figure, axes = plt.subplots(
        1, 3, figsize=(7.15, 2.35), gridspec_kw={"width_ratios": [1.05, 0.9, 1.15]}
    )
    # A: one light line per family-by-seed trace; summaries use matched seeds.
    x = np.arange(3)
    for row in cluster:
        axes[0].plot(
            x,
            [row["delta_direct"], row["delta_actor"], row["delta_strategic"]],
            color="#B8BEC7",
            lw=0.45,
            alpha=0.35,
            zorder=1,
        )
    for index, key in enumerate(("delta_direct", "delta_actor", "delta_strategic")):
        values = np.asarray([row[key] for row in matched_seed_summary], dtype=float)
        low, high = _ci(values, 601 + index)
        axes[0].errorbar(
            index,
            float(np.mean(values)),
            yerr=[[float(np.mean(values) - low)], [float(high - np.mean(values))]],
            fmt="o",
            color=[COLORS["direct"], COLORS["actor"], COLORS["strategic"]][index],
            capsize=2.5,
            ms=4.5,
            lw=1.5,
            zorder=3,
        )
    axes[0].axhline(0, color=COLORS["text"], lw=0.6, ls="--")
    axes[0].set_xticks(x, ["Proxy/direct\n(observer)", "Actor", "Strategic"])
    axes[0].set_ylabel(r"paired improvement $\Delta$")
    axes[0].set_title("A  response layers")
    axes[0].text(
        0.03,
        0.04,
        f"{len(cluster)} family-seed traces\nsummary N={len(matched_seed_summary)} matched seeds",
        transform=axes[0].transAxes,
        fontsize=5.8,
    )
    # B: raw family-by-seed distributions with matched-seed summary intervals.
    effects = [
        np.asarray([row["delta_actor"] - row["delta_direct"] for row in cluster]),
        np.asarray([row["delta_strategic"] - row["delta_actor"] for row in cluster]),
    ]
    violin = axes[1].violinplot(effects, positions=[0, 1], widths=0.7, showextrema=False)
    for body, color in zip(violin["bodies"], [COLORS["actor"], COLORS["strategic"]]):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.24)
    rng = np.random.default_rng(602)
    summary_effects = [
        np.asarray([row["delta_actor"] - row["delta_direct"] for row in matched_seed_summary]),
        np.asarray([row["delta_strategic"] - row["delta_actor"] for row in matched_seed_summary]),
    ]
    for pos, values, summary_values, color in zip(
        [0, 1], effects, summary_effects, [COLORS["actor"], COLORS["strategic"]]
    ):
        axes[1].scatter(
            np.full(values.size, pos) + rng.normal(0, 0.045, values.size),
            values,
            s=7,
            alpha=0.38,
            color=color,
            edgecolors="none",
        )
        low, high = _ci(summary_values, 610 + pos)
        mean = float(np.mean(summary_values))
        axes[1].errorbar(
            pos,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="o",
            color=COLORS["text"],
            ms=3.4,
            capsize=2,
            lw=1.1,
            zorder=3,
        )
    axes[1].axhline(0, color=COLORS["text"], lw=0.6, ls="--")
    axes[1].set_xticks([0, 1], ["Actor\n- direct", "Strategic\n- actor"])
    axes[1].set_ylabel("layer effect")
    axes[1].set_title("B  response-effect distribution")
    axes[1].text(
        0.03,
        0.04,
        "points: family-seed traces\nintervals: matched-seed bootstrap",
        transform=axes[1].transAxes,
        fontsize=5.5,
    )
    # C: cluster-level strategic reversal plane.
    family_style = {
        "fixed": (COLORS["direct"], "o"),
        "reactive": (COLORS["actor"], "s"),
        "best_response": (COLORS["strategic"], "D"),
        "gradient_adaptive": (COLORS["strategic"], "^"),
        "rl_evolutionary": (COLORS["strategic"], "v"),
    }
    for mode in MODE_ORDER:
        subset = [row for row in cluster if row["opponent_mode"] == mode]
        if not subset:
            continue
        color, marker = family_style[mode]
        axes[2].scatter(
            [row["delta_actor"] for row in subset],
            [row["delta_strategic"] for row in subset],
            s=13,
            alpha=0.65,
            label=MODE_LABEL[mode],
            color=color,
            marker=marker,
        )
    lim = _safe_range(
        [*(row["delta_actor"] for row in cluster), *(row["delta_strategic"] for row in cluster)],
        pad=0.14,
    )
    axes[2].plot(lim, lim, color=COLORS["grid"], ls=":", lw=0.8)
    axes[2].axhline(0, color=COLORS["text"], lw=0.6, ls="--")
    axes[2].axvline(0, color=COLORS["text"], lw=0.6, ls="--")
    axes[2].fill_between([0, lim[1]], lim[0], 0, color=COLORS["negative"], alpha=0.035)
    axes[2].set_xlim(lim)
    axes[2].set_ylim(lim)
    axes[2].set_xlabel(r"actor $\Delta$")
    axes[2].set_ylabel(r"strategic $\Delta$")
    axes[2].set_title("C  strategic reversal plane")
    # Use the registered cluster-level SIRR (the same estimand reported in the
    # manuscript), rather than pooling transition rows with unequal eligibility.
    summary = _json(source_paths[1])["by_mode"]
    family_sirrs = [
        float(row["SIRR"])
        for row in summary
        if row["opponent_mode"] in {"best_response", "gradient_adaptive", "rl_evolutionary"}
    ]
    sirr = float(np.mean(family_sirrs)) if family_sirrs else float("nan")
    axes[2].text(
        0.04,
        0.05,
        f"adaptive family-mean\nSIRR={100 * sirr:.2f}%",
        transform=axes[2].transAxes,
        fontsize=6.0,
    )
    # Family identity is intentionally shown in the more expansive strategic
    # generalization figure.  A second legend here would overlap this panel's
    # title at two-column width.
    figure.tight_layout(w_pad=1.0)
    out_rows = [{**row, "figure_panel": "cluster_layers"} for row in cluster]
    return _bundle(
        root,
        output,
        "fig6_world_response_decomposition",
        figure,
        out_rows,
        source_paths,
        "Where does a proposed update change across direct observer, actor, and strategic worlds?",
        "matched opponent seed (summary); family-by-seed trace (raw)",
        appendix=True,
    )


def _figure7(root: Path, output: Path, source_paths: list[Path]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    reports = _csv(source_paths[0])
    for row in reports:
        row["isc_effect"] = float(row["transition_ISC"]) - float(row["global_ISC"])
        row["ide_gain"] = float(row["global_IDE"]) - float(row["transition_IDE"])
    split_rows = []
    for split in SPLIT_ORDER:
        subset = [row for row in reports if row["split"] == split]
        for metric in ("isc_effect", "ide_gain"):
            values = [float(row[metric]) for row in subset]
            split_rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "estimate": float(np.mean(values)),
                    "span_low": float(np.min(values)),
                    "span_high": float(np.max(values)),
                    "family_n": len(values),
                }
            )
    figure, axes = plt.subplots(
        1, 3, figsize=(7.15, 2.25), gridspec_kw={"width_ratios": [1.05, 1.05, 1.0]}
    )
    y = np.arange(len(SPLIT_ORDER))
    for axis, metric, title, xlabel in [
        (axes[0], "isc_effect", "A  ISC effect", r"$ISC_{transition}-ISC_{global}$"),
        (axes[1], "ide_gain", "B  IDE gain", r"$IDE_{global}-IDE_{transition}$"),
    ]:
        subset = [row for row in split_rows if row["metric"] == metric]
        estimates = np.asarray([row["estimate"] for row in subset])
        lows = np.asarray([row["span_low"] for row in subset])
        highs = np.asarray([row["span_high"] for row in subset])
        axis.errorbar(
            estimates,
            y,
            xerr=[estimates - lows, highs - estimates],
            fmt="o",
            color=COLORS["global"],
            ecolor=COLORS["global"],
            capsize=3,
            lw=1.2,
            ms=4.5,
        )
        axis.axvline(0, color=COLORS["text"], lw=0.65, ls="--")
        axis.set_yticks(y, [SPLIT_LABEL[s] for s in SPLIT_ORDER])
        axis.invert_yaxis()
        axis.set_xlabel(xlabel)
        axis.set_title(title)
    # C: direct paired scatter exposes the null heterogeneity without using a
    # bar chart or pretending the two families are independent seed draws.
    colors = {"bayesian_linear": COLORS["global"], "bootstrap_ensemble": COLORS["lucb"]}
    for family in sorted({str(row["family"]) for row in reports}):
        subset = [row for row in reports if row["family"] == family]
        axes[2].scatter(
            [float(row["global_ISC"]) for row in subset],
            [float(row["transition_ISC"]) for row in subset],
            s=28,
            color=colors.get(family, COLORS["proxy"]),
            marker=("o" if family == "bayesian_linear" else "s"),
            label=family.replace("_", " "),
        )
    limits = _safe_range(
        [
            *(float(row["global_ISC"]) for row in reports),
            *(float(row["transition_ISC"]) for row in reports),
        ],
        pad=0.12,
    )
    axes[2].plot(limits, limits, color=COLORS["grid"], ls=":", lw=0.9)
    axes[2].set_xlim(limits)
    axes[2].set_ylim(limits)
    axes[2].set_xlabel(r"$ISC_{global}$")
    axes[2].set_ylabel(r"$ISC_{transition}$")
    axes[2].set_title("C  paired OOD reports")
    axes[2].legend(fontsize=5.8, loc="upper left")
    figure.tight_layout(w_pad=1.0)
    return _bundle(
        root,
        output,
        "fig7_learned_ood_null",
        figure,
        reports + split_rows,
        source_paths,
        "Does the transition-specific learner outperform the global evaluator under registered OOD splits?",
        "OOD split and fitted model-family report; whiskers are descriptive spans",
        appendix=True,
    )


def _figure8(root: Path, output: Path, source_paths: list[Path]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    calibration = _json(source_paths[0])
    posterior = [
        {**row, "section": "posterior"} for row in calibration["posterior_sample_stability"]
    ]
    costs = [{**row, "section": "cost"} for row in calibration["cost_misspecification"]]
    baseline = next(row for row in costs if float(row["cost_multiplier"]) == 1.0)
    for row in costs:
        row["delta_CISR"] = float(row["mean_CISR"]) - float(baseline["mean_CISR"])
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.2))
    # Only aggregate group means were frozen for this robustness check.  Draw
    # the observed means, not pseudo-binomial intervals for a Jaccard score.
    p = np.asarray([float(row["selected_set_jaccard"]) for row in posterior])
    axes[0].plot(
        [float(row["posterior_samples"]) for row in posterior],
        p,
        marker="o",
        color=COLORS["pivot"],
        label="query-set agreement",
    )
    axes[0].set_xlabel("posterior samples")
    axes[0].set_ylabel("query-set Jaccard\nvs 1024-sample reference", fontsize=7.4)
    axes[0].set_title("A  Monte Carlo stability")
    axes[0].set_ylim(0, 1)
    axes[0].text(
        0.03,
        0.05,
        f"mean across {int(posterior[0]['groups'])} calibration groups",
        transform=axes[0].transAxes,
        fontsize=6.0,
    )
    c = np.asarray([float(row["cost_multiplier"]) for row in costs])
    d = np.asarray([float(row["delta_CISR"]) for row in costs])
    axes[1].plot(c, d, marker="D", color=COLORS["strategic"], lw=1.3)
    axes[1].axhline(0, color=COLORS["text"], lw=0.65, ls="--")
    axes[1].axvline(1, color=COLORS["grid"], lw=0.8, ls=":")
    axes[1].set_xlabel(r"assumed cost $\hat c/c$")
    axes[1].set_ylabel(r"$\Delta$CISR vs correct cost")
    axes[1].set_title("B  decision degradation")
    axes[1].text(
        0.03, 0.05, f"group N={int(costs[0]['groups'])}", transform=axes[1].transAxes, fontsize=6.5
    )
    p2 = np.asarray([float(row["selected_query_jaccard"]) for row in costs])
    axes[2].plot(c, p2, marker="s", color=COLORS["global"])
    axes[2].set_xlabel(r"assumed cost $\hat c/c$")
    axes[2].set_ylabel("query agreement")
    axes[2].set_title("C  query stability")
    axes[2].set_ylim(0, 1)
    axes[2].text(
        0.03,
        0.05,
        f"mean across {int(costs[0]['groups'])} calibration groups",
        transform=axes[2].transAxes,
        fontsize=6.0,
    )
    figure.tight_layout(w_pad=1.0)
    return _bundle(
        root,
        output,
        "fig8_voi_robustness",
        figure,
        posterior + costs,
        source_paths,
        "Do posterior approximation and cost misspecification materially change PIVOT-VOI decisions?",
        "calibration group",
        appendix=True,
    )


def _figure9(root: Path, output: Path, source_paths: list[Path]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    rows = _jsonl_gz(source_paths[0])
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["opponent_mode"]), str(row["opponent_seed"]))].append(row)
    clusters: list[dict[str, Any]] = []
    for (mode, seed), subset in sorted(grouped.items()):
        clusters.append(
            {
                "opponent_mode": mode,
                "opponent_seed": seed,
                "strategic_effect": float(np.mean([row["strategic_effect"] for row in subset])),
                "delta_actor": float(np.mean([row["delta_actor"] for row in subset])),
                "delta_strategic": float(np.mean([row["delta_strategic"] for row in subset])),
                "actor_positive_rate": float(
                    np.mean([bool(row["actor_positive"]) for row in subset])
                ),
            }
        )
    figure, axes = plt.subplots(
        1, 3, figsize=(7.15, 2.35), gridspec_kw={"width_ratios": [0.95, 1.0, 1.18]}
    )
    distributions = [
        np.asarray([row["strategic_effect"] for row in clusters if row["opponent_mode"] == mode])
        for mode in MODE_ORDER
    ]
    valid = [(mode, values) for mode, values in zip(MODE_ORDER, distributions) if values.size]
    positions = np.arange(len(valid))
    violin = axes[0].violinplot(
        [v for _, v in valid], positions=positions, widths=0.75, showextrema=False
    )
    for body, (mode, _) in zip(violin["bodies"], valid):
        body.set_facecolor(
            COLORS["strategic"]
            if mode in {"best_response", "gradient_adaptive", "rl_evolutionary"}
            else COLORS["actor"]
        )
        body.set_edgecolor("none")
        body.set_alpha(0.24)
    rng = np.random.default_rng(903)
    for pos, (mode, values) in zip(positions, valid):
        axes[0].scatter(
            np.full(values.size, pos) + rng.normal(0, 0.045, values.size),
            values,
            s=7,
            alpha=0.5,
            color=COLORS["strategic"]
            if mode in {"best_response", "gradient_adaptive", "rl_evolutionary"}
            else COLORS["actor"],
            edgecolors="none",
        )
        low, high = _ci(values, 910 + int(pos))
        mean = float(np.mean(values))
        axes[0].errorbar(
            pos,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="o",
            color=COLORS["text"],
            capsize=2,
            ms=3.3,
            lw=1.0,
        )
    axes[0].axhline(0, color=COLORS["text"], lw=0.6, ls="--")
    short_labels = {
        "fixed": "Fixed",
        "reactive": "React.",
        "best_response": "Best\nresp.",
        "gradient_adaptive": "Grad.",
        "rl_evolutionary": "RL/\nEvo.",
    }
    axes[0].set_xticks(positions, [short_labels[m] for m, _ in valid], fontsize=5.5)
    axes[0].tick_params(axis="x", pad=2)
    axes[0].set_ylabel(r"$\Delta_{strategic}-\Delta_{actor}$")
    axes[0].set_title("A  effect distribution")
    # Forest panel includes a numeric effect, CI, and SIRR annotation in the source table.
    summary = _json(source_paths[1])["by_mode"]
    summary = [row for row in summary if row["opponent_mode"] in MODE_ORDER]
    ys = np.arange(len(summary))
    means = np.asarray([float(row["mean_strategic_effect"]) for row in summary])
    lows = np.asarray([float(row["strategic_effect_ci_low"]) for row in summary])
    highs = np.asarray([float(row["strategic_effect_ci_high"]) for row in summary])
    axes[1].errorbar(
        means,
        ys,
        xerr=[means - lows, highs - means],
        fmt="o",
        color=COLORS["strategic"],
        ecolor=COLORS["strategic"],
        capsize=2.5,
        lw=1.2,
    )
    axes[1].axvline(0, color=COLORS["text"], lw=0.6, ls="--")
    axes[1].set_yticks(ys, [MODE_LABEL[str(row["opponent_mode"])] for row in summary], fontsize=6.8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel(r"mean strategic effect")
    axes[1].set_title("B  opponent-family forest")
    for y_, row in zip(ys, summary):
        axes[1].text(
            0.99,
            y_,
            f"SIRR {float(row['SIRR']):.2f} | N={int(row['cluster_n'])}",
            transform=axes[1].get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=5.8,
        )
    # Reversal plane: color and marker encode opponent family without relying on color alone.
    family_style = {
        "fixed": (COLORS["direct"], "o"),
        "reactive": (COLORS["actor"], "s"),
        "best_response": (COLORS["strategic"], "D"),
        "gradient_adaptive": (COLORS["strategic"], "^"),
        "rl_evolutionary": (COLORS["strategic"], "v"),
    }
    for mode in MODE_ORDER:
        subset = [row for row in clusters if row["opponent_mode"] == mode]
        if not subset:
            continue
        color, marker = family_style[mode]
        axes[2].scatter(
            [row["delta_actor"] for row in subset],
            [row["delta_strategic"] for row in subset],
            s=16,
            alpha=0.62,
            label=MODE_LABEL[mode],
            color=color,
            marker=marker,
        )
    adaptive_summary = [
        row
        for row in summary
        if row["opponent_mode"] in {"best_response", "gradient_adaptive", "rl_evolutionary"}
    ]
    sirr = (
        float(np.mean([float(row["SIRR"]) for row in adaptive_summary]))
        if adaptive_summary
        else float("nan")
    )
    lim = _safe_range(
        [*(row["delta_actor"] for row in clusters), *(row["delta_strategic"] for row in clusters)],
        pad=0.15,
    )
    axes[2].plot(lim, lim, color=COLORS["grid"], ls=":", lw=0.8)
    axes[2].axhline(0, color=COLORS["text"], lw=0.6, ls="--")
    axes[2].axvline(0, color=COLORS["text"], lw=0.6, ls="--")
    axes[2].fill_between([0, lim[1]], lim[0], 0, color=COLORS["negative"], alpha=0.035)
    axes[2].set_xlim(lim)
    axes[2].set_ylim(lim)
    axes[2].set_xlabel(r"actor $\Delta$")
    axes[2].set_ylabel(r"strategic $\Delta$")
    axes[2].set_title("C  strategic reversal plane")
    axes[2].text(
        0.04,
        0.06,
        f"adaptive family-mean\nSIRR={100 * sirr:.2f}%",
        transform=axes[2].transAxes,
        fontsize=5.7,
    )
    axes[2].legend(fontsize=5.4, loc="upper left")
    figure.tight_layout(w_pad=1.0)
    out_rows = clusters + [{**row, "figure_panel": "forest"} for row in summary]
    return _bundle(
        root,
        output,
        "fig9_strategic_generalization",
        figure,
        out_rows,
        source_paths,
        "Does strategic reversal generalize across independently adapting opponent mechanisms?",
        "opponent-seed cluster",
        appendix=True,
    )


def _figure_finance(root: Path, output: Path, source_paths: list[Path]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    rows = _json(source_paths[0])
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["participation_rate"])].append(row)
    levels = sorted(grouped)
    means = []
    lows = []
    highs = []
    source_rows: list[dict[str, Any]] = []
    for level in levels:
        subset = grouped[level]
        values = np.asarray(
            [float(row["delta_f2_depth"]) - float(row["delta_f1"]) for row in subset]
        )
        low, high = _ci(values, 950 + len(source_rows))
        means.append(float(np.mean(values)))
        lows.append(low)
        highs.append(high)
        source_rows.append(
            {
                "participation_rate": level,
                "effect": float(np.mean(values)),
                "ci_low": low,
                "ci_high": high,
                "n": len(values),
                "causal_impact_identified": False,
            }
        )
    figure, axis = plt.subplots(figsize=(4.8, 2.35))
    x = np.asarray(levels)
    mean_array = np.asarray(means)
    axis.plot(x, mean_array, marker="o", color=COLORS["actor"], lw=1.4, label="depth - replay")
    axis.fill_between(x, lows, highs, color=COLORS["actor"], alpha=0.15)
    axis.axhline(0, color=COLORS["text"], lw=0.6, ls="--")
    axis.set_xlabel("participation rate")
    axis.set_ylabel(r"mechanical effect $\Delta_{depth}-\Delta_{replay}$", labelpad=5)
    axis.set_title("Finance boundary: observational footprint diagnostic", fontsize=9)
    axis.text(
        0.98,
        0.88,
        "virtual fills; causal reversal\nnot identified",
        transform=axis.transAxes,
        fontsize=6.2,
        va="top",
        ha="right",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )
    axis.legend(fontsize=6.2, loc="lower left", bbox_to_anchor=(0.01, 0.01))
    figure.tight_layout(pad=0.9)
    return _bundle(
        root,
        output,
        "figE_finance_boundary",
        figure,
        source_rows,
        source_paths,
        "What does the public finance stress test identify as participation changes?",
        "session-level observational diagnostic",
        appendix=True,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build V10 publication figures from frozen source evidence"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    built = build(args.root)
    print(
        json.dumps(
            {"count": len(built), "figures": [item["figure_id"] for item in built]}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
