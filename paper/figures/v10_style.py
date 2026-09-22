"""Publication style shared by the V10 figure builders.

The style deliberately uses both color and redundant line/marker encodings so
the figures remain legible when printed or viewed in grayscale.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib as mpl

# These tokens mirror paper/figures/scientific_figure_style.json, a checked-in
# snapshot of the scientific-figure-skills universal high-impact profile.
STYLE_VERSION = "pivot-v12-scientific-figure-suite-1"

# Figure dimensions are specified in inches so that the same canvas is used
# for the PDF, SVG, and PNG exports.  Wide figures match the official ICLR 5.5-inch text width;
# text tokens are specified at the final printed size.
FIGURE_SIZES: dict[str, tuple[float, float]] = {
    "single": (3.35, 2.35),
    "wide": (5.5, 2.25),
    "wide_tall": (5.5, 3.5),
    "appendix": (5.5, 2.5),
}

# Explicit text tokens prevent one-off fontsize values from drifting between
# the canonical V10 builders and the revision/Highway builders.
TEXT_SIZES: dict[str, float] = {
    "base": 8.0,
    "title": 9.0,
    "panel": 8.0,
    "axis": 7.4,
    "tick": 7.0,
    "legend": 7.0,
    "annotation": 7.0,
}
COLORS = {
    "proxy": "#6B7280",
    "global": "#0072B2",
    "lucb": "#CC79A7",
    "pivot": "#009E73",
    "oracle": "#1F2937",
    "direct": "#56B4E9",
    "actor": "#0072B2",
    "strategic": "#D55E00",
    "positive": "#009E73",
    "negative": "#D55E00",
    "cohort_leduc": "#0072B2",
    "cohort_kuhn": "#CC79A7",
    "cohort_melting_pot": "#E69F00",
    "text": "#1F2937",
    "grid": "#D7DEE5",
    "shade": "#F7F9FB",
}

METHOD_STYLE = {
    "proxy_only": {"label": "Proxy Only", "color": COLORS["proxy"], "marker": "o", "ls": "--"},
    "global_value": {
        "label": "Global evaluator",
        "color": COLORS["global"],
        "marker": "s",
        "ls": ":",
    },
    "global_voi": {"label": "Global-VOI", "color": COLORS["global"], "marker": "s", "ls": ":"},
    "paired_lucb": {"label": "Paired LUCB", "color": COLORS["lucb"], "marker": "^", "ls": "-."},
    "pivot_voi": {"label": "PIVOT-KG", "color": COLORS["pivot"], "marker": "D", "ls": "-"},
    "all_hf": {"label": "All-HF reference", "color": COLORS["oracle"], "marker": "x", "ls": "--"},
}


def apply() -> None:
    """Set deterministic, vector-friendly defaults for paper figures."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "regular",
            "font.size": TEXT_SIZES["base"],
            "axes.labelsize": TEXT_SIZES["axis"],
            "axes.titlesize": TEXT_SIZES["panel"],
            "axes.titleweight": "bold",
            "axes.titlepad": 4.0,
            "xtick.labelsize": TEXT_SIZES["tick"],
            "ytick.labelsize": TEXT_SIZES["tick"],
            "legend.fontsize": TEXT_SIZES["legend"],
            "axes.linewidth": 0.7,
            "axes.edgecolor": COLORS["text"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.4,
            "grid.alpha": 0.42,
            "legend.frameon": False,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.2,
            "axes.axisbelow": True,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "svg.fonttype": "none",
            # Matplotlib otherwise generates random SVG element identifiers,
            # which makes byte-identical figure rebuilds impossible.
            "svg.hashsalt": STYLE_VERSION,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def save(figure: Any, stem: Path, formats: Iterable[str] = ("pdf", "svg", "png")) -> list[Path]:
    """Write vector and raster outputs using the same canvas."""

    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for fmt in formats:
        target = stem.with_suffix(f".{fmt}")
        figure.savefig(
            target,
            dpi=320,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
            edgecolor="none",
            metadata={"Creator": STYLE_VERSION} if fmt in {"pdf", "svg"} else None,
        )
        outputs.append(target)
    return outputs


def figure_size(kind: str) -> tuple[float, float]:
    """Return a registered canvas size and reject accidental one-off sizes."""

    try:
        return FIGURE_SIZES[kind]
    except KeyError as error:
        raise ValueError(f"unknown publication figure size: {kind}") from error


def method_style(method: str) -> dict[str, str]:
    return METHOD_STYLE.get(
        method,
        {"label": method.replace("_", " "), "color": COLORS["proxy"], "marker": "o", "ls": "-"},
    )


def style_axes(axis: Any, *, grid_axis: str = "y") -> Any:
    """Apply the shared print-safe axes treatment to an existing axis."""

    axis.set_facecolor("white")
    axis.grid(False)
    if grid_axis in {"x", "both"}:
        axis.grid(axis="x", color=COLORS["grid"], linewidth=0.4, alpha=0.42)
    if grid_axis in {"y", "both"}:
        axis.grid(axis="y", color=COLORS["grid"], linewidth=0.4, alpha=0.42)
    axis.tick_params(direction="out", pad=2.0, colors=COLORS["text"])
    for name, spine in axis.spines.items():
        spine.set_linewidth(0.65)
        spine.set_color(COLORS["text"])
        spine.set_visible(name not in {"top", "right"})
    return axis


def panel_title(axis: Any, label: str, title: str) -> Any:
    """Set a compact left-aligned panel title with a stable label format."""

    title_text = f"({label}) {title}" if label else title
    axis.set_title(
        title_text,
        loc="left",
        pad=4.0,
        fontsize=TEXT_SIZES["panel"],
        fontweight="bold",
        color=COLORS["text"],
    )
    return axis


def figure_legend(
    figure: Any,
    handles: list[Any],
    labels: list[str],
    *,
    ncol: int | None = None,
    y: float = 0.995,
) -> Any:
    """Place one compact, figure-level legend outside the data panels."""

    return figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol or max(1, min(len(labels), 4)),
        frameon=False,
        fontsize=TEXT_SIZES["legend"],
        handlelength=2.0,
        handletextpad=0.45,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
