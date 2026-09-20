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
STYLE_VERSION = "pivot-v10-scientific-figure-suite-1"
COLORS = {
    "proxy": "#6B7280",
    "global": "#0072B2",
    "lucb": "#CC79A7",
    "pivot": "#009E73",
    "oracle": "#2F2F2F",
    "direct": "#6C757D",
    "actor": "#0072B2",
    "strategic": "#D55E00",
    "positive": "#009E73",
    "negative": "#D55E00",
    "cohort_leduc": "#0072B2",
    "cohort_kuhn": "#CC79A7",
    "cohort_melting_pot": "#D55E00",
    "text": "#202124",
    "grid": "#D9E2E8",
    "shade": "#F4F6F8",
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
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "axes.edgecolor": COLORS["text"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.45,
            "grid.alpha": 0.55,
            "legend.frameon": False,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.2,
            "axes.axisbelow": True,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "svg.fonttype": "none",
            # Matplotlib otherwise generates random SVG element identifiers,
            # which makes byte-identical figure rebuilds impossible.
            "svg.hashsalt": "pivot-v10-scientific-figure-suite-1",
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
        figure.savefig(target, dpi=320, bbox_inches="tight", pad_inches=0.04)
        outputs.append(target)
    return outputs


def method_style(method: str) -> dict[str, str]:
    return METHOD_STYLE.get(
        method,
        {"label": method.replace("_", " "), "color": COLORS["proxy"], "marker": "o", "ls": "-"},
    )
