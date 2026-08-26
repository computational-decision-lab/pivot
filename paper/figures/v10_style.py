"""Publication style shared by the V10 figure builders.

The style deliberately uses both color and redundant line/marker encodings so
the figures remain legible when printed or viewed in grayscale.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib as mpl

STYLE_VERSION = "pivot-v10-publication-style-1"
COLORS = {
    "proxy": "#6B7280",
    "global": "#2F6DAE",
    "lucb": "#9467BD",
    "pivot": "#007C83",
    "oracle": "#1F2937",
    "direct": "#8C8C8C",
    "actor": "#2F6DAE",
    "strategic": "#B23A48",
    "positive": "#2E8B57",
    "negative": "#B23A48",
    "text": "#202124",
    "grid": "#D6D9DE",
    "shade": "#F2F4F7",
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
    "pivot_voi": {"label": "PIVOT-VOI", "color": COLORS["pivot"], "marker": "D", "ls": "-"},
    "all_hf": {"label": "All-HF oracle", "color": COLORS["oracle"], "marker": "x", "ls": "--"},
}


def apply() -> None:
    """Set deterministic, vector-friendly defaults for paper figures."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "axes.edgecolor": COLORS["text"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.45,
            "grid.alpha": 0.7,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.5,
            "svg.fonttype": "none",
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
