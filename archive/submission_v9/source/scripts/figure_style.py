"""Shared publication style adapted from the locked figure skill repositories.

The upstream repositories provide conventions rather than a runtime dependency.
Keeping the small adapter here makes figure generation deterministic and keeps
the public repository free of vendored third-party code.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "text": "#272727",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

DEFAULT_COLORS = (
    PALETTE["blue_main"],
    PALETTE["green_3"],
    PALETTE["red_strong"],
    PALETTE["teal"],
    PALETTE["violet"],
    PALETTE["neutral"],
)


@dataclass(frozen=True)
class FigureStyle:
    """Portable subset of the figures4papers publication contract."""

    font_size: int = 10
    axes_linewidth: float = 1.2
    font_family: tuple[str, ...] = ("DejaVu Sans", "Arial", "Helvetica", "sans-serif")


def apply_publication_style(style: FigureStyle | None = None) -> None:
    """Apply consistent typography, axes, legends, and editable SVG text."""

    import matplotlib as mpl

    selected = style or FigureStyle()
    mpl.rcParams.update(
        {
            # Matplotlib treats a family list as a set of sequential lookups
            # and logs every missing family.  Pick the first portable font for
            # deterministic headless builds; the documented stack remains in
            # FigureStyle for environments that provide the journal font.
            "font.family": [selected.font_family[0]],
            "font.size": selected.font_size,
            "axes.linewidth": selected.axes_linewidth,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def reversal_cmap() -> Any:
    """Return a diverging map where reversals are red and gains are green."""

    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "pivot_reversal",
        [PALETTE["red_strong"], PALETTE["neutral"], PALETTE["green_3"]],
    )


def finalize_figure(
    figure: Any,
    out_path: str | Path,
    *,
    formats: Iterable[str] | None = None,
    dpi: int = 300,
    close: bool = True,
    pad: float = 0.05,
) -> list[Path]:
    """Save a figure using stable paper-ready output settings."""

    import matplotlib.pyplot as plt

    base = Path(out_path)
    if formats is None:
        formats = (base.suffix.lstrip(".") or "png",)
    normalized = tuple(str(fmt).lstrip(".").lower() for fmt in formats)
    if not normalized:
        raise ValueError("at least one output format is required")
    if base.suffix:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for fmt in normalized:
        if fmt not in {"pdf", "svg", "eps", "png", "jpg", "jpeg", "tif", "tiff"}:
            raise ValueError(f"unsupported figure format: {fmt}")
        target = base.with_suffix(f".{fmt}")
        figure.savefig(target, dpi=dpi, bbox_inches="tight", pad_inches=pad)
        outputs.append(target)
    if close:
        plt.close(figure)
    return outputs
