"""Central V9 figure-style import used by reproducible builders."""

from scripts.figure_style import DEFAULT_COLORS, PALETTE, FigureStyle, apply_publication_style, finalize_figure, reversal_cmap

STYLE_VERSION = "pivot-v9-figures4papers-adapter-1"

__all__ = ["DEFAULT_COLORS", "PALETTE", "FigureStyle", "STYLE_VERSION", "apply_publication_style", "finalize_figure", "reversal_cmap"]
