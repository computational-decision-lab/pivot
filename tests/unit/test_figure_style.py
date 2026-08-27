from __future__ import annotations

from pathlib import Path

import matplotlib
from pytest import MonkeyPatch

from scripts.figure_style import PALETTE, FigureStyle, apply_publication_style, finalize_figure


def test_figure_style_exposes_semantic_palette_and_publication_rcparams() -> None:
    apply_publication_style(FigureStyle(font_size=10, axes_linewidth=1.5))
    assert PALETTE["blue_main"] == "#0F4D92"
    assert PALETTE["red_strong"] == "#B64342"
    assert matplotlib.rcParams["axes.spines.top"] is False
    assert matplotlib.rcParams["axes.spines.right"] is False
    assert matplotlib.rcParams["legend.frameon"] is False
    assert matplotlib.rcParams["svg.fonttype"] == "none"


def test_finalize_figure_writes_requested_publication_formats(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    outputs = finalize_figure(figure, tmp_path / "example", formats=("png", "pdf"), dpi=300)
    assert {path.suffix for path in outputs} == {".png", ".pdf"}
    assert all(path.is_file() and path.stat().st_size > 100 for path in outputs)


def test_v10_svg_exports_are_byte_stable(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    import matplotlib.pyplot as plt

    from paper.figures.v10_style import apply, save

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1787227200")

    def render(stem: Path) -> bytes:
        apply()
        figure, axis = plt.subplots()
        axis.plot([0, 1], [1, 0])
        save(figure, stem, formats=("svg",))
        plt.close(figure)
        return stem.with_suffix(".svg").read_bytes()

    assert render(tmp_path / "first") == render(tmp_path / "second")
