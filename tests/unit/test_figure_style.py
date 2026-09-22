from __future__ import annotations

from pathlib import Path

import matplotlib
from pytest import MonkeyPatch

from paper.figures.v10_style import (
    FIGURE_SIZES,
    STYLE_VERSION,
    TEXT_SIZES,
    apply,
    figure_size,
    save,
)
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

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1787227200")

    def render(stem: Path) -> bytes:
        apply()
        figure, axis = plt.subplots()
        axis.plot([0, 1], [1, 0])
        save(figure, stem, formats=("svg",))
        plt.close(figure)
        return stem.with_suffix(".svg").read_bytes()

    assert render(tmp_path / "first") == render(tmp_path / "second")


def test_v10_style_registers_shared_canvas_and_text_tokens() -> None:
    apply()
    assert figure_size("wide") == FIGURE_SIZES["wide"]
    assert figure_size("wide_tall") == FIGURE_SIZES["wide_tall"]
    assert STYLE_VERSION.startswith("pivot-v11-")
    assert min(TEXT_SIZES.values()) >= 6.5
    assert matplotlib.rcParams["font.family"] == ["DejaVu Sans"]
    assert matplotlib.rcParams["mathtext.fontset"] == "dejavusans"


def test_main_results_table_names_estimands_and_gain_direction() -> None:
    from scripts.build_revision_evidence import render_main_results_table

    rows = {
        key: {"mean_isr": value}
        for key, value in {
            "leduc_uniform": 0.2,
            "leduc_pivot": 0.1,
            "kuhn_uniform": 0.2,
            "kuhn_pivot": 0.2,
            "v4_uniform": 0.3,
            "v4_pivot": 0.3,
            "v4_one_pivot": 0.2,
            "v4_short_uniform": 0.4,
            "v4_short_pivot": 0.5,
        }.items()
    }
    statistic = {"mean": 0.1, "lo": 0.0, "hi": 0.2}
    rendered = render_main_results_table(
        rows,
        statistic,
        {"mean": 0.0, "lo": -0.1, "hi": 0.1},
        {"mean": 0.0, "lo": -0.1, "hi": 0.1},
        statistic,
        {"mean": -0.1, "lo": -0.2, "hi": 0.0},
    )
    assert "Uniform ISR" in rendered
    assert "PIVOT-KG ISR" in rendered
    assert "Gain" in rendered
    assert "Interpretation" in rendered
    assert "Status" not in rendered
    assert "PIVOT--Uniform" not in rendered
