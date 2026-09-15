from __future__ import annotations

from pathlib import Path

import pytest

from scripts.verify_paper import (
    _parse_fonts,
    _portable_path,
    parse_appendix_start_page,
    parse_references_start_page,
    scan_log,
)


def test_portable_path_avoids_machine_absolute_prefix() -> None:
    rendered = _portable_path(Path.cwd() / "paper" / "main.tex")
    assert rendered == "paper/main.tex"
    assert "/opt/projects/" not in rendered


def test_parse_appendix_start_page_reads_latex_aux_label() -> None:
    aux = r"""\relax
\newlabel{app:artifact}{{A}{10}}
"""
    assert parse_appendix_start_page(aux) == 10


def test_parse_appendix_start_page_requires_label() -> None:
    with pytest.raises(ValueError, match="appendix label"):
        parse_appendix_start_page(r"\relax")


def test_parse_references_start_page_reads_latex_aux_label() -> None:
    aux = r"""\relax
\newlabel{refs:start}{{}{10}}
"""
    assert parse_references_start_page(aux) == 10


def test_parse_references_start_page_requires_label() -> None:
    with pytest.raises(ValueError, match="references label"):
        parse_references_start_page(r"\relax")


def test_parse_fonts_uses_embedded_column_not_subset_column() -> None:
    output = """name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
ABCDEF+Times-Roman                  Type 1            Custom           yes yes yes     10  0
Unembedded                            Type 1            Custom           no  yes yes     11  0
"""
    fonts = _parse_fonts(output)
    assert [font["embedded"] for font in fonts] == [True, False]


def test_scan_log_rejects_undefined_references_and_overfull_boxes() -> None:
    result = scan_log("LaTeX Warning: Reference `x' undefined\nOverfull \\hbox (3.0pt too wide)")
    assert result["undefined_references"] is True
    assert result["overfull_hboxes"] == 1
