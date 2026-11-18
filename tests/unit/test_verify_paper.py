from __future__ import annotations

from pathlib import Path

import pytest

from scripts.verify_paper import (
    _parse_fonts,
    _portable_path,
    _resolve_aux,
    parse_conclusion_end_page,
    parse_appendix_start_page,
    parse_references_start_page,
    scan_log,
)
from scripts.verify_iclr_submission import _conclusion_end_page


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


def test_parse_conclusion_end_page_reads_latex_aux_label() -> None:
    aux = r"""\relax
\newlabel{sec:conclusion-end}{{}{10}}
"""
    assert parse_conclusion_end_page(aux) == 10


def test_parse_conclusion_end_page_requires_label() -> None:
    with pytest.raises(ValueError, match="conclusion end label"):
        parse_conclusion_end_page(r"\relax")


def test_submission_verifier_reads_conclusion_end_page_from_aux(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    aux = tmp_path / "main.aux"
    aux.write_text(r"\newlabel{sec:conclusion-end}{{}{11}}", encoding="utf-8")

    assert _conclusion_end_page(pdf, aux) == 11


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


def test_resolve_aux_accepts_build_sidecar_for_submission_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    sidecar = tmp_path / "build" / "main.aux"
    sidecar.parent.mkdir()
    sidecar.write_text("\\relax\n", encoding="utf-8")

    assert _resolve_aux(pdf) == sidecar.resolve()


def test_resolve_aux_prefers_explicit_sidecar(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    explicit = tmp_path / "locked.aux"
    explicit.write_text("\\relax\n", encoding="utf-8")

    assert _resolve_aux(pdf, explicit) == explicit.resolve()
