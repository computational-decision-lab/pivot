from __future__ import annotations

import pytest

from scripts.verify_paper import parse_appendix_start_page, scan_log


def test_parse_appendix_start_page_reads_latex_aux_label() -> None:
    aux = r"""\relax
\newlabel{app:artifact}{{A}{10}}
"""
    assert parse_appendix_start_page(aux) == 10


def test_parse_appendix_start_page_requires_label() -> None:
    with pytest.raises(ValueError, match="appendix label"):
        parse_appendix_start_page(r"\relax")


def test_scan_log_rejects_undefined_references_and_overfull_boxes() -> None:
    result = scan_log("LaTeX Warning: Reference `x' undefined\nOverfull \\hbox (3.0pt too wide)")
    assert result["undefined_references"] is True
    assert result["overfull_hboxes"] == 1
