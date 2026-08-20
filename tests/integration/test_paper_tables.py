from __future__ import annotations

from pathlib import Path

from scripts.build_paper_tables import build_tables


def test_paper_tables_render_from_snapshot() -> None:
    paths = build_tables(Path("paper/snapshot"), Path("/tmp/pivot-paper-tables-test"))
    assert len(paths) == 3
    assert all(path.exists() and path.read_text(encoding="utf-8").strip() for path in paths)
