"""Resolve the local assets referenced by the frozen LaTeX manuscript."""

from __future__ import annotations

import re
from pathlib import Path


def manuscript_assets(source: Path) -> list[Path]:
    """Return the source, included TeX, bibliography, and referenced graphics.

    Resolve paths as a build from the manuscript directory would, including
    graphicspath entries. Missing assets fail the release instead of silently
    falling back to a stale figure or an incomplete supplementary archive.
    """

    source = source.resolve()
    paper_root = source.parent
    pending = [source]
    assets: set[Path] = set()
    graphic_dirs = [paper_root]

    def resolve(name: str, directories: list[Path], suffixes: tuple[str, ...]) -> Path:
        for directory in directories:
            for suffix in ("", *suffixes):
                candidate = (directory / (name + suffix)).resolve()
                if not candidate.is_relative_to(paper_root):
                    raise ValueError(f"Manuscript asset escapes paper directory: {name}")
                if candidate.is_file():
                    return candidate
        raise FileNotFoundError(f"Missing manuscript asset: {name}")

    while pending:
        tex = pending.pop()
        if tex in assets:
            continue
        assets.add(tex)
        text = re.sub(r"(?m)(?<!\\)%.*$", "", tex.read_text(encoding="utf-8"))
        for block in re.findall(r"\\graphicspath\s*\{((?:\s*\{[^{}]*\})+)\s*\}", text):
            graphic_dirs.extend(paper_root / name for name in re.findall(r"\{([^{}]*)\}", block))
        for name in re.findall(r"\\(?:input|include)\s*\{([^{}]+)\}", text):
            pending.append(resolve(name, [paper_root], (".tex",)))
        for name in re.findall(r"\\includegraphics\*?\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}", text):
            assets.add(resolve(name, graphic_dirs, (".pdf", ".png", ".jpg", ".jpeg")))
        for names in re.findall(r"\\bibliography\s*\{([^{}]+)\}", text):
            for name in names.split(","):
                assets.add(resolve(name.strip(), [paper_root], (".bib",)))
    return sorted(assets)
