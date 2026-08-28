from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def paper_context_audit(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    pdf = root / "paper/iclr2027/pivot_iclr2027_submission.pdf"
    release = root / "paper/iclr2027/figures/release"
    pages: int | None = None
    if pdf.is_file():
        try:
            info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
            for line in info.splitlines():
                if line.startswith("Pages:"):
                    pages = int(line.split(":", 1)[1].strip())
                    break
        except (OSError, subprocess.CalledProcessError, ValueError):
            pages = None
    figures = sorted(path.name for path in release.glob("*.png")) if release.is_dir() else []
    result = {
        "pdf_exists": pdf.is_file(),
        "pdf_pages": pages,
        "release_png_count": len(figures),
        "release_figures": figures,
        "status": "PASS" if pdf.is_file() and pages is not None and figures else "BLOCKED",
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit figures in the rendered paper context")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(paper_context_audit(args.root), sort_keys=True))
