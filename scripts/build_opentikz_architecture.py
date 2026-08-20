#!/usr/bin/env python3
"""Build the pinned OpenTikZ architecture source into paper-ready assets."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-30:])
        raise RuntimeError(f"command failed ({' '.join(command)}):\n{detail}")


def build_architecture(
    source: Path,
    output_pdf: Path,
    output_svg: Path,
    opentikz_root: Path,
) -> dict[str, str]:
    """Compile a standalone architecture source and render its SVG preview."""

    source = source.resolve()
    output_pdf = output_pdf.resolve()
    output_svg = output_svg.resolve()
    opentikz_root = opentikz_root.resolve()
    renderer = opentikz_root / "tools" / "render_preview.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    if not renderer.is_file():
        raise FileNotFoundError(renderer)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("SOURCE_DATE_EPOCH", "1787227200")
    with tempfile.TemporaryDirectory(prefix="pivot-opentikz-build-") as temporary:
        build_dir = Path(temporary)
        _run(
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={build_dir}",
                source.name,
            ],
            cwd=source.parent,
            env=env,
        )
        compiled_pdf = build_dir / f"{source.stem}.pdf"
        if not compiled_pdf.is_file():
            raise RuntimeError(f"latexmk did not produce {compiled_pdf}")
        shutil.copy2(compiled_pdf, output_pdf)
        _run(
            [
                sys.executable,
                str(renderer),
                str(source),
                "--backend",
                "dvisvgm",
                "--output",
                str(output_svg),
            ],
            cwd=source.parent,
            env=env,
        )
    if not output_svg.is_file():
        raise RuntimeError(f"OpenTikZ renderer did not produce {output_svg}")
    return {
        "source": str(source),
        "pdf": str(output_pdf),
        "svg": str(output_svg),
        "opentikz": str(opentikz_root),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "paper/iclr2027/figures/fig3_pivot_architecture.tex",
    )
    parser.add_argument("--output-pdf", type=Path)
    parser.add_argument("--output-svg", type=Path)
    parser.add_argument(
        "--opentikz-root",
        type=Path,
        default=project_root / ".tools/opentikz",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    output_pdf = (args.output_pdf or source.with_suffix(".pdf")).resolve()
    output_svg = (args.output_svg or source.with_suffix(".svg")).resolve()
    result = build_architecture(source, output_pdf, output_svg, args.opentikz_root)
    print(result)


if __name__ == "__main__":
    main()
