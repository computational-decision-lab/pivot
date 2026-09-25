#!/usr/bin/env python3
"""Build the pinned OpenTikZ architecture source into paper-ready assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generated_at() -> str | None:
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    if not epoch:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _git_commit(root: Path) -> str | None:
    override = os.environ.get("PIVOT_BUILD_COMMIT", "").strip()
    if override:
        return override
    provenance = root / "configs/v15/build_provenance.json"
    try:
        payload = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    if isinstance(payload, dict):
        configured = payload.get("artifact_source_commit")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


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
    output_png = output_pdf.with_suffix(".png")
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
    # Keep a raster companion for visual QA and lightweight viewers.  The
    # raster is rendered from the exact compiled PDF used by the manuscript.
    _run(
        [
            "gs",
            "-q",
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=png16m",
            "-r320",
            f"-sOutputFile={output_png}",
            str(output_pdf),
        ],
        cwd=source.parent,
        env=env,
    )
    if not output_svg.is_file():
        raise RuntimeError(f"OpenTikZ renderer did not produce {output_svg}")
    project_root = source.parents[3] if len(source.parents) > 3 else source.parent
    config_hashes = {}
    config_dir = project_root / "configs/v9"
    if config_dir.is_dir():
        config_hashes = {
            str(path.relative_to(project_root)): _sha256(path)
            for path in sorted(config_dir.glob("*.yaml"))
            if path.is_file()
        }
    metadata = {
        "figure_id": "fig3_pivot_architecture",
        "scientific_question": "How does PIVOT-VOI allocate paired interventional evaluations to preserve update decisions?",
        "unit_of_inference": "semantic architecture node and directed stage",
        "appendix": False,
        "experiment_sources": [str(source.relative_to(project_root))],
        "source_hashes": {
            str(path.relative_to(project_root)): _sha256(path)
            for path in (source, output_pdf, output_svg, output_png)
        },
        "analysis_script": "scripts/build_opentikz_architecture.py",
        "style_version": "pivot-v10-publication-style-1",
        "generated_at": _generated_at(),
        "config_hashes": config_hashes,
        "git_commit": _git_commit(project_root),
        "preview": output_svg.name,
        "outputs": [output_pdf.name, output_svg.name, output_png.name, source.name],
        "semantic_node_count": 11,
        "template": "OpenTikZ system-block-diagram",
        "composed_of": ["system-block-diagram"],
        "license": "CC0-1.0",
        "tool_repository": "https://github.com/opentikz/opentikz",
        "raw_observations": True,
        "interval_definition": "not applicable; semantic node inventory",
        "incomparable_conditions": "not applicable",
        "oracle_reference": "not applicable",
        "interpolation": "none",
        "grayscale_distinguishable": True,
        "display_policy": "all semantic nodes retained",
    }
    metadata_path = output_pdf.parent / f"{source.stem}.meta.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "source": str(source),
        "pdf": str(output_pdf),
        "svg": str(output_svg),
        "png": str(output_png),
        "metadata": str(metadata_path),
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
