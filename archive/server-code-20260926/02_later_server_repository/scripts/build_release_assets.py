"""Materialize neutral paper-facing aliases for frozen figure assets.

The historical build directories remain available for provenance, but the
manuscript should not expose internal iteration labels in its source or PDF.
This helper copies already-rendered assets; it never changes data or reruns an
experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

# Support both ``python scripts/...`` and package imports from the repository
# root, as used by the supplementary rebuild instructions.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_paper_snippets import _neutralize_macro_text


def _neutralize_metadata(path: Path) -> None:
    """Rewrite copied metadata so release provenance has no iteration labels."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    def clean(value: object) -> object:
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, str):
            value = value.replace("scripts/build_v10_paper_snippets.py", "scripts/build_paper_snippets.py")
            return re.sub(r"(?i)\bv(?:10|11)\b", "release", value)
        return value

    cleaned = clean(payload)
    if isinstance(cleaned, dict) and "analysis_script" in cleaned:
        cleaned["analysis_script"] = "publication_figure_builder"
    if isinstance(cleaned, dict) and "style_version" in cleaned:
        cleaned["style_version"] = "pivot-publication-style-1"
    path.write_text(json.dumps(cleaned, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build(root: Path) -> list[Path]:
    root = Path(root).resolve()
    # Historical rendered assets stay under their provenance directory; the
    # release-facing tree is neutral and is the only path used by LaTeX.
    source = root / "paper/iclr2027/figures/v10"
    target = root / "paper/iclr2027/figures/release"
    if not source.is_dir():
        raise FileNotFoundError(source)
    target.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        destination = target / path.name
        shutil.copyfile(path, destination)
        if destination.suffix.casefold() == ".json":
            _neutralize_metadata(destination)
        copied.append(destination)
    architecture = root / "paper/iclr2027/figures/fig3_pivot_architecture"
    for suffix in (".pdf", ".svg", ".png", ".meta.json"):
        source_path = architecture.with_suffix(suffix)
        if source_path.is_file():
            destination = target / f"fig3_pivot_architecture{suffix}"
            shutil.copyfile(source_path, destination)
            if destination.suffix.casefold() == ".json":
                _neutralize_metadata(destination)
            copied.append(destination)
    macros = root / "paper/iclr2027/results_macros.tex"
    if not macros.is_file():
        # Build from the frozen semantic source only when the neutral macro
        # file has not yet been materialized.
        macros = root / "paper/iclr2027/v10_results_macros.tex"
    if not macros.is_file():
        raise FileNotFoundError(macros)
    neutral_macros = root / "paper/iclr2027/results_macros.tex"
    text = _neutralize_macro_text(root, macros.read_text(encoding="utf-8"))
    neutral_macros.write_text(text, encoding="utf-8")
    copied.append(neutral_macros)
    return copied


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_public_release(root: Path, output: Path | None = None, *, force: bool = False) -> list[Path]:
    """Build the small, reviewer-safe handoff directory.

    Raw DEV traces, sandbox trees, credentials, and local runtime installations
    stay outside this directory.  The supplementary archive is already built
    by the paper finalizer and is treated as the source-of-truth public source
    release.
    """

    root = Path(root).resolve()
    destination = (output or root / "release/v15").resolve()
    if destination.exists() and any(destination.iterdir()) and not force:
        raise FileExistsError(f"release directory is non-empty: {destination}; pass --force to rebuild")
    destination.mkdir(parents=True, exist_ok=True)
    if force:
        for stale in destination.iterdir():
            if stale.is_file() or stale.is_symlink():
                stale.unlink()
            else:
                raise IsADirectoryError(f"refusing to remove unexpected release subdirectory: {stale}")

    pdf = root / "paper/iclr2027/pivot_iclr2027_submission.pdf"
    supplement = root / "paper/iclr2027/pivot_iclr2027_supplementary.zip"
    verification = root / "paper/iclr2027/submission_verification.json"
    lock = root / "experiments/v15/confirmatory_lock.json"
    final_report = root / "V15_FINAL_REPORT.md"
    for path in (pdf, supplement, verification, lock, final_report):
        if not path.is_file():
            raise FileNotFoundError(path)

    copied: list[Path] = []
    mapping = {
        pdf: destination / "paper.pdf",
        supplement: destination / "supplementary.zip",
        lock: destination / "confirmatory_lock.json",
        final_report: destination / "V15_FINAL_REPORT.md",
    }
    for source, target in mapping.items():
        shutil.copyfile(source, target)
        copied.append(target)

    raw_verification = json.loads(verification.read_text(encoding="utf-8"))
    # Keep the release summary useful without copying the large member list or
    # any path-bearing diagnostic payload into a top-level public artifact.
    public_verification = {
        "package": raw_verification.get("package"),
        "decision": raw_verification.get("decision"),
        "submission_ready": raw_verification.get("submission_ready"),
        "blocking_gates": raw_verification.get("blocking_gates", []),
        "pdf_pages": raw_verification.get("pdf_pages"),
        "main_pages": raw_verification.get("main_pages"),
        "machine_checks": raw_verification.get("machine_checks", {}),
        "archive_content_checks": raw_verification.get("archive_content_checks", {}),
        "public_release_note": "The confirmatory modern-agent study remains pre-outcome and is not represented as completed evidence.",
    }
    verification_target = destination / "submission_verification.json"
    verification_target.write_text(json.dumps(public_verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    copied.append(verification_target)

    readme = destination / "README.md"
    readme.write_text(
        "# PIVOT ICLR 2027 Curated Release\n\n"
        "This directory contains the anonymous manuscript and the deterministic\n"
        "supplementary archive. The local release/audit layer is complete, while\n"
        "the registered confirmatory modern-agent phases remain pre-outcome and\n"
        "must not be read as completed scientific evidence.\n\n"
        "## Files\n\n"
        "- `paper.pdf`: anonymous manuscript.\n"
        "- `supplementary.zip`: sanitized source, figures, tables, and audit artifacts.\n"
        "- `confirmatory_lock.json`: immutable pre-outcome protocol lock.\n"
        "- `submission_verification.json`: compact machine audit summary.\n"
        "- `V15_FINAL_REPORT.md`: explicit status and remaining gates.\n"
        "- `SHA256SUMS`: hashes for every release file except the checksum file.\n\n"
        "No raw Inspect traces, sandbox trees, candidate archives, private paths,\n"
        "credentials, or local runtime installations are included. Rebuild the\n"
        "paper and supplement with `make v15-finalize` from the repository root.\n"
        "Figure metadata uses the tracked build provenance anchor in\n"
        "`configs/v15/build_provenance.json` so clean rebuilds preserve the\n"
        "hash-bound visual review.\n",
        encoding="utf-8",
    )
    copied.append(readme)

    checksums = destination / "SHA256SUMS"
    checksum_lines = [f"{_sha256(path)}  {path.name}" for path in sorted(copied, key=lambda item: item.name)]
    checksums.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    copied.append(checksums)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Build neutral paper-facing asset aliases")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--public-release", action="store_true", help="also build the curated release/v15 handoff")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="replace an existing non-empty public release directory")
    args = parser.parse_args()
    paths = build_public_release(args.root, args.output, force=args.force) if args.public_release else build(args.root)
    print({"copied": len(paths), "target": str(args.output or (args.root / "release/v15")) if args.public_release else "paper/iclr2027/figures/release"})


if __name__ == "__main__":
    main()
