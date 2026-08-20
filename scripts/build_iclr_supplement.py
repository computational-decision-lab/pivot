#!/usr/bin/env python3
"""Build a deterministic, anonymized ICLR supplementary archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

ALLOWLIST = ("src", "scripts", "experiments", "configs", "tests", "benchmarks/improvementbench/v1")
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".tex", ".bib", ".txt", ".csv"}
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".venv", "build", "pivot_research.egg-info"}
PRIVATE_RE = re.compile(r"(?:/opt/projects|/home/ubuntu|/tmp(?:/[A-Za-z0-9_.-]*)?)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _copy_sanitized(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.casefold() in TEXT_SUFFIXES:
        text = source.read_text(encoding="utf-8", errors="replace")
        text = PRIVATE_RE.sub("<local-root>", text)
        text = EMAIL_RE.sub("<redacted-email>", text)
        target.write_text(text, encoding="utf-8")
    else:
        shutil.copyfile(source, target)


def build_supplement(project_root: Path, output_root: Path) -> list[Path]:
    """Copy the public allowlist and sanitize local provenance strings."""

    if output_root.exists():
        for path in sorted(output_root.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    output_root.mkdir(parents=True, exist_ok=True)
    original_manifest = project_root / "paper" / "snapshot" / "manifest.json"
    original_manifest_sha256 = hashlib.sha256(original_manifest.read_bytes()).hexdigest()
    copied: list[Path] = []
    for root_name in ALLOWLIST:
        root = project_root / root_name
        for source in sorted(root.rglob("*")):
            if not source.is_file() or any(part in SKIP_PARTS for part in source.parts):
                continue
            relative = source.relative_to(project_root)
            target = output_root / relative
            _copy_sanitized(source, target)
            copied.append(target)
    for source in sorted((project_root / "paper" / "snapshot").rglob("*")):
        if not source.is_file():
            continue
        relative = Path("snapshot") / source.relative_to(project_root / "paper" / "snapshot")
        target = output_root / relative
        _copy_sanitized(source, target)
        copied.append(target)
    for name in ("controlled_results.tex", "public_results.tex", "ablation_results.tex"):
        source = project_root / "paper" / "tables" / name
        target = output_root / "tables" / name
        _copy_sanitized(source, target)
        copied.append(target)
    _rewrite_snapshot_manifest(
        output_root / "snapshot" / "manifest.json",
        original_manifest_sha256,
    )

    readme = output_root / "README.md"
    readme.write_text(
        """# IMPROVE-X / PIVOT ICLR 2027 Supplementary Artifact

This archive contains the anonymous source, configurations, tests, the
controlled ImprovementBench release, and the hash-indexed paper snapshot used
for the submission PDF. From the
repository root, install the project in editable mode and run:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/python scripts/build_paper_tables.py --snapshot paper/snapshot --output paper/tables
```

The public finance audit uses virtual fills and observational depth
proxies. No credentials, private data, raw vendor archives, live
orders, or author identity metadata are included.
""",
        encoding="utf-8",
    )
    copied.append(readme)
    return copied


def _rewrite_snapshot_manifest(path: Path, original_sha256: str) -> None:
    """Bind manifest hashes to sanitized files in the supplement."""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    for relative, entry in manifest.get("files", {}).items():
        candidate = path.parent / relative
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        entry["sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        entry["size_bytes"] = candidate.stat().st_size
    manifest["source_roots"] = {key: "<local-root>" for key in manifest.get("source_roots", {})}
    manifest["sanitized_for_submission"] = True
    manifest["original_manifest_sha256"] = original_sha256
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_deterministic_zip(root: Path, archive: Path) -> str:
    """Write sorted files with fixed timestamps and return SHA-256."""

    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            output.writestr(info, path.read_bytes())
    return hashlib.sha256(archive.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PIVOT ICLR supplementary archive")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    files = build_supplement(args.project_root.resolve(), args.output_root.resolve())
    digest = write_deterministic_zip(args.output_root.resolve(), args.archive.resolve())
    manifest = {
        "file_count": len(files),
        "archive": args.archive.name,
        "sha256": digest,
        "snapshot_manifest_sha256": hashlib.sha256(
            (args.output_root / "snapshot" / "manifest.json").read_bytes()
        ).hexdigest(),
        "allowlist": list(ALLOWLIST) + ["paper/snapshot", "paper/tables"],
    }
    (args.output_root / "supplement_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
