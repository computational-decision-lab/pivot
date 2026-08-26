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

ALLOWLIST = (
    "src",
    "scripts",
    "experiments",
    "configs",
    "results/theory",
    "results/v7",
    "results/v9",
    "figures/v9",
    "figures/v10",
    "tables/v9",
    "artifacts/v9",
    "artifacts/v10",
    "snapshot/v9_preupgrade",
    "research",
    "tests",
    "benchmarks/improvementbench/v1",
    "benchmarks/improvementbench/v2",
    "benchmarks/improvementbench/v7",
    "paper/figures/v10_style.py",
)
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".tex", ".bib", ".txt", ".csv"}
SKIP_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    "build",
    "pivot_research.egg-info",
}
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
        if root.is_file():
            sources = [root]
        elif root.is_dir():
            sources = sorted(root.rglob("*"))
        else:
            raise FileNotFoundError(root)
        for source in sources:
            if not source.is_file() or any(part in SKIP_PARTS for part in source.parts):
                continue
            # Parquet is retained in the repository artifact root, but the
            # anonymous supplement audit intentionally excludes raw archive
            # formats. CSV/JSON and vector sources remain fully auditable.
            if (
                source.suffix.casefold() in {".parquet", ".feather", ".zip", ".tar", ".gz"}
                and "results/v9" not in source.as_posix()
            ):
                continue
            if source.suffix.casefold() == ".parquet":
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
    for relative_path in ("paper/iclr2027/v10_results_macros.tex",):
        source = project_root / relative_path
        target = output_root / relative_path
        _copy_sanitized(source, target)
        copied.append(target)
    for relative_path in (
        "docs/v7-evidence-2026-08-25.md",
        "docs/external-environment-provenance.md",
        "docs/claim_boundary.md",
        "V10_PRE_FINAL_AUDIT.md",
        "V10_METRIC_SCALE_AUDIT.md",
        "V10_NUMBER_AUDIT.md",
        "V10_FIGURE_AUDIT.md",
        "V10_REVIEWER_ATTACK_AUDIT.md",
        "V10_CLAIM_AUDIT.md",
        "V10_BIBLIOGRAPHY_AUDIT.md",
        "V10_FINAL_REPORT.md",
    ):
        source = project_root / relative_path
        target = output_root / relative_path
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
controlled ImprovementBench v1/v2 releases, and the hash-indexed paper
snapshot used for the submission PDF. From the
repository root, install the project in editable mode and run:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/python scripts/build_paper_tables.py --snapshot paper/snapshot --output paper/tables
```

The public finance audit uses virtual fills and observational depth
proxies. No credentials, private data, raw vendor archives, live
orders, or author identity metadata are included.

The controlled value-versus-improvement diagnostic can be regenerated with:

```bash
.venv/bin/python experiments/e4_value_vs_improvement.py \\
  --config configs/sweeps/e4_value_vs_improvement.yaml \\
  --output artifacts/v9/reproduction/e4-value-vs-improvement
```

This is a controlled estimand diagnostic, not a universal method or market
performance claim.

Earlier controlled checks and frozen external classifications remain under
`results/theory` and `results/v7` for provenance. Decompress a row stream with
`gzip -dk transition_rows.jsonl.gz` when a row-level audit is needed. The
analytic checks can be regenerated with:

```bash
.venv/bin/python experiments/e10_theory_empirical.py \\
  --config configs/theory/v6_empirical.yaml \\
  --output artifacts/v9/reproduction/e10-theory-empirical
```

They test the constructive Global Fidelity Blindness and Response-Footprint
Sensitivity claims; they are not causal market evidence.

The frozen confirmatory package is included under `results/v9`; publication
transforms are under `figures/v10` and `experiments/v10`. They do not rerun
science: they read hash-indexed source rows and emit PDF/SVG/PNG figures plus
CSV provenance tables. Rebuild and audit the complete package with:

```bash
.venv/bin/python -m experiments.v10.finalize --root .
```

The manuscript reports the scientific names of the evidence layers. Internal
run identifiers are retained only in manifests and source metadata.

The editable architecture source is adapted from the pinned OpenTikZ
`system-block-diagram` template. Rebuild it after installing the lock-bound
checkout with `scripts/bootstrap_opentikz.py` and
`scripts/build_opentikz_architecture.py`.
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
        "allowlist": list(ALLOWLIST)
        + ["paper/snapshot", "paper/tables", "paper/iclr2027/v10_results_macros.tex"],
    }
    (args.output_root / "supplement_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
