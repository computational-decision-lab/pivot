#!/usr/bin/env python3
"""Build a deterministic, anonymized ICLR supplementary archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

# Keep the documented ``python scripts/...`` entry point independent of an
# editable install.  The repository root is the package import root for the
# V15 experiment modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.v15.planes import redact_task_manifest

ALLOWLIST = (
    "src",
    "scripts",
    "experiments",
    "configs",
    "results/theory",
    "results/v7",
    "results/v9",
    "results/v15",
    "figures/v9",
    "figures/v10",
    "figures/v15",
    "tables/v9",
    "artifacts/v9",
    "artifacts/v10",
    "artifacts/v15",
    "snapshot/v9_preupgrade",
    "research",
    "tests",
    "benchmarks/improvementbench/v1",
    "benchmarks/improvementbench/v2",
    "benchmarks/improvementbench/v7",
    "paper/figures/v10_style.py",
)
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".jsonl", ".ndjson", ".md", ".tex", ".bib", ".txt", ".csv"}
SKIP_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    "build",
    "pivot_research.egg-info",
    ".codex",
}
PRIVATE_RE = re.compile(r"(?:/opt/projects|/home/ubuntu|/tmp(?:/[A-Za-z0-9_.-]*)?)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ASSISTANT_RE = re.compile(r"codex", flags=re.IGNORECASE)
API_BASE_FIELD_RE = re.compile(r'("api_base"\s*:\s*)"[^"]*"', flags=re.IGNORECASE)
PUBLIC_PARQUET_PREFIXES = (
    "results/v15/canonical/",
    "figures/v15/",
    "artifacts/v15/",
)
SEALED_HISTORY_RELATIVE = "experiments/v15/confirmatory_lock_history.jsonl"
SEALED_MANIFEST_RELATIVE = "configs/v15/task_manifest.json"
SEALED_LOCK_RELATIVE = "experiments/v15/confirmatory_lock.json"
PUBLIC_MANIFEST_RELATIVE = "configs/v15/task_manifest.public.json"


def _is_public_parquet(relative: Path) -> bool:
    """Return whether a generated Parquet table is safe for the release."""

    normalized = relative.as_posix().lstrip("./")
    return any(normalized.startswith(prefix) for prefix in PUBLIC_PARQUET_PREFIXES)


def _is_public_v15_path(relative: Path) -> bool:
    """Keep V15 schemas and summaries, never raw sandbox/task executions."""

    parts = relative.as_posix().split("/")
    if len(parts) < 3 or parts[0:2] != ["results", "v15"]:
        return True
    if parts[2].startswith("dev-"):
        # DEV execution manifests expose counts and terminal states without
        # carrying task instructions.  Raw DEV rows are reproducible locally
        # but can embed prompts, traces, or final trees, so keep them private.
        return relative.name == "manifest.json"
    private_parts = {
        "artifacts",
        "inspect-logs",
        "final_tree",
        "agent-reviewer-artifacts",
        "agent-reviewer-logs",
    }
    if private_parts.intersection(parts):
        return False
    # JSONL manifests and candidate archives are useful; raw trajectory
    # streams are not. The latter are identified by their filename.
    if relative.suffix.casefold() in {".traj", ".jsonl"} and relative.name.endswith(".traj.json"):
        return False
    return not relative.name.endswith(".execution.json")


def _is_sealed_public_input(relative: Path) -> bool:
    """Return whether a path is a local-only sealed protocol input."""

    normalized = relative.as_posix().lstrip("./")
    return normalized in {SEALED_HISTORY_RELATIVE, SEALED_MANIFEST_RELATIVE}


def _sanitize_json_payload(
    relative: Path, payload: object, *, source_sha256: str | None = None
) -> object:
    """Remove hidden task contents from copied protocol JSON files."""

    normalized = relative.as_posix().lstrip("./")
    if normalized == SEALED_MANIFEST_RELATIVE and isinstance(payload, dict):
        return redact_task_manifest(payload, source_sha256=source_sha256)
    if normalized == SEALED_LOCK_RELATIVE and isinstance(payload, dict):
        cleaned = dict(payload)
        sealed = cleaned.get("sealed_planes")
        if isinstance(sealed, dict):
            cleaned["sealed_planes"] = redact_task_manifest(sealed)
        return cleaned
    return payload


def _copy_sanitized(source: Path, target: Path, relative: Path | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.casefold() in TEXT_SUFFIXES:
        raw_text = source.read_text(encoding="utf-8", errors="replace")
        text = raw_text
        # Apply structural redaction before path/token substitutions so task
        # hashes remain hashes of the sealed source definitions.
        if source.suffix.casefold() == ".json" and relative is not None:
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError:
                payload = None
            if payload is not None:
                text = json.dumps(
                    _sanitize_json_payload(
                        relative,
                        payload,
                        source_sha256=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                    ),
                    indent=2,
                    sort_keys=True,
                ) + "\n"
        text = PRIVATE_RE.sub("<local-root>", text)
        text = EMAIL_RE.sub("<redacted-email>", text)
        # The endpoint host is runtime infrastructure, not reproducibility
        # evidence.  Keep the field name while removing provider topology from
        # reviewer-facing artifacts.
        text = API_BASE_FIELD_RE.sub(r'\1"<external-endpoint>"', text)
        # Reviewer-facing artifacts must not disclose the internal editing
        # assistant.  Use an identifier-safe neutral token so copied Python
        # and JSON remain parseable where possible.
        text = ASSISTANT_RE.sub("implementation_assistant", text)
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
            # Keep generated V15/figure parquet sources: they are compact,
            # schema-bearing artifacts rather than raw vendor archives.  Older
            # frozen archives and compressed blobs remain excluded.
            if source.suffix.casefold() in {".feather", ".zip", ".tar", ".gz"}:
                continue
            if source.suffix.casefold() == ".parquet" and not _is_public_parquet(
                source.relative_to(project_root)
            ):
                # Phase-level external tables retain executor paths for the
                # internal response audit.  Only the path-free canonical V15
                # tables and the already audited figure/artifact tables belong
                # in the anonymous release.
                continue
            relative = source.relative_to(project_root)
            if _is_sealed_public_input(relative):
                # Historical locks may contain pre-redaction task contents;
                # and the source task manifest contains gate/assessment
                # instructions.  Both are internal provenance and are not
                # needed by reviewers.
                continue
            if not _is_public_v15_path(relative):
                continue
            target = output_root / relative
            _copy_sanitized(source, target, relative)
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
    for relative_path in ("paper/iclr2027/results_macros.tex",):
        source = project_root / relative_path
        target = output_root / relative_path
        _copy_sanitized(source, target)
        copied.append(target)
    for relative_path in (
        "docs/v7-evidence-2026-08-25.md",
        "docs/external-environment-provenance.md",
        "docs/claim_boundary.md",
        "V15_BASELINE_SNAPSHOT.md",
        "V15_NUMBER_AUDIT.md",
        "V15_FIGURE_STATUS.md",
        "V15_REVIEWER_ATTACK_AUDIT.md",
        "V15_CLAIM_AUDIT.md",
        "V15_REFERENCE_AUDIT.md",
        "V15_FINAL_REPORT.md",
        "V15_SCIENTIFIC_SUMMARY.md",
        "V15_TERMINAL_STATE_AUDIT.md",
        "V15_REPRODUCIBILITY_AUDIT.md",
        "V15_CONSTRUCT_VALIDITY.md",
        "snapshot/v15_pre_modern_agent/PROVENANCE.txt",
    ):
        source = project_root / relative_path
        target = output_root / relative_path
        _copy_sanitized(source, target)
        copied.append(target)
    _rewrite_snapshot_manifest(
        output_root / "snapshot" / "manifest.json",
        original_manifest_sha256,
    )

    # The public membership summary is intentionally required in the archive
    # so reviewers can verify plane counts and hashes without receiving the
    # sealed task files.  Fail closed if a future allowlist change drops it.
    public_manifest_target = output_root / PUBLIC_MANIFEST_RELATIVE
    if not public_manifest_target.is_file():
        raise FileNotFoundError(public_manifest_target)

    readme = output_root / "README.md"
    readme.write_text(
        """# IMPROVE-X / PIVOT ICLR 2027 Supplementary Artifact

This archive contains the anonymous source, public configurations, tests, the
controlled ImprovementBench v1/v2 releases, and the hash-indexed paper
snapshot used for the submission PDF. The sealed V15 task manifest and lock
history remain local; only the redacted task-membership summary is included.
From the
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
transforms are under the historical figure/source directories. They do not rerun
science: they read hash-indexed source rows and emit PDF/SVG/PNG figures plus
CSV provenance tables. Rebuild and audit the complete package with:

```bash
.venv/bin/python -m experiments.v15 reports --root .
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
    manifest = {
        "schema_version": "pivot-anonymous-supplement-manifest-2",
        "file_count": len(files) + 1,
        "archive": args.archive.name,
        # The archive digest cannot be embedded in the archive without making
        # the digest self-referential.  Keep the in-archive value explicit and
        # publish the computed digest in the sidecar manifest below.
        "archive_sha256": None,
        "snapshot_manifest_sha256": hashlib.sha256(
            (args.output_root / "snapshot" / "manifest.json").read_bytes()
        ).hexdigest(),
        "allowlist": list(ALLOWLIST)
        + ["paper/snapshot", "paper/tables", "paper/iclr2027/results_macros.tex"],
    }
    # Put a self-describing manifest inside the archive.  The archive digest is
    # intentionally kept in the sidecar below because embedding it would make
    # the digest self-referential.
    manifest_path = args.output_root / "supplement_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = write_deterministic_zip(args.output_root.resolve(), args.archive.resolve())
    sidecar = {**manifest, "archive_sha256": digest, "manifest_inside_archive": True}
    manifest_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(sidecar, sort_keys=True))


if __name__ == "__main__":
    main()
