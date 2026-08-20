#!/usr/bin/env python3
"""Freeze verified experiment outputs for the paper build.

The paper must consume a local, hash-indexed snapshot rather than an implicit
temporary clean-room directory.  This script copies only the declared output
surface; raw market archives and credentials are never included.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

FIGURE_FILES = tuple(
    f"{stem}.{suffix}"
    for stem in (
        "fig1_when_better_gets_worse",
        "fig2_reversal_phase_diagram",
        "fig3_optimizing_wrong_world",
        "fig4_policy_vs_improvement_fidelity",
        "fig5_pivot_budget_frontier",
        "fig6_observer_actor_strategic",
        "fig7_strategic_reversal",
    )
    for suffix in ("png", "csv")
)

SUMMARY_FILES = (
    "p2-summary.json",
    "e4-summary.json",
    "e5-summary.json",
    "e6-summary.json",
    "f-summary.json",
    "e9-summary.json",
    "e3/overoptimization.json",
    "figures/figure_validation.json",
)


def freeze_snapshot(
    source_root: Path,
    ablation_root: Path,
    public_root: Path,
    output: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Copy declared artifacts and write a deterministic manifest."""

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty snapshot: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path(source_root).resolve()
    ablation_root = Path(ablation_root).resolve()
    public_root = Path(public_root).resolve()
    files: dict[str, dict[str, Any]] = {}
    for relative in FIGURE_FILES:
        source = source_root / "figures" / relative
        _copy_record(source, output / "figures" / relative, relative, "controlled", files)
    for relative in SUMMARY_FILES:
        source = source_root / relative
        target_relative = f"summaries/{relative.replace('/', '__')}"
        _copy_record(source, output / target_relative, relative, "controlled", files)
    _copy_record(
        ablation_root / "ablation-aggregate.json",
        output / "summaries/ablation-aggregate.json",
        "ablation-aggregate.json",
        "ablations",
        files,
    )
    _copy_record(
        public_root / "summary.json",
        output / "summaries/public-expansion-summary.json",
        "summary.json",
        "public-expansion",
        files,
    )
    _copy_record(
        public_root / "provenance.json",
        output / "summaries/public-expansion-provenance.json",
        "provenance.json",
        "public-expansion",
        files,
    )
    validation = json.loads(
        (output / "summaries/figures__figure_validation.json").read_text(encoding="utf-8")
    )
    if validation.get("valid") is not True or len(validation.get("checked", [])) != 7:
        raise ValueError("paper snapshot requires seven validated figures")
    manifest = {
        "snapshot_version": "pivot-paper-snapshot-v1",
        "source_roots": {
            "controlled": str(source_root),
            "ablations": str(ablation_root),
            "public_expansion": str(public_root),
        },
        "source_commits": {
            "project": _git_commit(project_root or Path.cwd()),
            "controlled": _read_commit(source_root),
            "ablations": _read_commit(ablation_root),
            "public_expansion": _read_commit(public_root),
        },
        "files": files,
        "claim_boundary": {
            "public_causal_impact_identified": False,
            "public_live_orders": False,
            "e3_overoptimization_claim": False,
            "llm_evoquant_m3_used": False,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _copy_record(
    source: Path,
    target: Path,
    source_relative: str,
    source_kind: str,
    files: dict[str, dict[str, Any]],
) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"required paper artifact is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    files[str(target.relative_to(target.parents[1]))] = {
        "source_kind": source_kind,
        "source_relative": source_relative,
        "sha256": _sha256(target),
        "size_bytes": target.stat().st_size,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_commit(root: Path) -> str | None:
    manifest = root / "provenance.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("git_commit"):
            return str(payload["git_commit"])
    return None


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze PIVOT paper artifacts")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    manifest = freeze_snapshot(
        args.source_root,
        args.ablation_root,
        args.public_root,
        args.output,
        project_root=args.project_root,
    )
    print(json.dumps({"files": len(manifest["files"]), "snapshot": str(args.output)}))


if __name__ == "__main__":
    main()
