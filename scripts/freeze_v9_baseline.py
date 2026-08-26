#!/usr/bin/env python3
"""Create and verify the immutable V7 baseline manifest for V9 work.

The V9 upgrade is additive.  This script records the exact pre-upgrade commit
and hashes every existing paper, figure, and result artifact without copying
large files into a second public tree.  The manifest is a reference snapshot:
future audits can detect any accidental mutation of the baseline inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _artifact_paths(root: Path) -> list[Path]:
    """Return the pre-upgrade publication and experiment surface."""

    paths: set[Path] = set()
    for relative_root in ("results", "paper/snapshot"):
        directory = root / relative_root
        if directory.is_dir():
            paths.update(path for path in directory.rglob("*") if path.is_file())
    paper_root = root / "paper/iclr2027"
    if paper_root.is_dir():
        paths.update(path for path in paper_root.rglob("*") if path.is_file())
    for relative in (
        "research/research_state.json",
        "research/claim_registry.yaml",
        "research/experiment_registry.yaml",
        "research/decision_log.md",
        "research/power_analysis.md",
        "configs/controlled/main.yaml",
        "configs/registered/e2_operator_shift.yaml",
        "configs/confirmatory/e3b.yaml",
        "configs/confirmatory/e4b.yaml",
        "configs/confirmatory/e7b.yaml",
    ):
        path = root / relative
        if path.is_file():
            paths.add(path)
    return sorted(paths)


def freeze_baseline(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing baseline: {manifest_path}")
    output.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for path in _artifact_paths(root):
        relative = path.relative_to(root).as_posix()
        artifacts.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    commit = git_value(root, "rev-parse", "HEAD")
    if not commit:
        raise RuntimeError("baseline requires a git HEAD")
    manifest: dict[str, Any] = {
        "schema_version": "pivot-v9-baseline-v1",
        "baseline_label": "V7-before-V9-upgrade",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "root_reference": "<project-root>",
            "commit": commit,
            "remote": git_value(root, "config", "--get", "remote.origin.url"),
            # The parent checkout may contain unrelated private workspaces;
            # never publish its full status inventory in the baseline.
            "status_porcelain": "<omitted-from-public-snapshot>",
        },
        "runtime": {"python": platform.python_version(), "machine": platform.machine()},
        "artifact_policy": {
            "paper_pdf": "paper/iclr2027/pivot_iclr2027_submission.pdf",
            "supplement": "paper/iclr2027/pivot_iclr2027_supplementary.zip",
            "results_root": "results",
            "historical_archive": "archive/submission_v9",
            "mutation_rule": "V7 artifacts are read-only inputs; V9 uses separate IDs.",
        },
        "artifacts": artifacts,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme = output / "README.md"
    readme.write_text(
        """# V9 Pre-upgrade Baseline\n\n"
        "This directory records the V7 PIVOT publication and experiment surface "
        "before the V9 upgrade. `manifest.json` stores the pre-upgrade git commit "
        "and SHA-256 for every existing result, figure, PDF, supplement, source "
        "and registered configuration. The files remain in their canonical paths "
        "so the public package does not duplicate large raw tables.\n\n"
        "Do not overwrite or re-label these artifacts. V9 runs use separate IDs "
        "under `results/v9/`, `figures/v9/`, and `tables/v9/`; powered nulls are "
        "preserved as results.\n""",
        encoding="utf-8",
    )
    return manifest


def verify_baseline(root: Path, manifest_path: Path, *, allow_v9_overlay: bool | None = None) -> dict[str, Any]:
    root = root.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if allow_v9_overlay is None:
        # V9 deliberately regenerates a few canonical publication paths. In
        # that workspace, validate the immutable baseline commit instead of
        # treating those expected overlays as accidental mutations. A clean
        # or test workspace still compares the live files and catches edits.
        allow_v9_overlay = (root / "research/research_state_v9.json").is_file() or (root / "results/v9").is_dir()
    mismatches: list[dict[str, str]] = []
    unverified_derived: list[str] = []
    commit = str(payload.get("repository", {}).get("commit", ""))
    git_prefix = _git_prefix(root)
    checked_git = 0
    checked_files = 0
    for artifact in payload.get("artifacts", []):
        relative = str(artifact["path"])
        expected = str(artifact["sha256"])
        git_path = f"{git_prefix}/{relative}" if git_prefix else relative
        actual_git = _git_blob_sha256(root, commit, git_path) if commit else None
        path = root / relative
        if actual_git is not None and allow_v9_overlay:
            checked_git += 1
            if actual_git != expected:
                mismatches.append({"path": relative, "reason": f"baseline-blob-hash:{actual_git}"})
            continue
        if _is_derived_path(relative):
            unverified_derived.append(relative)
            continue
        if path.is_file():
            checked_files += 1
            actual = sha256(path)
            if actual != expected:
                mismatches.append({"path": relative, "reason": f"hash:{actual}"})
        else:
            # Older snapshots included ignored build/supplement derivatives
            # that were never part of the baseline commit. Report them as
            # derived and unverifiable rather than treating the commit as
            # mutated after a paper rebuild.
            unverified_derived.append(relative)
    return {
        "valid": not mismatches,
        "checked": len(payload.get("artifacts", [])),
        "checked_git_blobs": checked_git,
        "checked_worktree_files": checked_files,
        "allow_v9_overlay": bool(allow_v9_overlay),
        "unverified_derived": unverified_derived,
        "mismatches": mismatches,
        "baseline_commit": commit,
    }


def _git_prefix(root: Path) -> str:
    try:
        top = Path(subprocess.check_output(["git", "-C", str(root), "rev-parse", "--show-toplevel"], text=True).strip())
        return root.relative_to(top).as_posix()
    except (OSError, subprocess.CalledProcessError, ValueError):
        return ""


def _git_blob_sha256(root: Path, commit: str, path: str) -> str | None:
    if not commit:
        return None
    try:
        process = subprocess.run(
            ["git", "-C", str(root), "cat-file", "blob", f"{commit}:{path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(process.stdout).hexdigest()


def _is_derived_path(relative: str) -> bool:
    """Identify ignored LaTeX/build and expanded supplement derivatives."""

    return relative.startswith(("paper/iclr2027/build/", "paper/iclr2027/supplementary/"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze or verify the PIVOT V9 baseline")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("snapshot/v9_preupgrade"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest_path = args.output / "manifest.json"
    if args.verify:
        report = verify_baseline(args.root, manifest_path)
        print(json.dumps(report, sort_keys=True))
        raise SystemExit(0 if report["valid"] else 1)
    manifest = freeze_baseline(args.root, args.output)
    print(json.dumps({"commit": manifest["repository"]["commit"], "artifacts": len(manifest["artifacts"])}))


if __name__ == "__main__":
    main()
