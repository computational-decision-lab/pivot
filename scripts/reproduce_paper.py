#!/usr/bin/env python3
"""Rebuild and audit the complete local PIVOT paper package."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.validation import validate_run_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce and verify the PIVOT paper package")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checks: dict[str, Any] = {}
    theory = validate_run_artifacts(root / "results/theory/e10-theory-empirical")
    if not theory["valid"]:
        raise RuntimeError(f"V6 theory artifact invalid: {theory['errors']}")
    checks["v6_theory_artifact"] = True
    benchmark = root / "benchmarks/improvementbench/v7"
    checks["improvementbench_v7_manifest"] = _verify_manifest(benchmark)
    if not checks["improvementbench_v7_manifest"]:
        raise RuntimeError("ImprovementBench V7 manifest mismatch")
    if not args.check_only:
        subprocess.run(["bash", "paper/iclr2027/build.sh"], cwd=root, check=True)
        subprocess.run(
            [
                sys.executable,
                "scripts/verify_paper.py",
                "--pdf",
                "paper/iclr2027/build/main.pdf",
                "--source",
                "paper/iclr2027/main.tex",
                "--output",
                "paper/iclr2027/verification.json",
                "--preview",
                "paper/iclr2027/preview.png",
                "--max-main-pages",
                "9",
            ],
            cwd=root,
            check=True,
        )
        checks["paper_verification"] = True
    else:
        checks["paper_verification"] = json.loads(
            (root / "paper/iclr2027/verification.json").read_text(encoding="utf-8")
        ).get("valid", False)
    report = {"checks": checks, "valid": all(bool(value) for value in checks.values())}
    print(json.dumps(report, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


def _verify_manifest(directory: Path) -> bool:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return False
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = payload.get("files", {})
    return bool(files) and all(
        (directory / name).is_file()
        and hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
        for name, digest in files.items()
    )


if __name__ == "__main__":
    main()
