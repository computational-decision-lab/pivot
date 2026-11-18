#!/usr/bin/env python3
"""Create the immutable source/dependency manifest for v3c confirmation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


FROZEN_PATHS = (
    "pyproject.toml",
    "docs/rebuttal/audit/P1_LOOP_V3C_PROTOCOL_2026-11-10.md",
    "docs/rebuttal/audit/run_multiround_loop.py",
    "docs/rebuttal/audit/run_multiround_loop_v3b.py",
    "docs/rebuttal/audit/run_multiround_loop_v3c.py",
    "docs/rebuttal/audit/analyze_multiround_loop_v3c.py",
    "docs/rebuttal/audit/validate_v3b_numerics.py",
    "docs/rebuttal/audit/freeze_multiround_loop_v3c.py",
    "docs/rebuttal/audit_inputs/highway_redesign/source-manifest.json",
    "docs/rebuttal/audit_inputs/highway_redesign/source/experiments/revision/highway_calibration.py",
    "docs/rebuttal/audit_inputs/highway_redesign/source/experiments/revision/highway_b4_redesign.py",
    "src/pivot/environments/performative/world.py",
    "src/pivot/environments/performative/config.py",
    "src/pivot/core/policy.py",
    "src/pivot/core/result.py",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    files = {name: sha256_file(root / name) for name in FROZEN_PATHS}
    distributions = sorted(
        (dist.metadata.get("Name", ""), dist.version)
        for dist in importlib.metadata.distributions()
    )
    body = {
        "manifest_id": "P1-LOOP-V3C-FREEZE-2026-11-10",
        "purpose": "One-shot confirmation source, data, and dependency freeze",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": [{"name": name, "version": version} for name, version in distributions],
        "frozen_files": files,
    }
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    body["manifest_sha256"] = digest
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(args.output.resolve()), "manifest_sha256": digest, "frozen_file_count": len(files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
