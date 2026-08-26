"""Build an aggregate manifest for all V9 run and publication artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pivot.v9.artifacts import build_manifest, sha256, write_json


def build(root: Path) -> dict[str, Any]:
    result_root = root / "results/v9"
    runs: list[dict[str, Any]] = []
    for directory in sorted(path for path in result_root.glob("*") if path.is_dir()):
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        runs.append(
            {
                "run": directory.name,
                "experiment_id": payload.get("experiment_id"),
                "status": payload.get("status"),
                "manifest_sha256": sha256(manifest_path),
                "file_count": len(payload.get("files", {})),
            }
        )
    publication_roots = (root / "figures/v9", root / "tables/v9", root / "artifacts/v9")
    publication: dict[str, dict[str, Any]] = {}
    for directory in publication_roots:
        if directory.is_dir():
            # The per-directory manifest is deliberately generated from the
            # current files, so its hash is an auditable release pointer.
            manifest = build_manifest(directory, experiment_id=directory.name, status="PUBLISHED_ARTIFACTS")
            publication[str(directory.relative_to(root))] = {
                "manifest_sha256": sha256(directory / "manifest.json"),
                "file_count": len(manifest["files"]),
            }
    aggregate = {"schema_version": "pivot-v9-aggregate-manifest-v1", "runs": runs, "publication": publication}
    write_json(root / "artifacts/v9/aggregate_manifest.json", aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aggregate V9 manifest")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    aggregate = build(args.root.resolve())
    print(json.dumps({"runs": len(aggregate["runs"]), "publication_roots": len(aggregate["publication"])}, sort_keys=True))


if __name__ == "__main__":
    main()
