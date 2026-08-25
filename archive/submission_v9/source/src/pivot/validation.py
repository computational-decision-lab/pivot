from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_run_artifacts(run_dir: Path, required_files: Iterable[str] = ()) -> dict[str, object]:
    """Validate existence and hashes for a run manifest and requested files."""

    directory = Path(run_dir)
    manifest_path = directory / "manifest.json"
    errors: list[str] = []
    manifest: dict[str, object] = {}
    if not manifest_path.exists():
        errors.append("manifest.json is missing")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = manifest.get("files", {})
            if not isinstance(files, dict):
                errors.append("manifest.files is not a mapping")
            else:
                for name, expected in files.items():
                    path = directory / str(name)
                    if not path.exists():
                        errors.append(f"missing manifest file: {name}")
                    elif sha256(path) != str(expected):
                        errors.append(f"hash mismatch: {name}")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"invalid manifest: {error}")
    for name in required_files:
        if not (directory / name).exists():
            errors.append(f"missing required file: {name}")
    return {"valid": not errors, "errors": errors, "manifest": manifest}


def validate_figure_bundle(
    figure_dir: Path,
    stems: Iterable[str],
) -> dict[str, object]:
    directory = Path(figure_dir)
    errors: list[str] = []
    checked: list[str] = []
    for stem in stems:
        png = directory / f"{stem}.png"
        source_csv = directory / f"{stem}.csv"
        unavailable = directory / f"{stem}.unavailable"
        if unavailable.exists():
            checked.append(f"{stem}:unavailable")
            continue
        if not png.exists() or png.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            errors.append(f"{stem}: missing or invalid PNG")
        if not source_csv.exists():
            errors.append(f"{stem}: source CSV is missing")
        else:
            checked.append(stem)
    return {"valid": not errors, "errors": errors, "checked": checked}
