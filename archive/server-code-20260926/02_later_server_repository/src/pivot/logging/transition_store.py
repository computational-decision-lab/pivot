from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Manifest:
    row_count: int
    storage_format: str
    files: Mapping[str, str]
    required_columns: tuple[str, ...]
    git_commit: str | None
    created_at: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TransitionStore:
    def __init__(self, run_dir: Path, required_columns: Iterable[str]) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "transitions.jsonl"
        if self.path.exists():
            raise FileExistsError(f"refusing to overwrite existing transition file: {self.path}")
        self.required_columns = tuple(required_columns)
        if len(set(self.required_columns)) != len(self.required_columns):
            raise ValueError("required_columns must not contain duplicates")
        self.row_count = 0
        self._transition_ids: set[str] = set()

    def append(self, row: Mapping[str, Any]) -> None:
        missing = [column for column in self.required_columns if column not in row]
        if missing:
            raise ValueError(f"transition row missing required columns: {missing}")
        transition_id = row.get("transition_id")
        if transition_id is None or not str(transition_id):
            raise ValueError("transition_id must be a non-empty value")
        transition_key = str(transition_id)
        if transition_key in self._transition_ids:
            raise ValueError(f"duplicate transition_id: {transition_key}")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":"), default=str))
            handle.write("\n")
        self.row_count += 1
        self._transition_ids.add(transition_key)

    def finalize(self) -> Manifest:
        storage_format = "jsonl"
        files = {self.path.name: _sha256(self.path)}
        parquet_path = self.run_dir / "transitions.parquet"
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            rows = [
                _parquet_safe_row(json.loads(line))
                for line in self.path.read_text(encoding="utf-8").splitlines()
            ]
            if rows:
                pq.write_table(pa.Table.from_pylist(rows), parquet_path)
                files[parquet_path.name] = _sha256(parquet_path)
                storage_format = "parquet+jsonl"
        except ImportError:
            pass
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        manifest = Manifest(
            row_count=self.row_count,
            storage_format=storage_format,
            files=files,
            required_columns=self.required_columns,
            git_commit=commit,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest.__dict__, sort_keys=True, indent=2), encoding="utf-8"
        )
        schema_path = self.run_dir / "schema.json"
        schema_path.write_text(
            json.dumps(
                {
                    "required_columns": list(self.required_columns),
                    "row_count": self.row_count,
                    "storage_format": storage_format,
                },
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        # The schema is part of the integrity manifest, so downstream readers
        # can reject a run whose contract was changed after finalization.
        manifest_with_schema = Manifest(
            row_count=manifest.row_count,
            storage_format=manifest.storage_format,
            files={**manifest.files, schema_path.name: _sha256(schema_path)},
            required_columns=manifest.required_columns,
            git_commit=manifest.git_commit,
            created_at=manifest.created_at,
        )
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest_with_schema.__dict__, sort_keys=True, indent=2), encoding="utf-8"
        )
        return manifest_with_schema

    @staticmethod
    def validate_manifest(run_dir: Path) -> bool:
        directory = Path(run_dir)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for name, expected in manifest["files"].items():
                path = directory / name
                if not path.exists() or _sha256(path) != expected:
                    return False
            return True
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return False


def _parquet_safe_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize open-ended JSON maps before Arrow schema inference.

    JSONL remains the lossless interchange format.  Parquet stores the same
    maps as canonical JSON strings because an empty struct has no Arrow child
    fields and cannot be written portably across pyarrow versions.
    """

    normalized: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, Mapping):
            normalized[key] = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
        else:
            normalized[key] = value
    return normalized
