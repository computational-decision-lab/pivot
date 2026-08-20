from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from improve_x.failures.taxonomy import classify_failure

SCHEMA_VERSION = "improvementbench-v1"
WORLD_LEVELS = frozenset({"observer", "actor", "strategic"})


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise TypeError("benchmark numeric fields must be numeric or None")
    return float(value)


@dataclass(frozen=True)
class ImprovementBenchRow:
    transition_id: str
    round_id: int | str
    incumbent_policy: Mapping[str, float]
    candidate_policy: Mapping[str, float]
    improvement_operator: str
    world_level: str
    proxy_delta: float | None
    deployment_delta: float | None
    actor_delta: float | None
    strategic_delta: float | None
    proxy_value: float | None
    deployment_value: float | None
    update_footprint: float | None
    environment_response: float | None
    failure_type: str
    seed: int | None
    candidate_index: int
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.world_level not in WORLD_LEVELS:
            raise ValueError(f"unsupported world_level: {self.world_level}")
        if self.candidate_index < 0:
            raise ValueError("candidate_index must be non-negative")
        object.__setattr__(self, "incumbent_policy", dict(self.incumbent_policy))
        object.__setattr__(self, "candidate_policy", dict(self.candidate_policy))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def delta_proxy(self) -> float | None:
        return self.proxy_delta

    @property
    def delta_actor(self) -> float | None:
        return self.actor_delta

    @property
    def delta_strategic(self) -> float | None:
        return self.strategic_delta

    @property
    def delta_true(self) -> float | None:
        return self.deployment_delta

    @classmethod
    def from_transition(cls, record: Mapping[str, Any], world_level: str) -> ImprovementBenchRow:
        if not world_level:
            raise ValueError("world_level must not be empty")
        proxy = _number(record.get("delta_proxy"))
        actor = _number(record.get("delta_actor"))
        strategic = _number(record.get("delta_strategic"))
        true_delta = _number(record.get("delta_true"))
        deployment = true_delta if true_delta is not None else actor
        failure = classify_failure(
            delta_proxy=proxy,
            delta_actor=actor if actor is not None else deployment,
            delta_strategic=strategic,
        )
        return cls(
            transition_id=str(record["transition_id"]),
            round_id=record["round_id"],
            incumbent_policy={str(k): float(v) for k, v in dict(record.get("incumbent_parameters", {})).items()},
            candidate_policy={str(k): float(v) for k, v in dict(record.get("candidate_parameters", {})).items()},
            improvement_operator=str(record["improvement_operator"]),
            world_level=world_level,
            proxy_delta=proxy,
            deployment_delta=deployment,
            actor_delta=actor,
            strategic_delta=strategic,
            proxy_value=_number(record.get("proxy_candidate_value")),
            deployment_value=_number(record.get("true_candidate_value")),
            update_footprint=_number(record.get("update_footprint")),
            environment_response=_number(record.get("response_strength")),
            failure_type=str(failure.value),
            seed=None if record.get("seed") is None else int(record["seed"]),
            candidate_index=int(record.get("candidate_index", 0)),
            metadata={
                "config_id": record.get("config_id"),
                "competition_strength": record.get("competition_strength"),
                "opponent_context": record.get("opponent_context", {}),
            },
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "transition_id": self.transition_id,
            "round_id": self.round_id,
            "incumbent_policy": dict(self.incumbent_policy),
            "candidate_policy": dict(self.candidate_policy),
            "improvement_operator": self.improvement_operator,
            "world_level": self.world_level,
            "proxy_delta": self.proxy_delta,
            "deployment_delta": self.deployment_delta,
            "actor_delta": self.actor_delta,
            "strategic_delta": self.strategic_delta,
            "proxy_value": self.proxy_value,
            "deployment_value": self.deployment_value,
            "update_footprint": self.update_footprint,
            "environment_response": self.environment_response,
            "failure_type": self.failure_type,
            "seed": self.seed,
            "candidate_index": self.candidate_index,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ImprovementBenchRow:
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported ImprovementBench schema")
        return cls(
            transition_id=str(record["transition_id"]),
            round_id=record["round_id"],
            incumbent_policy={str(k): float(v) for k, v in dict(record["incumbent_policy"]).items()},
            candidate_policy={str(k): float(v) for k, v in dict(record["candidate_policy"]).items()},
            improvement_operator=str(record["improvement_operator"]),
            world_level=str(record["world_level"]),
            proxy_delta=_number(record.get("proxy_delta")),
            deployment_delta=_number(record.get("deployment_delta")),
            actor_delta=_number(record.get("actor_delta")),
            strategic_delta=_number(record.get("strategic_delta")),
            proxy_value=_number(record.get("proxy_value")),
            deployment_value=_number(record.get("deployment_value")),
            update_footprint=_number(record.get("update_footprint")),
            environment_response=_number(record.get("environment_response")),
            failure_type=str(record["failure_type"]),
            seed=None if record.get("seed") is None else int(record["seed"]),
            candidate_index=int(record["candidate_index"]),
            metadata=dict(record.get("metadata", {})),
        )


@dataclass
class ImprovementBenchDataset:
    rows: list[ImprovementBenchRow] = field(default_factory=list)
    metadata: Mapping[str, object] = field(default_factory=dict)
    directory: Path | None = field(default=None, repr=False, compare=False)

    def append(self, row: ImprovementBenchRow) -> None:
        if any(existing.transition_id == row.transition_id for existing in self.rows):
            raise ValueError(f"duplicate transition_id: {row.transition_id}")
        self.rows.append(row)

    def write(self, directory: Path, *, created_at: str | None = None) -> dict[str, object]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        transitions = directory / "transitions.jsonl"
        lines = [json.dumps(row.to_record(), sort_keys=True, separators=(",", ":")) for row in self.rows]
        transitions.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "row_count": len(self.rows),
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
            "source_commit": _git_commit(),
            **dict(self.metadata),
        }
        metadata_path = directory / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "row_count": len(self.rows),
            "files": {transitions.name: _hash(transitions), metadata_path.name: _hash(metadata_path)},
        }
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.directory = directory
        return manifest

    @classmethod
    def read(cls, directory: Path) -> ImprovementBenchDataset:
        directory = Path(directory)
        try:
            raw_lines = (directory / "transitions.jsonl").read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ValueError(f"cannot read ImprovementBench JSONL: {error}") from error
        rows: list[ImprovementBenchRow] = []
        for line in raw_lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in ImprovementBench: {error}") from error
            rows.append(ImprovementBenchRow.from_record(payload))
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        return cls(rows, metadata, directory)

    def validate(self, directory: Path | None = None) -> dict[str, object]:
        directory = Path(directory or self.directory or ".")
        errors: list[str] = []
        manifest_path = directory / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != SCHEMA_VERSION:
                errors.append("manifest schema_version mismatch")
            if manifest.get("row_count") != len(self.rows):
                errors.append("manifest row_count mismatch")
            for name, digest in manifest["files"].items():
                path = directory / name
                if not path.is_file() or _hash(path) != digest:
                    errors.append(f"hash mismatch: {name}")
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            errors.append(f"invalid manifest: {error}")
        metadata_path = directory / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("schema_version") != SCHEMA_VERSION:
                errors.append("metadata schema_version mismatch")
            if metadata.get("row_count") != len(self.rows):
                errors.append("metadata row_count mismatch")
        except (OSError, TypeError, json.JSONDecodeError) as error:
            errors.append(f"invalid metadata: {error}")
        return {"valid": not errors, "errors": errors}


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
