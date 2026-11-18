from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "improvementbench-v7"
SPLITS = frozenset({"development", "validation", "test"})
GROUP_KEYS = ("trajectory_id", "environment_id", "operator_id", "opponent_family", "response_regime")


@dataclass(frozen=True)
class ImprovementBenchV7Row:
    transition_id: str
    trajectory_id: str
    round_id: int | str
    operator_id: str
    environment_id: str
    incumbent_policy_id: str
    candidate_policy_id: str
    delta_proxy: float | None
    delta_direct: float | None
    delta_actor: float | None
    delta_strategic: float | None
    update_footprint: float | None
    operator_shift: float | None
    response_strength: float | None
    competition_strength: float | None
    paired_rollout_ids: tuple[str, ...]
    proxy_rank: int | None
    true_rank: int | None
    failure_type: str
    hf_queried: bool
    hf_cost: float
    seed: int | None
    split: str
    opponent_family: str = "none"
    response_regime: str = "default"
    metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: Mapping[str, Any], *, split: str) -> ImprovementBenchV7Row:
        if split not in SPLITS:
            raise ValueError(f"unsupported ImprovementBench V7 split: {split}")
        paired = record.get("paired_rollout_ids", ())
        if not isinstance(paired, Sequence) or isinstance(paired, (str, bytes)):
            raise TypeError("paired_rollout_ids must be a sequence")
        return cls(
            transition_id=str(record["transition_id"]),
            trajectory_id=str(record.get("trajectory_id", record.get("trajectory_id", "unknown"))),
            round_id=record.get("round_id", 0),
            operator_id=str(record.get("operator_id", record.get("improvement_operator", "unknown"))),
            environment_id=str(record.get("environment_id", record.get("config_id", "unknown"))),
            incumbent_policy_id=str(record.get("incumbent_policy_id", "unknown")),
            candidate_policy_id=str(record.get("candidate_policy_id", "unknown")),
            delta_proxy=_optional_float(record.get("delta_proxy")),
            delta_direct=_optional_float(record.get("delta_direct", record.get("delta_true"))),
            delta_actor=_optional_float(record.get("delta_actor")),
            delta_strategic=_optional_float(record.get("delta_strategic")),
            update_footprint=_optional_float(record.get("update_footprint")),
            operator_shift=_optional_float(record.get("operator_shift")),
            response_strength=_optional_float(record.get("response_strength")),
            competition_strength=_optional_float(record.get("competition_strength")),
            paired_rollout_ids=tuple(str(value) for value in paired),
            proxy_rank=_optional_int(record.get("proxy_rank")),
            true_rank=_optional_int(record.get("true_rank")),
            failure_type=str(record.get("failure_type", "unknown")),
            hf_queried=bool(record.get("hf_queried", False)),
            hf_cost=float(record.get("hf_cost", record.get("hf_query_cost", 0.0)) or 0.0),
            seed=_optional_int(record.get("seed")),
            split=split,
            opponent_family=str(record.get("opponent_family", "none")),
            response_regime=str(record.get("response_regime", _response_regime(record.get("response_strength")))),
            metadata={"source": str(record.get("source", "v7"))},
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "transition_id": self.transition_id,
            "trajectory_id": self.trajectory_id,
            "round_id": self.round_id,
            "operator_id": self.operator_id,
            "environment_id": self.environment_id,
            "incumbent_policy_id": self.incumbent_policy_id,
            "candidate_policy_id": self.candidate_policy_id,
            "delta_proxy": self.delta_proxy,
            "delta_direct": self.delta_direct,
            "delta_actor": self.delta_actor,
            "delta_strategic": self.delta_strategic,
            "update_footprint": self.update_footprint,
            "operator_shift": self.operator_shift,
            "response_strength": self.response_strength,
            "competition_strength": self.competition_strength,
            "paired_rollout_ids": list(self.paired_rollout_ids),
            "proxy_rank": self.proxy_rank,
            "true_rank": self.true_rank,
            "failure_type": self.failure_type,
            "hf_queried": self.hf_queried,
            "hf_cost": self.hf_cost,
            "seed": self.seed,
            "split": self.split,
            "opponent_family": self.opponent_family,
            "response_regime": self.response_regime,
            "metadata": dict(self.metadata),
        }


@dataclass
class ImprovementBenchV7Dataset:
    rows: list[ImprovementBenchV7Row] = field(default_factory=list)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        for group_key in GROUP_KEYS:
            validate_group_splits(self.rows, group_key=group_key)
        ids = [row.transition_id for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate transition_id in ImprovementBench V7")

    def write(self, directory: Path) -> dict[str, object]:
        self.validate()
        directory.mkdir(parents=True, exist_ok=True)
        transitions = directory / "transitions.jsonl"
        transitions.write_text(
            "".join(json.dumps(row.to_record(), sort_keys=True, separators=(",", ":")) + "\n" for row in self.rows),
            encoding="utf-8",
        )
        metadata_path = directory / "metadata.json"
        metadata_path.write_text(
            json.dumps({"schema_version": SCHEMA_VERSION, "row_count": len(self.rows), **dict(self.metadata)}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        readme_path = directory / "README.md"
        readme_path.write_text(
            "# ImprovementBench V7\n\n"
            "Transition-level records from the V7 development pipeline. Splits "
            "are assigned by connected components of trajectory, environment, "
            "operator, opponent-family, and response-regime groups; the current "
            "small development pool is intentionally reported as development-only "
            "when those groups are fully connected. No row is reused across a "
            "different split.\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "row_count": len(self.rows),
            "files": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (transitions, metadata_path, readme_path)
            },
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest


def assign_group_split(group: str, *, seed: int = 20260825) -> str:
    if not group:
        raise ValueError("split group must not be empty")
    value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) % 100
    if value < 70:
        return "development"
    if value < 85:
        return "validation"
    return "test"


def assign_leakage_safe_splits(
    records: Sequence[Mapping[str, Any]], *, seed: int = 20260825
) -> list[str]:
    """Assign splits by connected components of every registered leakage group."""

    if not records:
        return []
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    seen: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        for key in GROUP_KEYS:
            value = str(record.get(key, record.get("trajectory_id", "unknown")))
            token = (key, value)
            if token in seen:
                union(index, seen[token])
            else:
                seen[token] = index
    components: dict[int, list[int]] = {}
    for index in range(len(records)):
        components.setdefault(find(index), []).append(index)
    assignments: dict[int, str] = {}
    for root, members in components.items():
        tokens = sorted(
            f"{key}={records[members[0]].get(key, records[members[0]].get('trajectory_id', 'unknown'))}"
            for key in GROUP_KEYS
        )
        assignments[root] = assign_group_split("|".join(tokens), seed=seed)
    return [assignments[find(index)] for index in range(len(records))]


def validate_group_splits(rows: Iterable[ImprovementBenchV7Row], *, group_key: str = "trajectory_id") -> None:
    if group_key not in GROUP_KEYS:
        raise ValueError(f"unsupported leakage group key: {group_key}")
    groups: dict[str, str] = {}
    for row in rows:
        if row.split not in SPLITS:
            raise ValueError(f"unsupported split: {row.split}")
        group = str(getattr(row, group_key))
        previous = groups.setdefault(group, row.split)
        if previous != row.split:
            raise ValueError(f"group leakage detected for {group_key}: {group}")


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise TypeError("benchmark float fields must be numeric or None")
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, (int, float, str)):
        raise TypeError("benchmark integer fields must be scalar or None")
    return int(value)


def _response_regime(value: object) -> str:
    if value is None:
        return "default"
    if not isinstance(value, (int, float)):
        raise TypeError("response strength must be numeric or None")
    numeric = float(value)
    if numeric <= 0.33:
        return "low"
    if numeric <= 0.66:
        return "medium"
    return "high"
