"""Immutable protocol objects and canonical artifact writers.

The objects in this module are deliberately independent of a particular agent
framework.  They make the directed replacement transition explicit and give
every candidate a content-derived identity before any hidden-world evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

TERMINAL_STATES = frozenset(
    {
        "IMPLEMENTATION_FAILURE",
        "DESIGN_INVALID",
        "UNDERPOWERED",
        "HYPOTHESIS_SUPPORTED",
        "HYPOTHESIS_NOT_SUPPORTED",
    }
)


def validate_terminal_state(value: str) -> bool:
    """Return whether *value* is one of the closed scientific terminal states."""

    return value in TERMINAL_STATES


def _plain(value: Any) -> Any:
    """Convert nested mappings/sequences to deterministic JSON-compatible data."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    """Serialize a value with stable ordering and separators."""

    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(value: Any, *, length: int = 64) -> str:
    """Return a SHA-256 hash of canonical JSON content."""

    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest[:length]


def file_hash(path: Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _plain(item) for key, item in (value or {}).items()})


@dataclass(frozen=True)
class AgentPolicy:
    """A serializable coding-agent scaffold policy.

    The policy is broader than a prompt: loop, tool, search, test, and context
    controls are all part of the content hash.  This prevents a candidate from
    being identified only by its visible text while silently changing runtime
    behavior.
    """

    system_prompt: str
    agent_loop_config: Mapping[str, Any] = field(default_factory=dict)
    tool_policy: Mapping[str, Any] = field(default_factory=dict)
    search_policy: Mapping[str, Any] = field(default_factory=dict)
    test_policy: Mapping[str, Any] = field(default_factory=dict)
    context_policy: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)
    policy_hash: str = field(init=False)

    def __post_init__(self) -> None:
        prompt = str(self.system_prompt).strip()
        if not prompt:
            raise ValueError("system_prompt must not be empty")
        object.__setattr__(self, "system_prompt", prompt)
        for name in (
            "agent_loop_config",
            "tool_policy",
            "search_policy",
            "test_policy",
            "context_policy",
        ):
            object.__setattr__(self, name, _mapping(getattr(self, name)))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType({str(key): str(value) for key, value in self.metadata.items()}),
        )
        payload = self.to_record(include_hash=False)
        object.__setattr__(self, "policy_hash", content_hash(payload))

    @classmethod
    def minimal(cls) -> AgentPolicy:
        """Create the frozen local-reference starting scaffold."""

        return cls(
            system_prompt="Inspect the repository, make the smallest correct edit, and run tests.",
            agent_loop_config={"max_steps": 8, "stop_on_failure": True},
            tool_policy={"shell": True, "read": True, "write": True},
            search_policy={"depth": 2, "max_files": 12},
            test_policy={"run_tests": True, "repair": False},
            context_policy={"max_tokens": 2048, "summarize": True},
            metadata={"edit_type": "baseline", "scaffold": "local-reference"},
        )

    def with_updates(self, **updates: Any) -> AgentPolicy:
        """Return a new policy with explicitly named fields replaced."""

        fields: dict[str, Any] = {
            "system_prompt": self.system_prompt,
            "agent_loop_config": dict(self.agent_loop_config),
            "tool_policy": dict(self.tool_policy),
            "search_policy": dict(self.search_policy),
            "test_policy": dict(self.test_policy),
            "context_policy": dict(self.context_policy),
            "metadata": dict(self.metadata),
        }
        for key, value in updates.items():
            if key not in fields:
                raise TypeError(f"unknown policy field: {key}")
            if key == "metadata":
                merged = dict(fields["metadata"])
                merged.update({str(k): str(v) for k, v in value.items()})
                fields[key] = merged
            else:
                fields[key] = value
        return AgentPolicy(**fields)

    def to_record(self, *, include_hash: bool = True) -> dict[str, Any]:
        record: dict[str, Any] = {
            "system_prompt": self.system_prompt,
            "agent_loop_config": dict(self.agent_loop_config),
            "tool_policy": dict(self.tool_policy),
            "search_policy": dict(self.search_policy),
            "test_policy": dict(self.test_policy),
            "context_policy": dict(self.context_policy),
            "metadata": dict(self.metadata),
        }
        if include_hash:
            record["policy_hash"] = self.policy_hash
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> AgentPolicy:
        policy = cls(
            system_prompt=str(record.get("system_prompt", "")),
            agent_loop_config=record.get("agent_loop_config", {}),
            tool_policy=record.get("tool_policy", {}),
            search_policy=record.get("search_policy", {}),
            test_policy=record.get("test_policy", {}),
            context_policy=record.get("context_policy", {}),
            metadata=record.get("metadata", {}),
        )
        expected = record.get("policy_hash")
        if expected is not None and str(expected) != policy.policy_hash:
            raise ValueError("policy_hash does not match policy content")
        return policy

    def diff(self, other: AgentPolicy) -> dict[str, float]:
        """Compute pre-deployment static footprint features."""

        def numeric(mapping: Mapping[str, Any], key: str) -> float:
            value = mapping.get(key, 0.0)
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        def changed(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
            return float(canonical_json(left) != canonical_json(right))

        left_tokens = self.system_prompt.split()
        right_tokens = other.system_prompt.split()
        max_tokens = max(len(left_tokens), len(right_tokens), 1)
        common = sum(a == b for a, b in zip(left_tokens, right_tokens))
        return {
            "prompt_token_delta": float(len(right_tokens) - len(left_tokens)),
            "prompt_semantic_distance": 1.0 - (common / max_tokens),
            "skill_diff_size": float(len(set(self.metadata.items()) ^ set(other.metadata.items()))),
            "skills_added": float(len(set(other.metadata) - set(self.metadata))),
            "skills_removed": float(len(set(self.metadata) - set(other.metadata))),
            "tool_schema_change": changed(self.tool_policy, other.tool_policy),
            "tool_count_delta": numeric(other.tool_policy, "tool_count")
            - numeric(self.tool_policy, "tool_count"),
            "loop_parameter_delta": numeric(other.agent_loop_config, "max_steps")
            - numeric(self.agent_loop_config, "max_steps"),
            "context_policy_change": changed(self.context_policy, other.context_policy),
            "test_policy_change": changed(self.test_policy, other.test_policy),
            "search_policy_change": numeric(other.search_policy, "depth")
            - numeric(self.search_policy, "depth"),
        }


@dataclass(frozen=True)
class TransitionRecord:
    """Canonical directed candidate transition with no hidden outcomes in its footprint."""

    run_id: str
    scaffold: str
    operator: str
    task_family: str
    round_index: int
    candidate_index: int
    incumbent: AgentPolicy
    candidate: AgentPolicy
    delta_proxy: float
    delta_actor: float | None = None
    delta_strategic: float | None = None
    footprint: Mapping[str, float] = field(default_factory=dict)
    resource_metrics: Mapping[str, Any] = field(default_factory=dict)
    seed: int | None = None
    paired_seed_ids: tuple[int, ...] = ()
    source_digest: str | None = None
    config_hash: str | None = None
    hf_queried: bool = False
    hf_query_reason: str | None = None
    terminal_state: str | None = None
    # Optional level scores make the policy-level versus transition-level
    # estimands identifiable when an external runner records both policies.
    # They are deliberately excluded from the content-derived transition id.
    proxy_incumbent_score: float | None = None
    proxy_candidate_score: float | None = None
    actor_incumbent_score: float | None = None
    actor_candidate_score: float | None = None
    transition_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.run_id or not self.scaffold or not self.operator or not self.task_family:
            raise ValueError("run_id, scaffold, operator, and task_family are required")
        if self.round_index < 0 or self.candidate_index < 0:
            raise ValueError("round_index and candidate_index must be non-negative")
        if self.incumbent.policy_hash == self.candidate.policy_hash:
            raise ValueError("candidate must differ from incumbent")
        for name, value in (
            ("delta_proxy", self.delta_proxy),
            ("delta_actor", self.delta_actor),
            ("delta_strategic", self.delta_strategic),
            ("proxy_incumbent_score", self.proxy_incumbent_score),
            ("proxy_candidate_score", self.proxy_candidate_score),
            ("actor_incumbent_score", self.actor_incumbent_score),
            ("actor_candidate_score", self.actor_candidate_score),
        ):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        object.__setattr__(self, "footprint", MappingProxyType({str(k): float(v) for k, v in self.footprint.items()}))
        object.__setattr__(self, "resource_metrics", MappingProxyType(dict(self.resource_metrics)))
        key = {
            "run_id": self.run_id,
            "scaffold": self.scaffold,
            "operator": self.operator,
            "task_family": self.task_family,
            "round_index": self.round_index,
            "candidate_index": self.candidate_index,
            "incumbent_hash": self.incumbent.policy_hash,
            "candidate_hash": self.candidate.policy_hash,
            "seed": self.seed,
        }
        object.__setattr__(self, "transition_id", content_hash(key, length=20))
        if self.terminal_state is not None and not validate_terminal_state(self.terminal_state):
            raise ValueError(f"unknown terminal state: {self.terminal_state}")

    @property
    def proxy_positive(self) -> bool:
        return self.delta_proxy > 0.0

    @property
    def actor_reversal(self) -> bool:
        return self.delta_actor is not None and self.proxy_positive and self.delta_actor < 0.0

    @property
    def strategic_reversal(self) -> bool:
        return (
            self.delta_strategic is not None
            and self.delta_actor is not None
            and self.delta_actor > 0.0
            and self.delta_strategic < 0.0
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "run_id": self.run_id,
            "scaffold": self.scaffold,
            "operator": self.operator,
            "task_family": self.task_family,
            "round": self.round_index,
            "candidate_index": self.candidate_index,
            "incumbent_hash": self.incumbent.policy_hash,
            "candidate_hash": self.candidate.policy_hash,
            "incumbent_policy": self.incumbent.to_record(),
            "candidate_policy": self.candidate.to_record(),
            "delta_proxy": self.delta_proxy,
            "delta_actor": self.delta_actor,
            "delta_strategic": self.delta_strategic,
            "proxy_positive": self.proxy_positive,
            "actor_reversal": self.actor_reversal,
            "strategic_reversal": self.strategic_reversal,
            "footprint": dict(self.footprint),
            "resource_metrics": dict(self.resource_metrics),
            "seed": self.seed,
            "paired_seed_ids": list(self.paired_seed_ids),
            "source_digest": self.source_digest,
            "config_hash": self.config_hash,
            "hf_queried": self.hf_queried,
            "hf_query_reason": self.hf_query_reason,
            "terminal_state": self.terminal_state,
            "proxy_incumbent_score": self.proxy_incumbent_score,
            "proxy_candidate_score": self.proxy_candidate_score,
            "actor_incumbent_score": self.actor_incumbent_score,
            "actor_candidate_score": self.actor_candidate_score,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> TransitionRecord:
        incumbent = AgentPolicy.from_record(record.get("incumbent_policy", {}))
        candidate = AgentPolicy.from_record(record.get("candidate_policy", {}))
        transition = cls(
            run_id=str(record["run_id"]),
            scaffold=str(record["scaffold"]),
            operator=str(record["operator"]),
            task_family=str(record["task_family"]),
            round_index=int(record.get("round", record.get("round_index", 0))),
            candidate_index=int(record["candidate_index"]),
            incumbent=incumbent,
            candidate=candidate,
            delta_proxy=float(record["delta_proxy"]),
            delta_actor=None if record.get("delta_actor") is None else float(record["delta_actor"]),
            delta_strategic=None
            if record.get("delta_strategic") is None
            else float(record["delta_strategic"]),
            footprint=record.get("footprint", {}),
            resource_metrics=record.get("resource_metrics", {}),
            seed=None if record.get("seed") is None else int(record["seed"]),
            paired_seed_ids=tuple(int(value) for value in record.get("paired_seed_ids", [])),
            source_digest=record.get("source_digest"),
            config_hash=record.get("config_hash"),
            hf_queried=bool(record.get("hf_queried", False)),
            hf_query_reason=record.get("hf_query_reason"),
            terminal_state=record.get("terminal_state"),
            proxy_incumbent_score=None
            if record.get("proxy_incumbent_score") is None
            else float(record["proxy_incumbent_score"]),
            proxy_candidate_score=None
            if record.get("proxy_candidate_score") is None
            else float(record["proxy_candidate_score"]),
            actor_incumbent_score=None
            if record.get("actor_incumbent_score") is None
            else float(record["actor_incumbent_score"]),
            actor_candidate_score=None
            if record.get("actor_candidate_score") is None
            else float(record["actor_candidate_score"]),
        )
        if str(record.get("incumbent_hash", incumbent.policy_hash)) != incumbent.policy_hash:
            raise ValueError("incumbent_hash does not match policy content")
        if str(record.get("candidate_hash", candidate.policy_hash)) != candidate.policy_hash:
            raise ValueError("candidate_hash does not match policy content")
        if str(record.get("transition_id", transition.transition_id)) != transition.transition_id:
            raise ValueError("transition_id does not match transition content")
        return transition


def _table_row(row: Mapping[str, Any], columns: Sequence[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for column in columns:
        value = row.get(column)
        if value is None and column.startswith("footprint_"):
            footprint = row.get("footprint", {})
            if isinstance(footprint, Mapping):
                value = footprint.get(column.removeprefix("footprint_"))
        if value is None and column.startswith("resource_"):
            resources = row.get("resource_metrics", {})
            if isinstance(resources, Mapping):
                key = column.removeprefix("resource_")
                value = resources.get(key)
                if value is None:
                    # The local runner uses ``tokens`` while external adapters
                    # may expose the more explicit ``token_usage`` spelling.
                    aliases = {
                        "token_usage": "tokens",
                        "timeout": "timeouts",
                        "crash": "crashes",
                    }
                    value = resources.get(aliases.get(key, key))
        output[column] = canonical_json(value) if isinstance(value, (Mapping, list, tuple)) else value
    return output


def write_table(
    rows: Sequence[Mapping[str, Any]],
    stem: Path,
    *,
    columns: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Write a canonical table as deterministic CSV and Parquet files."""

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    selected = tuple(columns) if columns is not None else tuple(sorted({str(key) for row in rows for key in row}))
    if not rows and columns is None:
        raise ValueError("columns are required for an empty table")
    normalized = [_table_row(row, selected) for row in rows]
    csv_path = stem.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized)
    parquet_path = stem.with_suffix(".parquet")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - exercised only without optional dependency
        raise RuntimeError("pyarrow is required to write canonical Parquet artifacts") from error
    # Cast values to strings explicitly so null/mixed diagnostic columns remain portable.
    string_rows = [{column: None if row[column] is None else str(row[column]) for column in selected} for row in normalized]
    table = pa.Table.from_pylist(string_rows, schema=pa.schema([(column, pa.string()) for column in selected]))
    pq.write_table(table, parquet_path, compression="zstd", compression_level=9, version="2.6")
    return {"csv": csv_path, "parquet": parquet_path}


def write_jsonl(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """Write sorted-key JSONL with a trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_plain(row), sort_keys=True, ensure_ascii=True) + "\n")
    return path
