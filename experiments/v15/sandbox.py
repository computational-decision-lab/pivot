"""Fresh paired sandbox execution for local construct-validity checks.

This is a deliberately small reference runner, not a substitute for a pinned
external coding-agent scaffold.  It proves the isolation and provenance
contract used by the eventual adapters.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .planes import TaskSpec
from .protocol import AgentPolicy


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class SandboxExecution:
    task_id: str
    policy_hash: str
    initial_manifest_hash: str
    root_hash: str
    success: float
    resource_metrics: dict[str, Any]
    trace: tuple[str, ...]


@dataclass(frozen=True)
class PairedSandboxResult:
    incumbent: SandboxExecution
    candidate: SandboxExecution

    @property
    def delta(self) -> float:
        return self.candidate.success - self.incumbent.success

    @property
    def behavioral_footprint(self) -> dict[str, float]:
        """Return pre-gate behavioral shifts from the paired traces.

        These descriptors are intentionally limited to execution behavior that
        is observable before any sealed deployment outcome is queried.
        """

        left = self.incumbent.resource_metrics
        right = self.candidate.resource_metrics

        def difference(key: str) -> float:
            try:
                return float(right.get(key, 0.0)) - float(left.get(key, 0.0))
            except (TypeError, ValueError):
                return 0.0

        left_trace = set(self.incumbent.trace)
        right_trace = set(self.candidate.trace)
        union = left_trace | right_trace
        action_distance = 0.0 if not union else 1.0 - len(left_trace & right_trace) / len(union)
        return {
            "tool_call_distribution_shift": abs(difference("tool_calls")),
            "shell_command_distribution_shift": 0.0,
            "test_execution_shift": abs(difference("tests_executed")),
            "files_read_shift": abs(difference("files_read")),
            "files_written_shift": abs(difference("files_written")),
            "dependency_operation_shift": abs(difference("dependency_operations")),
            "token_usage_shift": abs(difference("tokens")),
            "context_peak_shift": abs(difference("context_peak")),
            "wall_clock_shift": abs(difference("wall_clock_seconds")),
            "action_sequence_distance": action_distance,
        }


class PairedSandboxRunner:
    """Create independent copies of one task and evaluate a policy pair."""

    def evaluate_pair(
        self, task: TaskSpec, incumbent: AgentPolicy, candidate: AgentPolicy, *, seed: int
    ) -> PairedSandboxResult:
        with tempfile.TemporaryDirectory(prefix="pivot-paired-") as temporary:
            root = Path(temporary)
            base = root / "base"
            self._materialize(task, base)
            initial_manifest_hash = _tree_hash(base)
            incumbent_root = root / "incumbent"
            candidate_root = root / "candidate"
            shutil.copytree(base, incumbent_root)
            shutil.copytree(base, candidate_root)
            incumbent_result = self._run(task, incumbent, incumbent_root, seed, initial_manifest_hash)
            candidate_result = self._run(task, candidate, candidate_root, seed, initial_manifest_hash)
        return PairedSandboxResult(incumbent=incumbent_result, candidate=candidate_result)

    @staticmethod
    def _materialize(task: TaskSpec, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        for relative, content in task.files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    @staticmethod
    def _run(
        task: TaskSpec, policy: AgentPolicy, root: Path, seed: int, initial_manifest_hash: str
    ) -> SandboxExecution:
        trace = ["fresh_sandbox", "read_task"]
        target = str(task.metadata.get("target", "app.py"))
        target_path = root / target
        tests_requested = bool(policy.test_policy.get("run_tests", False))
        repair_requested = bool(policy.test_policy.get("repair", False))
        search_depth = int(policy.search_policy.get("depth", 0))
        max_steps = int(policy.agent_loop_config.get("max_steps", 0))
        prompt_supports_repair = "smallest correct edit" in policy.system_prompt.casefold()
        repaired = False
        if target_path.is_file() and repair_requested and search_depth >= 2 and max_steps >= 4:
            source = target_path.read_text(encoding="utf-8")
            source = source.replace("BUG = True", "BUG = False")
            target_path.write_text(source, encoding="utf-8")
            repaired = "BUG = False" in source
            trace.append("write_patch")
        if tests_requested:
            trace.append("run_tests")
        success = float(repaired and tests_requested and prompt_supports_repair)
        resource_metrics: dict[str, Any] = {
            "fresh_sandbox": True,
            "seed": seed,
            "tool_calls": 2 + int(tests_requested) + int(repaired),
            "tests_executed": int(tests_requested),
            "files_read": min(search_depth, len(task.files)),
            "files_written": int(repaired),
            "tokens": len(policy.system_prompt.split()),
            "context_peak": int(policy.context_policy.get("max_tokens", 0)),
            "wall_clock_seconds": 0.0,
            "timeouts": 0,
            "crashes": 0,
            "dependency_operations": 0,
            "cpu_seconds": 0.0,
            "memory_mb": 0.0,
            "max_steps": max_steps,
        }
        return SandboxExecution(
            task_id=task.task_id,
            policy_hash=policy.policy_hash,
            initial_manifest_hash=initial_manifest_hash,
            root_hash=_tree_hash(root),
            success=success,
            resource_metrics=resource_metrics,
            trace=tuple(trace),
        )
