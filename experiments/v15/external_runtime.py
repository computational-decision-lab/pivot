"""Auditable execution bridge for the V15 external agent study.

The project deliberately keeps the external runtime behind this module.  The
research code owns task-plane access, pairing, hashing, and artifact
provenance; the pinned mini-SWE-agent owns the coding trajectory.  Inspect AI
is used as the task/score/log control plane, while the coding agent itself runs
in a fresh Docker container for every policy-task pair.

This module has no import-time dependency on either external framework.  The
normal project test environment can therefore validate the contracts without
opening a model or a sealed task plane.  Actual execution is performed with
``.tools/v15/runtime/bin/python`` after the pinned runtime has been installed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .planes import TaskSpec
from .protocol import AgentPolicy, canonical_json

_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "auth_token",
    "token",
    "password",
    "secret",
}
_COMMAND_WORDS = re.compile(r"\b(?:cat|head|tail|sed|awk|grep|rg|find|ls|tree|git\s+show)\b")
_WRITE_WORDS = re.compile(r"\b(?:tee|cp|mv|touch|mkdir|rm|printf|echo|sed\s+-i|python(?:3)?\s+-c)\b")
_TEST_WORDS = re.compile(
    r"\b(?:pytest|unittest|tox|nox|mypy|ruff\s+check|npm\s+test|go\s+test)\b",
    re.IGNORECASE,
)
_EVALUATOR_DIR_NAMES = frozenset({"tests", "test", "__tests__", "spec", "specs"})
_EVALUATOR_FILE_PREFIXES = ("test_", "test-", "test.", "spec_", "spec-", "spec.")
_EVALUATOR_FILE_SUFFIXES = (
    "_test",
    "-test",
    ".test",
    "_tests",
    "-tests",
    ".tests",
    "_spec",
    "-spec",
    ".spec",
)


def locked_runtime_python(root: Path) -> Path:
    """Return the pinned external-runtime interpreter.

    The project test environment is Python 3.10, while the locked
    Inspect/mini-SWE/LiteLLM environment is Python 3.11.  Resolving the
    interpreter from the provenance manifest prevents a command launched from
    the project venv from silently importing an incompatible dependency set.
    """

    root = Path(root).resolve()
    manifest_path = root / "configs/v15/runtime_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"external runtime manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime_value = payload.get("runtime_python") if isinstance(payload, Mapping) else None
    if not isinstance(runtime_value, str) or not runtime_value:
        raise ValueError(f"runtime_python is missing from {manifest_path}")
    # Keep the manifest's venv entry point rather than resolving its ``python``
    # symlink.  Executing the resolved base interpreter would silently lose the
    # external environment's site-packages and break the scientific runtime.
    runtime = root / runtime_value
    if not runtime.is_file() or not os.access(runtime, os.X_OK):
        raise FileNotFoundError(f"locked external runtime is unavailable: {runtime}")
    return runtime


def running_under_locked_runtime(root: Path) -> bool:
    """Return whether the current interpreter is the locked runtime."""

    try:
        return Path(sys.executable).resolve() == locked_runtime_python(root).resolve()
    except (FileNotFoundError, ValueError):
        return False


@dataclass(frozen=True)
class RuntimeSettings:
    """Fully resolved runtime settings after the scientific lock is chosen."""

    model_name: str
    provider: str
    api_base: str
    image: str
    image_digest: str
    dependency_lock: str
    artifact_root: Path
    log_root: Path
    token_limit: int = 2048
    wall_clock_seconds: int = 120
    tool_calls: int = 16
    container_cpu: int = 1
    container_memory_mb: int = 2048
    max_parallel: int = 1
    agent_step_limit: int = 8
    model_output_tokens: int = 512

    def __post_init__(self) -> None:
        if not self.model_name or not self.provider:
            raise ValueError("model_name and provider must be fixed before execution")
        if not self.image or not self.image_digest:
            raise ValueError("sandbox image and digest must be fixed before execution")
        if not self.dependency_lock:
            raise ValueError("dependency_lock must be fixed before execution")
        if not self.image_digest.startswith("sha256:") or self.image_digest.strip() != self.image_digest:
            raise ValueError("image_digest must be an immutable sha256 digest")
        lock_path = Path(self.dependency_lock)
        if lock_path.is_absolute() or ".." in lock_path.parts:
            raise ValueError("dependency_lock must be a repository-relative path")
        for name in (
            "token_limit",
            "wall_clock_seconds",
            "tool_calls",
            "container_cpu",
            "container_memory_mb",
            "max_parallel",
            "agent_step_limit",
            "model_output_tokens",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def model_key(self) -> str:
        """Return the provider-qualified model name expected by LiteLLM."""

        return self.model_name if "/" in self.model_name else f"{self.provider}/{self.model_name}"

    def to_manifest(self) -> dict[str, Any]:
        """Return the non-secret runtime manifest used by provenance checks."""

        return runtime_manifest(self)


@dataclass(frozen=True)
class ExecutionRecord:
    """One policy-task execution, including only non-secret provenance."""

    status: str
    terminal_state: str | None
    task_id: str
    task_hash: str
    policy_hash: str
    seed: int
    success: float
    inspect_log: str | None
    trajectory: str | None
    initial_tree_hash: str
    final_tree_hash: str
    test_returncode: int
    exit_status: str
    resource_metrics: dict[str, float]
    trace: tuple[str, ...]
    error: str | None = None
    final_tree_path: str | None = None
    budget_violation: str | None = None

    def to_record(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trace"] = list(self.trace)
        payload["resource_metrics"] = dict(self.resource_metrics)
        return payload


@dataclass(frozen=True)
class PairedExecutionRecord:
    """One same-task paired rollout used to estimate a policy difference."""

    task_id: str
    task_hash: str
    seed: int
    incumbent_policy_hash: str
    candidate_policy_hash: str
    incumbent_success: float
    candidate_success: float
    incumbent_execution: str | None
    candidate_execution: str | None
    inspect_log: str | None
    incumbent_final_tree_path: str | None = None
    candidate_final_tree_path: str | None = None
    incumbent_resource_metrics: dict[str, float] | None = None
    candidate_resource_metrics: dict[str, float] | None = None
    incumbent_trace: tuple[str, ...] = ()
    candidate_trace: tuple[str, ...] = ()
    incumbent_status: str = "COMPLETED"
    candidate_status: str = "COMPLETED"
    incumbent_error: str | None = None
    candidate_error: str | None = None
    # These fields make the paired-rollout contract explicit.  They are added
    # after the existing defaults so older artifact readers/constructors remain
    # source-compatible.
    incumbent_initial_tree_hash: str | None = None
    candidate_initial_tree_hash: str | None = None
    pairing_contract_hash: str | None = None

    @property
    def delta(self) -> float:
        """Return candidate success minus incumbent success."""

        return float(self.candidate_success - self.incumbent_success)

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_hash": self.task_hash,
            "seed": self.seed,
            "incumbent_policy_hash": self.incumbent_policy_hash,
            "candidate_policy_hash": self.candidate_policy_hash,
            "incumbent_success": self.incumbent_success,
            "candidate_success": self.candidate_success,
            "delta": self.delta,
            "incumbent_execution": self.incumbent_execution,
            "candidate_execution": self.candidate_execution,
            "inspect_log": self.inspect_log,
            "incumbent_final_tree_path": self.incumbent_final_tree_path,
            "candidate_final_tree_path": self.candidate_final_tree_path,
            "incumbent_resource_metrics": self.incumbent_resource_metrics or {},
            "candidate_resource_metrics": self.candidate_resource_metrics or {},
            "incumbent_trace": list(self.incumbent_trace),
            "candidate_trace": list(self.candidate_trace),
            "incumbent_status": self.incumbent_status,
            "candidate_status": self.candidate_status,
            "incumbent_error": self.incumbent_error,
            "candidate_error": self.candidate_error,
            "incumbent_initial_tree_hash": self.incumbent_initial_tree_hash,
            "candidate_initial_tree_hash": self.candidate_initial_tree_hash,
            "pairing_contract_hash": self.pairing_contract_hash,
        }


def paired_execution_failed(pair: PairedExecutionRecord) -> bool:
    """Return whether either side of a pair is an execution failure."""

    return (
        pair.incumbent_status != "COMPLETED"
        or pair.candidate_status != "COMPLETED"
        or not pair.incumbent_execution
        or not pair.candidate_execution
    )


def budget_violation(metrics: Mapping[str, Any], settings: RuntimeSettings) -> str | None:
    """Return a deterministic resource-budget violation, if one occurred.

    The evaluator's primary utility is measured under a fixed deployment
    budget.  The scaffold's native limits stop most overruns, but provider and
    post-run bookkeeping can still reveal an over-budget trajectory.  This
    single check is therefore applied to every persisted execution record so a
    rollout cannot be reported as a success merely because its tests passed.
    Missing counters are treated as zero (useful for an execution that failed
    before an agent was constructed); non-finite counters fail closed.
    """

    checks = (
        ("tokens", int(settings.token_limit), "token"),
        ("tool_calls", int(settings.tool_calls), "tool-call"),
        ("wall_clock_seconds", float(settings.wall_clock_seconds), "wall-clock"),
    )
    violations: list[str] = []
    for key, limit, label in checks:
        raw = metrics.get(key, 0.0)
        try:
            observed = float(raw)
        except (TypeError, ValueError):
            violations.append(f"{label} budget is not numeric: {raw!r}")
            continue
        if not math.isfinite(observed) or observed < 0:
            violations.append(f"{label} budget is invalid: observed={raw!r}")
        elif observed > float(limit):
            rendered = f"{observed:.3f}" if key == "wall_clock_seconds" else str(int(observed))
            violations.append(f"{label} budget exceeded: observed={rendered} limit={limit}")
    return "; ".join(violations) if violations else None


def annotate_budget_metrics(metrics: Mapping[str, Any], settings: RuntimeSettings) -> tuple[dict[str, float], str | None]:
    """Copy metrics, add the registered limits, and return the violation.

    Keeping the output numeric makes the resource table schema stable across
    Inspect, mini-SWE, and Pi adapters.  The human-readable reason is stored
    separately on the execution record.
    """

    annotated: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            annotated[str(key)] = float(value)
        except (TypeError, ValueError):
            # Resource tables are numeric by contract; malformed counters are
            # represented by a NaN and consequently fail the budget check.
            annotated[str(key)] = float("nan")
    for key in ("tokens", "tool_calls", "wall_clock_seconds"):
        annotated.setdefault(key, 0.0)
    violation = budget_violation(annotated, settings)
    annotated.update(
        {
            "budget_token_limit": float(settings.token_limit),
            "budget_tool_call_limit": float(settings.tool_calls),
            "budget_wall_clock_limit": float(settings.wall_clock_seconds),
            "budget_within_limits": 1.0 if violation is None else 0.0,
            "budget_violation_count": float(bool(violation)),
        }
    )
    return annotated, violation


def _sanitize(value: Any, *, secret_values: Sequence[str] = ()) -> Any:
    """Redact credential-shaped fields before a trajectory is persisted."""

    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).casefold()
            if key_text in _SECRET_KEYS or any(part in key_text for part in ("api_key", "authorization", "auth_token")):
                clean[str(key)] = "<redacted>"
            else:
                clean[str(key)] = _sanitize(item, secret_values=secret_values)
        return clean
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, secret_values=secret_values) for item in value]
    if isinstance(value, str):
        result = value
        for secret in secret_values:
            if secret:
                result = result.replace(secret, "<redacted>")
        return result
    return value


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and ".pi-session" not in item.relative_to(root).parts
    ):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _is_evaluator_path(relative: str | Path) -> bool:
    """Return whether a relative path belongs to the protected evaluator.

    The check is lexical and conservative.  It covers common test/spec
    directory and filename conventions, including files created after the
    sandbox snapshot (for example ``test.py``), while avoiding false positives
    such as ``contest.py`` or ``testing_helpers.py``.
    """

    text = str(relative).replace("\\", "/").strip("/")
    if not text:
        return False
    path = Path(text)
    parts = tuple(part.casefold() for part in path.parts)
    if any(part in _EVALUATOR_DIR_NAMES for part in parts):
        return True
    name = path.name.casefold()
    stem = Path(name).stem
    if stem in {"test", "tests", "spec", "specs"}:
        return True
    if name.startswith(_EVALUATOR_FILE_PREFIXES):
        return True
    return stem.endswith(_EVALUATOR_FILE_SUFFIXES)


def _evaluator_paths(root: Path) -> tuple[Path, ...]:
    """Enumerate protected evaluator files/symlinks without following dirs."""

    paths: list[Path] = []
    for item in root.rglob("*"):
        try:
            relative = item.relative_to(root)
        except ValueError:
            continue
        if not _is_evaluator_path(relative):
            continue
        # Check symlinks before regular files so a link to an external file is
        # treated as tampering rather than evaluator data.
        if item.is_symlink() or item.is_file():
            paths.append(item)
    return tuple(sorted(paths, key=lambda item: item.relative_to(root).as_posix()))


def pairing_contract_hash(
    task: TaskSpec,
    settings: RuntimeSettings,
    *,
    seed: int,
    phase: str,
    role: str,
) -> str:
    """Hash the conditions that must be shared by both sides of a pair.

    The policy hash is intentionally absent: the policy is the intervention.
    Every other material evaluation condition is included so a silently
    mismatched image, budget, or task snapshot cannot be reported as a paired
    estimate.
    """

    payload = {
        "task_id": task.task_id,
        "task_hash": task.task_hash,
        "seed": int(seed),
        "phase": str(phase),
        "role": str(role),
        "image": settings.image,
        "image_digest": settings.image_digest,
        "dependency_lock": settings.dependency_lock,
        "token_limit": int(settings.token_limit),
        "wall_clock_seconds": int(settings.wall_clock_seconds),
        "tool_calls": int(settings.tool_calls),
        "container_cpu": int(settings.container_cpu),
        "container_memory_mb": int(settings.container_memory_mb),
        "agent_step_limit": int(settings.agent_step_limit),
        "model_output_tokens": int(settings.model_output_tokens),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _protected_evaluator_files(task: TaskSpec) -> dict[str, bytes]:
    """Snapshot test/evaluator files that an agent is never allowed to edit."""

    protected: dict[str, bytes] = {}
    for relative, content in task.files.items():
        if _is_evaluator_path(relative):
            protected[relative] = content.encode("utf-8")
    return protected


def _restore_evaluator_files(root: Path, snapshot: Mapping[str, bytes]) -> tuple[str, ...]:
    """Restore protected files and remove newly-created evaluator files.

    A candidate is not allowed to improve its score by editing or adding test
    code.  New evaluator-shaped files are removed from the disposable sandbox
    and reported as tampering; unrelated source files are left untouched.
    """

    changed: list[str] = []
    snapshot_paths = set(snapshot)
    current_paths = {
        path.relative_to(root).as_posix(): path for path in _evaluator_paths(root)
    }
    for relative in sorted(set(current_paths) - snapshot_paths):
        path = current_paths[relative]
        try:
            # Unlink only the file/symlink itself.  The sandbox is disposable,
            # but avoiding recursive deletion keeps this helper recoverable.
            path.unlink()
        except OSError:
            # Keep the tamper record even if a malformed path cannot be
            # removed; the caller will mark the execution as a failure.
            pass
        changed.append(relative)
    for relative, original in snapshot.items():
        path = root / relative
        current = None
        if path.is_file() and not path.is_symlink():
            try:
                current = path.read_bytes()
            except OSError:
                current = None
        if current != original:
            changed.append(relative)
            if path.is_symlink() or path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(original)
    return tuple(sorted(changed))


def _pinned_image_reference(settings: RuntimeSettings) -> str:
    """Return an immutable Docker image reference for every execution."""

    image = settings.image.strip()
    digest = settings.image_digest.strip()
    if not image or not digest:
        raise ValueError("sandbox image and digest must be fixed before execution")
    if "@" in image:
        name, existing_digest = image.rsplit("@", 1)
        if not name or existing_digest != digest:
            raise ValueError("sandbox image contains a digest different from image_digest")
        return image
    return f"{image}@{digest}"


def _task_instruction(task: TaskSpec) -> str:
    instruction = task.metadata.get("instruction")
    if instruction:
        return str(instruction)
    target = task.metadata.get("target", "the failing behavior")
    return f"Repair {target} in the repository and verify the registered tests pass."


def _test_command(task: TaskSpec) -> str:
    return str(task.metadata.get("test_command", "python -m unittest discover -v"))


def _model_kwargs(settings: RuntimeSettings) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Resolve provider credentials without ever returning them for storage."""

    token = os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY")
    if settings.provider.casefold() == "anthropic":
        if not token:
            raise RuntimeError("ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY is required")
        kwargs: dict[str, Any] = {
            "api_base": settings.api_base,
            "api_key": token,
            "max_tokens": settings.model_output_tokens,
            "temperature": 0,
            "drop_params": True,
        }
        return kwargs, (token,)
    if settings.provider.casefold() == "openai":
        token = os.getenv("OPENAI_API_KEY")
        if not token:
            raise RuntimeError("OPENAI_API_KEY is required")
        kwargs = {
            "api_base": settings.api_base,
            "api_key": token,
            "max_tokens": settings.model_output_tokens,
            "temperature": 0,
            "drop_params": True,
        }
        return kwargs, (token,)
    raise ValueError(f"unsupported provider: {settings.provider}")


def _policy_prompt(policy: AgentPolicy) -> str:
    """Map registered policy dimensions into the scaffold's prompt contract."""

    search = dict(policy.search_policy)
    tests = dict(policy.test_policy)
    context = dict(policy.context_policy)
    return (
        f"{policy.system_prompt}\n\n"
        "You are operating under a fixed evaluation budget. Use only the bash tool, "
        "inspect the repository before editing, make a minimal correct change, and "
        "run the registered tests before finishing. Do not modify evaluator files, "
        "task metadata, or files outside the repository.\n"
        f"Search depth target: {search.get('depth', 2)}; file budget: {search.get('max_files', 12)}.\n"
        f"Run tests: {bool(tests.get('run_tests', True))}; repair loop: {bool(tests.get('repair', False))}.\n"
        f"Context budget: {context.get('max_tokens', 2048)} tokens; summarize context: {bool(context.get('summarize', True))}."
    )


def _command_trace(agent: Any) -> tuple[str, ...]:
    commands: list[str] = []
    for message in getattr(agent, "messages", []):
        extra = message.get("extra", {}) if isinstance(message, Mapping) else {}
        actions = extra.get("actions", []) if isinstance(extra, Mapping) else []
        for action in actions:
            if isinstance(action, Mapping) and action.get("command") is not None:
                commands.append(str(action["command"]))
    return tuple(commands)


def _token_count(agent: Any) -> float:
    total = 0.0
    for message in getattr(agent, "messages", []):
        extra = message.get("extra", {}) if isinstance(message, Mapping) else {}
        response = extra.get("response", {}) if isinstance(extra, Mapping) else {}
        usage = response.get("usage", {}) if isinstance(response, Mapping) else {}
        if isinstance(usage, Mapping):
            raw_total = usage.get("total_tokens")
            try:
                if raw_total is not None and math.isfinite(float(raw_total)):
                    total += float(raw_total)
                    continue
            except (TypeError, ValueError):
                pass
            # Providers do not agree on usage field names.  Prefer the
            # prompt/completion pair when present, then fall back to the
            # input/output pair used by newer message APIs.  Do not add both
            # aliases when a response exposes both layouts.
            key_groups = (("prompt_tokens", "completion_tokens"), ("input_tokens", "output_tokens"))
            for keys in key_groups:
                values = [usage.get(key) for key in keys]
                if not any(value is not None for value in values):
                    continue
                for value in values:
                    try:
                        numeric = float(value or 0.0)
                    except (TypeError, ValueError):
                        numeric = float("nan")
                    if math.isfinite(numeric) and numeric >= 0.0:
                        total += numeric
                break
    return total


def _metrics(trace: Sequence[str], agent: Any, elapsed: float) -> dict[str, float]:
    joined = "\n".join(trace)
    model_calls = 0
    for message in getattr(agent, "messages", []):
        if not isinstance(message, Mapping):
            continue
        extra = message.get("extra", {})
        if message.get("role") == "assistant" or (isinstance(extra, Mapping) and "response" in extra):
            model_calls += 1
    return {
        "model_calls": float(model_calls),
        "tool_calls": float(len(trace)),
        "tokens": _token_count(agent),
        "context_peak": float(max((len(str(item)) for item in getattr(agent, "messages", [])), default=0)),
        "files_read": float(len(_COMMAND_WORDS.findall(joined))),
        "files_written": float(len(_WRITE_WORDS.findall(joined))),
        "tests_executed": float(len(_TEST_WORDS.findall(joined))),
        "dependency_operations": float(sum("pip install" in item or "npm install" in item for item in trace)),
        "wall_clock_seconds": float(elapsed),
    }


def execute_mini_task(
    task: TaskSpec,
    policy: AgentPolicy,
    settings: RuntimeSettings,
    *,
    seed: int,
    artifact_dir: Path,
) -> ExecutionRecord:
    """Run one real mini-SWE-agent trajectory in a fresh Docker copy."""

    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.docker import DockerEnvironment
    from minisweagent.models.litellm_model import LitellmModel

    artifact_dir.mkdir(parents=True, exist_ok=True)
    sandbox_parent = artifact_dir / "sandboxes"
    sandbox_parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    trajectory_path = artifact_dir / f"{task.task_id}__{policy.policy_hash[:12]}__{seed}.traj.json"
    test_returncode = -1
    exit_status = "not_started"
    error: str | None = None
    trace: tuple[str, ...] = ()
    initial_hash = ""
    final_hash = ""
    success = 0.0
    agent: Any | None = None
    env: Any | None = None
    secret_values: tuple[str, ...] = ()
    evaluator_snapshot = _protected_evaluator_files(task)
    evaluator_tamper_paths: tuple[str, ...] = ()
    with tempfile.TemporaryDirectory(prefix=f"v15-{task.task_id}-", dir=sandbox_parent) as sandbox_name:
        sandbox = Path(sandbox_name)
        for relative, content in task.files.items():
            destination = sandbox / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
        initial_hash = _tree_hash(sandbox)
        try:
            model_kwargs, secret_values = _model_kwargs(settings)
            model = LitellmModel(
                model_name=settings.model_key,
                model_kwargs=model_kwargs,
                cost_tracking="ignore_errors",
            )
            run_args = [
                "--rm",
                "--cpus",
                str(settings.container_cpu),
                "--memory",
                f"{settings.container_memory_mb}m",
                "--network",
                "none",
                "--pids-limit",
                "256",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "-v",
                f"{sandbox}:/workspace",
            ]
            env = DockerEnvironment(
                image=_pinned_image_reference(settings),
                cwd="/workspace",
                run_args=run_args,
                timeout=min(30, settings.wall_clock_seconds),
                container_timeout=f"{settings.wall_clock_seconds + 60}s",
                pull_timeout=180,
            )
            agent = DefaultAgent(
                model,
                env,
                system_template=_policy_prompt(policy),
                instance_template=(
                    "Solve this repository task: {{task}}\n"
                    "You may inspect and edit repository files with bash. "
                    "Finish only after running the tests and issuing the completion command."
                ),
                step_limit=min(settings.agent_step_limit, max(1, int(policy.agent_loop_config.get("max_steps", settings.agent_step_limit)))),
                cost_limit=0,
                wall_time_limit_seconds=settings.wall_clock_seconds,
                max_consecutive_format_errors=2,
            )
            result = agent.run(_task_instruction(task))
            exit_status = str(result.get("exit_status", ""))
            evaluator_tamper_paths = _restore_evaluator_files(sandbox, evaluator_snapshot)
            if evaluator_tamper_paths:
                error = "agent modified protected evaluator files: " + ", ".join(evaluator_tamper_paths)
            test_result = env.execute({"command": _test_command(task)}, cwd="/workspace")
            test_returncode = int(test_result.get("returncode", -1))
            success = 1.0 if test_returncode == 0 else 0.0
            trace = _command_trace(agent)
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            # External model/scaffold failures are recorded as implementation
            # failures so a partial trajectory cannot be mistaken for data.
            error = f"{type(exc).__name__}: {exc}"
            exit_status = type(exc).__name__
            if agent is not None:
                trace = _command_trace(agent)
        finally:
            restored_tamper = _restore_evaluator_files(sandbox, evaluator_snapshot)
            if restored_tamper:
                evaluator_tamper_paths = restored_tamper
                error = "agent modified protected evaluator files: " + ", ".join(restored_tamper)
            if agent is not None:
                payload = _sanitize(agent.serialize(), secret_values=secret_values)
                payload["v15_execution"] = {
                    "task_id": task.task_id,
                    "task_hash": task.task_hash,
                    "policy_hash": policy.policy_hash,
                    "seed": seed,
                    "image_digest": settings.image_digest,
                    "dependency_lock": settings.dependency_lock,
                }
                trajectory_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if env is not None:
                env.cleanup()
        final_hash = _tree_hash(sandbox)
        final_tree_path = artifact_dir / f"{task.task_id}__{policy.policy_hash[:12]}__{seed}.final_tree"
        if final_tree_path.exists():
            shutil.rmtree(final_tree_path)
        shutil.copytree(
            sandbox,
            final_tree_path,
            ignore=shutil.ignore_patterns(".pi-session"),
        )
    elapsed = time.monotonic() - started
    metrics = _metrics(trace, agent, elapsed) if agent is not None else {"wall_clock_seconds": float(elapsed)}
    metrics, budget_error = annotate_budget_metrics(metrics, settings)
    metrics["evaluator_tamper_detected"] = float(bool(evaluator_tamper_paths))
    metrics["protected_evaluator_file_count"] = float(len(evaluator_snapshot))
    if budget_error:
        # A passing test does not override the registered deployment budget.
        # Preserve any earlier integrity error while making the budget breach
        # explicit in both the record and the numeric resource table.
        error = f"{error}; {budget_error}" if error else budget_error
        success = 0.0
        exit_status = exit_status or "BudgetExceeded"
    record = ExecutionRecord(
        status="COMPLETED" if error is None else "IMPLEMENTATION_FAILURE",
        terminal_state=None if error is None else "IMPLEMENTATION_FAILURE",
        task_id=task.task_id,
        task_hash=task.task_hash,
        policy_hash=policy.policy_hash,
        seed=int(seed),
        success=success,
        inspect_log=None,
        trajectory=str(trajectory_path),
        initial_tree_hash=initial_hash,
        final_tree_hash=final_hash,
        test_returncode=test_returncode,
        exit_status=exit_status,
        resource_metrics=metrics,
        trace=trace,
        error=error,
        final_tree_path=str(final_tree_path),
        budget_violation=budget_error,
    )
    (artifact_dir / f"{task.task_id}__{policy.policy_hash[:12]}__{seed}.execution.json").write_text(
        json.dumps(record.to_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def evaluate_with_inspect(
    tasks: Sequence[TaskSpec],
    policy: AgentPolicy,
    settings: RuntimeSettings,
    *,
    seed: int,
    phase: str,
    role: str,
    run_id: str,
) -> list[ExecutionRecord]:
    """Run a batch under Inspect's task/scorer/log control plane."""

    if not tasks:
        return []
    from inspect_ai import Task, eval
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import Score, mean, scorer
    from inspect_ai.solver import solver

    task_map = {task.task_id: task for task in tasks}
    artifact_dir = settings.artifact_root / phase / run_id / f"seed-{seed}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    @solver
    def run_external() -> Any:
        async def solve(state: Any, generate: Any) -> Any:
            task = task_map[str(state.sample_id)]
            record = await asyncio.to_thread(
                execute_mini_task,
                task,
                policy,
                settings,
                seed=seed,
                artifact_dir=artifact_dir,
            )
            state.store.set("v15_execution", record.to_record())
            return state

        return solve

    @scorer(metrics=[mean()])
    def score_external() -> Any:
        async def score(state: Any, target: Any) -> Score:
            record = state.store.get("v15_execution", {})
            return Score(
                value=float(record.get("success", 0.0)),
                answer=str(record.get("test_returncode", -1)),
                explanation=str(record.get("exit_status", "")),
                metadata={
                    "task_id": str(record.get("task_id", state.sample_id)),
                    "trajectory": record.get("trajectory"),
                    "resource_metrics": record.get("resource_metrics", {}),
                },
            )

        return score

    samples = [
        Sample(
            input=_task_instruction(task),
            id=task.task_id,
            target="1",
            metadata={"task_id": task.task_id, "task_hash": task.task_hash, "role": role},
        )
        for task in tasks
    ]
    log_dir = settings.log_root / phase / run_id / f"seed-{seed}" / policy.policy_hash[:12]
    log_dir.mkdir(parents=True, exist_ok=True)
    logs = eval(
        Task(
            dataset=samples,
            solver=run_external(),
            scorer=score_external(),
            model="mockllm/model",
            name=f"v15_{phase}_{run_id}_{seed}",
            metadata={"v15_role": role, "policy_hash": policy.policy_hash},
        ),
        model="mockllm/model",
        display="none",
        log_realtime=False,
        log_dir=str(log_dir),
        max_samples=settings.max_parallel,
        fail_on_error=False,
        score_on_error=True,
    )
    by_id: dict[str, ExecutionRecord] = {}
    for log in logs:
        location = str(getattr(log, "location", ""))
        for sample in getattr(log, "samples", []):
            sample_id = str(getattr(sample, "id", ""))
            # Inspect serializes Store data in the sample transcript; the
            # persisted execution JSON remains the canonical source.
            records = list(artifact_dir.glob(f"{sample_id}__{policy.policy_hash[:12]}__{seed}.execution.json"))
            if not records:
                continue
            record_payload = json.loads(records[0].read_text(encoding="utf-8"))
            record_payload["inspect_log"] = location
            record = ExecutionRecord(
                **{**record_payload, "trace": tuple(record_payload.get("trace", []))}
            )
            by_id[sample_id] = record
    missing = [task.task_id for task in tasks if task.task_id not in by_id]
    if missing:
        raise RuntimeError(f"Inspect did not return execution records for: {missing}")
    return [by_id[task.task_id] for task in tasks]


def evaluate_paired_with_inspect(
    tasks: Sequence[TaskSpec],
    incumbent: AgentPolicy,
    candidate: AgentPolicy,
    settings: RuntimeSettings,
    *,
    seed: int,
    phase: str,
    role: str,
    run_id: str,
) -> list[PairedExecutionRecord]:
    """Evaluate two policies on the same sealed task snapshot and seed."""

    if incumbent.policy_hash == candidate.policy_hash:
        raise ValueError("paired policies must differ")
    incumbent_records = evaluate_with_inspect(
        tasks,
        incumbent,
        settings,
        seed=seed,
        phase=f"{phase}/incumbent",
        role=role,
        run_id=run_id,
    )
    candidate_records = evaluate_with_inspect(
        tasks,
        candidate,
        settings,
        seed=seed,
        phase=f"{phase}/candidate",
        role=role,
        run_id=run_id,
    )
    if {record.task_id for record in incumbent_records} != {record.task_id for record in candidate_records}:
        raise RuntimeError("paired executions returned different task sets")
    candidate_by_id = {record.task_id: record for record in candidate_records}
    task_by_id = {task.task_id: task for task in tasks}
    pairs: list[PairedExecutionRecord] = []
    for incumbent_record in incumbent_records:
        candidate_record = candidate_by_id.get(incumbent_record.task_id)
        if candidate_record is None:
            raise RuntimeError(f"paired candidate record missing: {incumbent_record.task_id}")
        if incumbent_record.task_hash != candidate_record.task_hash:
            raise RuntimeError(f"paired task hash mismatch: {incumbent_record.task_id}")
        if incumbent_record.seed != candidate_record.seed or incumbent_record.seed != int(seed):
            raise RuntimeError(f"paired seed mismatch: {incumbent_record.task_id}")
        if incumbent_record.policy_hash != incumbent.policy_hash or candidate_record.policy_hash != candidate.policy_hash:
            raise RuntimeError(f"paired policy hash mismatch: {incumbent_record.task_id}")
        if not incumbent_record.initial_tree_hash or not candidate_record.initial_tree_hash:
            raise RuntimeError(f"paired execution is missing initial sandbox hash: {incumbent_record.task_id}")
        if incumbent_record.initial_tree_hash != candidate_record.initial_tree_hash:
            raise RuntimeError(f"paired initial sandbox mismatch: {incumbent_record.task_id}")
        contract = pairing_contract_hash(
            task_by_id[incumbent_record.task_id],
            settings,
            seed=seed,
            phase=phase,
            role=role,
        )
        pairs.append(
            PairedExecutionRecord(
                task_id=incumbent_record.task_id,
                task_hash=incumbent_record.task_hash,
                seed=seed,
                incumbent_policy_hash=incumbent.policy_hash,
                candidate_policy_hash=candidate.policy_hash,
                incumbent_success=incumbent_record.success,
                candidate_success=candidate_record.success,
                incumbent_execution=incumbent_record.trajectory,
                candidate_execution=candidate_record.trajectory,
                inspect_log=";".join(
                    value for value in (incumbent_record.inspect_log, candidate_record.inspect_log) if value
                )
                or None,
                incumbent_final_tree_path=incumbent_record.final_tree_path,
                candidate_final_tree_path=candidate_record.final_tree_path,
                incumbent_resource_metrics=dict(incumbent_record.resource_metrics),
                candidate_resource_metrics=dict(candidate_record.resource_metrics),
                incumbent_trace=incumbent_record.trace,
                candidate_trace=candidate_record.trace,
                incumbent_status=incumbent_record.status,
                candidate_status=candidate_record.status,
                incumbent_error=incumbent_record.error,
                candidate_error=candidate_record.error,
                incumbent_initial_tree_hash=incumbent_record.initial_tree_hash,
                candidate_initial_tree_hash=candidate_record.initial_tree_hash,
                pairing_contract_hash=contract,
            )
        )
    return pairs


def image_digest(image: str) -> str:
    """Resolve a local Docker image digest without contacting the model API."""

    import subprocess

    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{index .RepoDigests 0}}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    value = result.stdout.strip()
    if "@" not in value:
        raise RuntimeError(f"Docker image has no immutable RepoDigest: {image}")
    return value.split("@", 1)[1]


def resolve_runtime_settings(
    root: Path,
    *,
    artifact_root: Path | None = None,
    log_root: Path | None = None,
    verify_image: bool = False,
) -> RuntimeSettings:
    """Resolve the locked runtime without reading or persisting credentials.

    The resolver only reads the public protocol configuration and environment
    variable names recorded there.  Credentials are checked lazily by
    :func:`_model_kwargs` immediately before an authorized model call.
    """

    root = Path(root).resolve()
    config_path = root / "configs/v15/confirmatory.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise TypeError(f"invalid confirmatory config: {config_path}")
    foundation = config.get("foundation_model", {})
    sandbox = config.get("sandbox", {})
    if not isinstance(foundation, Mapping) or not isinstance(sandbox, Mapping):
        raise TypeError("foundation_model and sandbox must be mappings")
    model_name = str(foundation.get("identifier") or "")
    provider = str(foundation.get("provider") or "")
    api_env = str(foundation.get("api_base_env") or "ANTHROPIC_BASE_URL")
    api_base = os.getenv(api_env, "https://api.anthropic.com")
    image = str(sandbox.get("image") or "")
    digest = str(sandbox.get("image_digest") or "")
    dependency_lock = str(sandbox.get("dependency_lock") or "")
    if dependency_lock and not (root / dependency_lock).is_file():
        raise ValueError(f"dependency lock does not exist: {dependency_lock}")
    settings = RuntimeSettings(
        model_name=model_name,
        provider=provider,
        api_base=api_base,
        image=image,
        image_digest=digest,
        dependency_lock=dependency_lock,
        artifact_root=artifact_root or root / "artifacts/v15/external",
        log_root=log_root or root / "artifacts/v15/inspect-logs",
        token_limit=int(config.get("resource_limits", {}).get("token_limit", 2048)),
        wall_clock_seconds=int(config.get("resource_limits", {}).get("wall_clock_seconds", 120)),
        tool_calls=int(config.get("resource_limits", {}).get("tool_calls", 16)),
        container_cpu=int(config.get("resource_limits", {}).get("container_cpu", 1)),
        container_memory_mb=int(config.get("resource_limits", {}).get("container_memory_mb", 2048)),
        agent_step_limit=int(config.get("agent_step_limit", 8)),
        model_output_tokens=int(config.get("model_output_tokens", 512)),
    )
    if verify_image and image_digest(settings.image) != settings.image_digest:
        raise ValueError("configured sandbox image digest does not match the local image")
    return settings


def runtime_manifest(settings: RuntimeSettings) -> dict[str, Any]:
    """Return a non-secret manifest suitable for confirmatory provenance."""

    payload = asdict(settings)
    payload["artifact_root"] = str(settings.artifact_root)
    payload["log_root"] = str(settings.log_root)
    payload["model_key"] = settings.model_key
    payload["settings_sha256"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return payload


__all__ = [
    "ExecutionRecord",
    "PairedExecutionRecord",
    "RuntimeSettings",
    "_is_evaluator_path",
    "_pinned_image_reference",
    "annotate_budget_metrics",
    "budget_violation",
    "evaluate_paired_with_inspect",
    "evaluate_with_inspect",
    "execute_mini_task",
    "image_digest",
    "locked_runtime_python",
    "paired_execution_failed",
    "pairing_contract_hash",
    "resolve_runtime_settings",
    "running_under_locked_runtime",
    "runtime_manifest",
]
