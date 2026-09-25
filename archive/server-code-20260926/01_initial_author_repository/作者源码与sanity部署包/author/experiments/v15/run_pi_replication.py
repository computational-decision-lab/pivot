"""DEV/confirmatory Pi cross-scaffold replication runner.

Pi is executed in a fresh host-isolated temporary repository because its
native tool runner is not Docker-aware. The task plane is role-gated, task
and policy hashes are recorded, actor/gate evaluation is paired, and the
runner never opens assessment tasks before a terminal assessment phase.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .confirmatory_guards import (
    registered_counts,
    reject_confirmatory_overrides,
    reject_existing_confirmatory_output,
    require_registered_count,
)
from .external_operators import assert_public_operator_input
from .external_promotion import _lock_protocol_inputs
from .external_runtime import (
    RuntimeSettings,
    _protected_evaluator_files,
    _restore_evaluator_files,
    locked_runtime_python,
    resolve_runtime_settings,
)
from .external_study import registered_operators
from .operators import ProposalContext
from .pi_runtime import (
    pi_cli_path,
    pi_policy_prompt,
    pi_request_digest,
    pi_runtime_status,
    pi_source_root,
)
from .planes import TaskSpec, load_task_planes
from .protocol import (
    AgentPolicy,
    TransitionRecord,
    canonical_json,
    content_hash,
    file_hash,
    write_jsonl,
    write_table,
)


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and not ({".pi-session", ".pi-home", ".tmp"} & set(item.relative_to(root).parts))
    ):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _safe_trajectory(payload: str, secret_values: Sequence[str]) -> str:
    clean = payload
    for secret in secret_values:
        if secret:
            clean = clean.replace(secret, "<redacted>")
    return clean


def _model_name(root: Path) -> str:
    config = yaml.safe_load((Path(root) / "configs/v15/confirmatory.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise TypeError("confirmatory config must be a mapping")
    foundation = config.get("foundation_model", {})
    if not isinstance(foundation, Mapping) or not foundation.get("identifier"):
        raise ValueError("foundation model identifier is missing")
    return str(foundation["identifier"])


def _test_command(task: TaskSpec) -> str:
    return str(task.metadata.get("test_command", "python -m unittest discover -v"))


def _portable_test_command(
    task: TaskSpec,
    *,
    inside_sandbox: bool,
    python_executable: Path | None = None,
    python_root: Path | None = None,
) -> str:
    command = _test_command(task)
    parts = command.split(maxsplit=1)
    if parts and parts[0] in {"python", "python3"}:
        if inside_sandbox and python_executable is not None:
            executable_path = Path(python_executable).resolve()
            runtime_root = Path(python_root).resolve() if python_root is not None else executable_path.parent.parent
            try:
                executable = f"/runtime/python/{executable_path.relative_to(runtime_root).as_posix()}"
            except ValueError as error:
                raise ValueError("pinned Python executable must be below its runtime root") from error
        else:
            executable = "/usr/bin/python3" if inside_sandbox else str(python_executable or sys.executable)
        command = f"{executable} {parts[1]}" if len(parts) == 2 else executable
    return command


def _test_sandbox_command(
    *,
    workspace: Path,
    command: str,
    python_executable: Path | None = None,
    python_root: Path | None = None,
    python_site: Path | None = None,
) -> list[str]:
    """Run a registered test command in a fresh, network-isolated namespace."""

    args = [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--dir",
        "/runtime",
        "--dir",
        "/workspace",
        "--bind",
        str(Path(workspace).resolve()),
        "/workspace",
    ]
    if python_executable is not None:
        executable = Path(python_executable).resolve()
        runtime_root = Path(python_root).resolve() if python_root is not None else executable.parent.parent
        if not runtime_root.is_dir():
            raise FileNotFoundError(f"Python runtime root is unavailable: {runtime_root}")
        try:
            executable.relative_to(runtime_root)
        except ValueError as error:
            raise ValueError("pinned Python executable must be below its runtime root") from error
        args.extend(["--ro-bind", str(runtime_root), "/runtime/python"])
        if python_site is not None:
            site = Path(python_site).resolve()
            if not site.is_dir():
                raise FileNotFoundError(f"Python site-packages are unavailable: {site}")
            args.extend(["--ro-bind", str(site), "/runtime/python-site"])
    for source in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        if Path(source).exists():
            args.extend(["--ro-bind", source, source])
    args.extend(
        [
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",
            "--chdir",
            "/workspace",
            "--",
            "/bin/bash",
            "-lc",
            command,
        ]
    )
    return args


def _run_registered_tests(
    task: TaskSpec,
    cwd: Path,
    *,
    timeout: int = 60,
    use_bwrap: bool = False,
    python_executable: Path | None = None,
    python_root: Path | None = None,
    python_site: Path | None = None,
) -> tuple[int, str]:
    command = _portable_test_command(
        task,
        inside_sandbox=use_bwrap,
        python_executable=python_executable,
        python_root=python_root,
    )
    invocation: str | list[str] = (
        _test_sandbox_command(
            workspace=cwd,
            command=command,
            python_executable=python_executable,
            python_root=python_root,
            python_site=python_site,
        )
        if use_bwrap
        else command
    )
    test_environment = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONPATH": "/workspace" if use_bwrap else str(cwd),
        "HOME": "/tmp" if use_bwrap else str(cwd / ".test-home"),
        "TMPDIR": "/tmp" if use_bwrap else str(cwd / ".test-tmp"),
        "PI_OFFLINE": "1",
    }
    if not use_bwrap:
        Path(test_environment["HOME"]).mkdir(parents=True, exist_ok=True)
        Path(test_environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        invocation,
        cwd=cwd,
        shell=isinstance(invocation, str),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=test_environment,
    )
    return int(completed.returncode), (completed.stdout + completed.stderr)[-4000:]


def _parse_pi_stats(jsonl: str) -> tuple[int, int, str | None]:
    model_calls = 0
    tool_calls = 0
    error: str | None = None
    for line in jsonl.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "message_end" and event.get("message", {}).get("role") == "assistant":
            model_calls += 1
        if event.get("type") in {"tool_execution_start", "tool_execution_end"}:
            tool_calls += int(event.get("type") == "tool_execution_start")
        if event.get("type") == "agent_end":
            messages = event.get("messages", [])
            if isinstance(messages, list):
                for message in messages:
                    if isinstance(message, Mapping) and message.get("role") == "assistant":
                        model_calls = max(model_calls, 1)
        message = event.get("message")
        if isinstance(message, Mapping) and message.get("role") == "assistant":
            content = message.get("content", [])
            if isinstance(content, list) and any(
                isinstance(item, Mapping) and item.get("type") == "text" and "error" in str(item.get("text", "")).casefold()
                for item in content
            ):
                error = str(content)
    return model_calls, tool_calls, error


def _pi_token_count(jsonl: str) -> float:
    """Extract aggregate token usage from Pi's JSONL trace.

    Pi and its provider adapters have emitted several usage layouts over time.
    We accept the common total/input/output spellings and inspect at most one
    usage object per event, avoiding double counting when a provider mirrors
    usage both at the event and message level.
    """

    total = 0.0
    usage_keys = (
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "input",
        "output",
    )
    for line in jsonl.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        candidates: list[Mapping[str, Any]] = []
        for value in (event.get("usage"), event.get("usageDetails")):
            if isinstance(value, Mapping):
                candidates.append(value)
        message = event.get("message")
        if isinstance(message, Mapping):
            for value in (message.get("usage"), message.get("usageDetails")):
                if isinstance(value, Mapping):
                    candidates.append(value)
        if not candidates:
            continue
        usage = candidates[0]
        try:
            if usage.get("total_tokens") is not None:
                total += float(usage["total_tokens"])
                continue
            values = [float(usage[key]) for key in usage_keys[1:] if usage.get(key) is not None]
            if values:
                total += sum(values)
        except (TypeError, ValueError):
            # A malformed provider usage record is represented as NaN so the
            # shared budget contract fails closed rather than silently passing.
            return float("nan")
    return total


def pi_budget_violation(
    *,
    tool_calls: int,
    elapsed_seconds: float,
    tool_limit: int | None,
    wall_limit: int | None,
    tokens: float = 0.0,
    token_limit: int | None = None,
) -> str | None:
    """Return a fixed-budget violation reason, if a Pi rollout exceeded limits."""

    violations: list[str] = []
    if token_limit is not None:
        try:
            observed_tokens = float(tokens)
        except (TypeError, ValueError):
            observed_tokens = float("nan")
        if not math.isfinite(observed_tokens) or observed_tokens < 0:
            violations.append(f"token budget is invalid: observed={tokens!r}")
        elif observed_tokens > float(token_limit):
            violations.append(
                f"token budget exceeded: observed={int(observed_tokens)} limit={int(token_limit)}"
            )
    if tool_limit is not None and int(tool_calls) > int(tool_limit):
        violations.append(f"tool-call budget exceeded: observed={int(tool_calls)} limit={int(tool_limit)}")
    if wall_limit is not None and float(elapsed_seconds) > float(wall_limit):
        violations.append(f"wall-clock budget exceeded: observed={float(elapsed_seconds):.3f} limit={int(wall_limit)}")
    return "; ".join(violations) if violations else None


def pi_sandbox_command(
    *,
    cli: Path,
    extension: Path,
    workspace: Path,
    prompt: str,
    model: str = "",
    use_bwrap: bool,
    pi_root: Path | None = None,
    node_executable: Path | None = None,
    python_executable: Path | None = None,
    python_site: Path | None = None,
) -> list[str]:
    """Build the Pi command with explicit read-only runtime mounts.

    ``bwrap`` keeps the native Pi process usable while preventing both writes
    and accidental reads of host paths outside the task workspace and the
    pinned runtime.  Network sharing is deliberate because the Pi process
    itself performs the provider request; credentials are passed through the
    child environment and are never written to artifacts.
    """

    cli = Path(cli).resolve()
    extension = Path(extension).resolve()
    workspace = Path(workspace).resolve()
    runtime_root = Path(pi_root).resolve() if pi_root is not None else cli.parent
    try:
        cli_relative = cli.relative_to(runtime_root)
    except ValueError as error:
        raise ValueError("Pi CLI must be below the explicitly mounted Pi runtime root") from error
    node_path = (
        Path(node_executable).resolve()
        if node_executable is not None
        else Path(shutil.which("node") or "node").resolve()
    )
    base = [
        "node" if not use_bwrap else "/runtime/node/bin/node",
        str(cli) if not use_bwrap else f"/runtime/pi-root/{cli_relative.as_posix()}",
        "--no-session",
        "--no-context-files",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-extensions",
        "--no-approve",
        "--extension",
        str(extension) if not use_bwrap else "/runtime/pivot-extension.ts",
        "--provider",
        "anthropic",
    ]
    if model:
        base.extend(["--model", model])
    base.extend(["--mode", "json", "--print", prompt])
    if not use_bwrap:
        return base
    node_root = node_path.parent.parent
    try:
        node_relative = node_path.relative_to(node_root)
    except ValueError as error:
        raise ValueError("Node executable must be below its explicitly mounted runtime root") from error
    if not runtime_root.is_dir():
        raise FileNotFoundError(f"Pi runtime root is unavailable: {runtime_root}")
    if not node_root.is_dir():
        raise FileNotFoundError(f"Node runtime root is unavailable: {node_root}")
    command = [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-uts",
        "--unshare-ipc",
        "--unshare-cgroup",
        "--dir",
        "/runtime",
        "--dir",
        "/workspace",
        "--ro-bind",
        str(runtime_root),
        "/runtime/pi-root",
        "--ro-bind",
        str(node_root),
        "/runtime/node",
        "--ro-bind",
        str(extension),
        "/runtime/pivot-extension.ts",
        "--bind",
        str(workspace),
        "/workspace",
    ]
    if python_executable is not None:
        python_path = Path(python_executable).resolve()
        python_root = python_path.parent.parent
        if not python_root.is_dir():
            raise FileNotFoundError(f"Python runtime root is unavailable: {python_root}")
        try:
            python_path.relative_to(python_root)
        except ValueError as error:
            raise ValueError("pinned Python executable must be below its runtime root") from error
        command.extend(["--ro-bind", str(python_root), "/runtime/python"])
        if python_site is not None:
            site = Path(python_site).resolve()
            if not site.is_dir():
                raise FileNotFoundError(f"Python site-packages are unavailable: {site}")
            command.extend(["--ro-bind", str(site), "/runtime/python-site"])
    # Only standard runtime directories are exposed.  In particular, do not
    # mount the host root, home directory, project checkout, or credentials.
    for source in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        if Path(source).exists():
            command.extend(["--ro-bind", source, source])
    command.extend(
        [
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/tmp/home",
        "--chdir",
        "/workspace",
        "--",
        *base,
        ]
    )
    # The inside path is stable even when the host's Node installation lives
    # under a user-specific directory.  The parent process supplies the
    # provider environment separately.
    command[command.index("/runtime/node/bin/node")] = f"/runtime/node/{node_relative.as_posix()}"
    return command


def _write_pi_execution_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pi_environment(
    *,
    sandbox: Path,
    session_dir: Path,
    use_bwrap: bool,
    python_root: Path | None = None,
    python_site: Path | None = None,
) -> dict[str, str]:
    """Return the minimal environment exposed to a Pi child process."""

    host_home = sandbox / ".pi-home"
    host_home.mkdir(parents=True, exist_ok=True)
    host_tmp = sandbox / ".tmp"
    host_tmp.mkdir(parents=True, exist_ok=True)
    provider_keys = ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")
    environment = {key: os.environ[key] for key in provider_keys if os.environ.get(key)}
    node_path = Path(shutil.which("node") or "node").resolve()
    environment.update(
        {
            "PATH": (
                "/runtime/node/bin:/runtime/python/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                if use_bwrap and python_root is not None
                else "/runtime/node/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                if use_bwrap
                else f"{(Path(python_root) / 'bin').resolve() if python_root is not None else ''}:{node_path.parent}:{os.environ.get('PATH', '')}"
            ),
            "HOME": "/tmp/home" if use_bwrap else str(host_home),
            "TMPDIR": "/tmp" if use_bwrap else str(host_tmp),
            "XDG_CONFIG_HOME": "/tmp/home/.config" if use_bwrap else str(host_home / ".config"),
            "XDG_CACHE_HOME": "/tmp/home/.cache" if use_bwrap else str(host_home / ".cache"),
            "PI_CODING_AGENT_DIR": "/workspace/.pi-session" if use_bwrap else str(session_dir),
            "PI_OFFLINE": "1",
        }
    )
    if python_site is not None:
        environment["PYTHONPATH"] = "/runtime/python-site" if use_bwrap else str(Path(python_site).resolve())
    return environment


def evaluate_pi_with_inspect(
    root: Path,
    tasks: Sequence[TaskSpec],
    policy: AgentPolicy,
    settings: RuntimeSettings,
    *,
    seed: int,
    phase: str,
    role: str,
    run_id: str,
    output: Path,
    agent_steps: int | None,
    tool_limit: int | None,
    wall_limit: int | None,
    token_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Run a Pi batch through Inspect's task, scorer, and trace control plane."""

    if not tasks:
        return []
    from inspect_ai import Task, eval
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import Score, mean, scorer
    from inspect_ai.solver import solver

    task_map = {task.task_id: task for task in tasks}
    artifact_dir = Path(output) / "pi-executions"
    log_dir = Path(output) / "inspect-logs"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    @solver
    def run_pi() -> Any:
        async def solve(state: Any, generate: Any) -> Any:
            task = task_map[str(state.sample_id)]
            try:
                record = await asyncio.to_thread(
                    _execute_pi,
                    root,
                    task,
                    policy,
                    seed=seed,
                    output=artifact_dir,
                    agent_steps=agent_steps,
                    tool_limit=tool_limit,
                    wall_limit=wall_limit,
                    token_limit=token_limit,
                )
            except (FileNotFoundError, OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
                record = _pi_error_record(task, policy, seed, error)
            _write_pi_execution_record(
                artifact_dir / f"{task.task_id}__{policy.policy_hash[:12]}__{seed}.inspect.json",
                record,
            )
            state.store.set("v15_pi_execution", record)
            return state

        return solve

    @scorer(metrics=[mean()])
    def score_pi() -> Any:
        async def score(state: Any, target: Any) -> Score:
            record = state.store.get("v15_pi_execution", {})
            return Score(
                value=float(record.get("success", 0.0)),
                answer=str(record.get("test_returncode", -1)),
                explanation=str(record.get("status", "")),
                metadata={
                    "task_id": str(record.get("task_id", state.sample_id)),
                    "trajectory": record.get("trajectory"),
                },
            )

        return score

    samples = [
        Sample(
            input=str(task.metadata.get("instruction", "Repair the repository task.")),
            id=task.task_id,
            target="1",
            metadata={"task_id": task.task_id, "task_hash": task.task_hash, "role": role},
        )
        for task in tasks
    ]
    logs = eval(
        Task(
            dataset=samples,
            solver=run_pi(),
            scorer=score_pi(),
            model="mockllm/model",
            name=f"v15_pi_{phase}_{run_id}_{seed}",
            metadata={"v15_role": role, "policy_hash": policy.policy_hash, "scaffold": "Pi"},
        ),
        model="mockllm/model",
        display="none",
        log_realtime=False,
        log_dir=str(log_dir),
        max_samples=settings.max_parallel,
        fail_on_error=False,
        score_on_error=True,
    )
    by_id: dict[str, dict[str, Any]] = {}
    for log in logs:
        location = str(getattr(log, "location", ""))
        for sample in getattr(log, "samples", []):
            sample_id = str(getattr(sample, "id", ""))
            record_path = artifact_dir / f"{sample_id}__{policy.policy_hash[:12]}__{seed}.inspect.json"
            if not record_path.is_file():
                continue
            payload = json.loads(record_path.read_text(encoding="utf-8"))
            payload["inspect_log"] = location
            by_id[sample_id] = payload
    missing = [task.task_id for task in tasks if task.task_id not in by_id]
    if missing:
        raise RuntimeError(f"Inspect did not return Pi execution records for: {missing}")
    return [by_id[task.task_id] for task in tasks]


def _execute_pi(
    root: Path,
    task: TaskSpec,
    policy: AgentPolicy,
    *,
    seed: int,
    output: Path,
    agent_steps: int | None,
    tool_limit: int | None = None,
    wall_limit: int | None = None,
    token_limit: int | None = None,
) -> dict[str, Any]:
    cli = pi_cli_path(root)
    if not cli.is_file():
        raise FileNotFoundError(f"Pi CLI is not built: {cli}")
    if not os.getenv("ANTHROPIC_AUTH_TOKEN") and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("an authorized Anthropic credential is required for Pi DEV execution")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"pivot-pi-{task.task_id}-", dir=output) as sandbox_name:
        sandbox = Path(sandbox_name)
        for relative, content in task.files.items():
            destination = sandbox / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
        initial_hash = _tree_hash(sandbox)
        evaluator_snapshot = _protected_evaluator_files(task)
        session_dir = sandbox / ".pi-session"
        session_dir.mkdir()
        prompt = f"{pi_policy_prompt(policy)}\n\nTask: {task.metadata.get('instruction', 'Repair the repository task.')}"
        if agent_steps is not None:
            prompt += f"\nWork within at most {int(agent_steps)} tool steps."
        use_bwrap = shutil.which("bwrap") is not None
        pinned_python = locked_runtime_python(root)
        pinned_python_base = pinned_python.resolve()
        pinned_python_root = pinned_python_base.parent.parent
        pinned_python_site = root / ".tools/v15/runtime/lib/python3.11/site-packages"
        allowed_environment = _pi_environment(
            sandbox=sandbox,
            session_dir=session_dir,
            use_bwrap=use_bwrap,
            python_root=pinned_python_root,
            python_site=pinned_python_site,
        )
        command = pi_sandbox_command(
            cli=cli,
            extension=Path(root).resolve() / "experiments/v15/pi_gateway_extension.ts",
            workspace=sandbox,
            prompt=prompt,
            model=_model_name(root),
            use_bwrap=use_bwrap,
            pi_root=pi_source_root(root),
            node_executable=Path(shutil.which("node") or "node").resolve(),
            python_executable=pinned_python,
            python_site=pinned_python_site,
        )
        completed = subprocess.run(
            command,
            cwd=sandbox,
            capture_output=True,
            text=True,
            timeout=max(1, int(wall_limit or 180) + 30),
            check=False,
            env=allowed_environment,
        )
        secret_values = tuple(value for value in (os.getenv("ANTHROPIC_AUTH_TOKEN"), os.getenv("ANTHROPIC_API_KEY")) if value)
        trajectory = _safe_trajectory(completed.stdout, secret_values)
        trajectory_path = output / f"{task.task_id}__{policy.policy_hash[:12]}__{seed}.jsonl"
        trajectory_path.write_text(trajectory, encoding="utf-8")
        evaluator_tamper_paths = _restore_evaluator_files(sandbox, evaluator_snapshot)
        test_returncode, test_output = _run_registered_tests(
            task,
            sandbox,
            use_bwrap=use_bwrap,
            python_executable=pinned_python,
            python_root=pinned_python_root,
            python_site=pinned_python_site,
        )
        final_hash = _tree_hash(sandbox)
        final_tree_path = output / f"{task.task_id}__{policy.policy_hash[:12]}__{seed}.final_tree"
        if final_tree_path.exists():
            shutil.rmtree(final_tree_path)
        shutil.copytree(sandbox, final_tree_path, ignore=shutil.ignore_patterns(".pi-session", ".pi-home", ".tmp"))
        model_calls, tool_calls, parsed_error = _parse_pi_stats(trajectory)
        tokens = _pi_token_count(trajectory)
        error = parsed_error
        if evaluator_tamper_paths:
            error = "agent modified protected evaluator files: " + ", ".join(evaluator_tamper_paths)
        if completed.returncode != 0:
            error = error or f"Pi exited with return code {completed.returncode}"
        if test_returncode != 0:
            error = error or "registered test command failed"
        elapsed = time.monotonic() - started
        budget_error = pi_budget_violation(
            tool_calls=tool_calls,
            elapsed_seconds=elapsed,
            tool_limit=tool_limit,
            wall_limit=wall_limit,
            tokens=tokens,
            token_limit=token_limit,
        )
        error = error or budget_error
        within_budget = budget_error is None
        result = {
            "status": "COMPLETED" if completed.returncode == 0 and test_returncode == 0 and error is None else "IMPLEMENTATION_FAILURE",
            "terminal_state": None if completed.returncode == 0 and test_returncode == 0 and error is None else "IMPLEMENTATION_FAILURE",
            "task_id": task.task_id,
            "task_hash": task.task_hash,
            "policy_hash": policy.policy_hash,
            "seed": int(seed),
            "request_digest": pi_request_digest(task.task_id, policy, seed),
            "pairing_conditions_hash": content_hash(
                {
                    "task_id": task.task_id,
                    "task_hash": task.task_hash,
                    "seed": int(seed),
                    "model": _model_name(root),
                    "tool_limit": tool_limit,
                    "wall_limit": wall_limit,
                    "token_limit": token_limit,
                    "sandbox_image": pi_runtime_status(root).get("sandbox_image"),
                }
            ),
            "success": float(test_returncode == 0 and within_budget),
            "trajectory": str(trajectory_path),
            "final_tree_path": str(final_tree_path),
            "initial_tree_hash": initial_hash,
            "final_tree_hash": final_hash,
            "test_returncode": test_returncode,
            "pi_returncode": int(completed.returncode),
            "model_calls": model_calls,
            "tool_calls": tool_calls,
            "tokens": tokens,
            "wall_clock_seconds": elapsed,
            "resource_metrics": {
                "model_calls": float(model_calls),
                "tool_calls": float(tool_calls),
                "tokens": float(tokens),
                "wall_clock_seconds": elapsed,
                "evaluator_tamper_detected": float(bool(evaluator_tamper_paths)),
                "protected_evaluator_file_count": float(len(evaluator_snapshot)),
                "budget_token_limit": float(token_limit or 0),
                "budget_tool_call_limit": float(tool_limit or 0),
                "budget_wall_clock_limit": float(wall_limit or 0),
                "budget_within_limits": float(within_budget),
                "budget_violation_count": float(bool(budget_error)),
            },
            "budget_violation": budget_error,
            "evaluator_tamper_paths": list(evaluator_tamper_paths),
            "sandbox": {
                "mode": "bubblewrap_explicit_mounts" if use_bwrap else "host_dev_fallback",
                "host_root_exposed": False,
                "workspace_mount": "/workspace" if use_bwrap else str(sandbox),
                "network": "provider_gateway_shared" if use_bwrap else "host_network",
            },
            "test_output_digest": hashlib.sha256(test_output.encode("utf-8")).hexdigest(),
            "error": error,
        }
        _write_pi_execution_record(
            output / f"{task.task_id}__{policy.policy_hash[:12]}__{seed}.execution.json",
            result,
        )
        return result


def _manifest(output: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["manifest_sha256"] = hashlib.sha256(canonical_json(result).encode("utf-8")).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def build_pi_confirmatory_plan(
    config: Mapping[str, Any], primary_manifest: Mapping[str, Any]
) -> dict[str, int]:
    """Validate the primary archive shape and return the frozen Pi run plan.

    The cross-scaffold replication is intentionally gated by the completed
    primary archive.  This helper performs only structural checks, so it is
    safe to call before opening the confirmatory lock or making a model call.
    """

    if str(primary_manifest.get("status", "")).upper() != "COMPLETED":
        raise ValueError("primary archive must have COMPLETED status")
    counts = registered_counts(config, operator_count=2)
    expected_transitions = counts["trajectories"] * counts["rounds"] * counts["candidates"]
    require_registered_count(
        True, int(primary_manifest.get("trajectory_count", 0)), counts["trajectories"], "primary trajectory"
    )
    require_registered_count(
        True, int(primary_manifest.get("round_count", 0)), counts["rounds"], "primary round"
    )
    require_registered_count(
        True, int(primary_manifest.get("candidate_count", 0)), expected_transitions, "primary candidate"
    )
    return {
        "operator_count": 2,
        "trajectory_count": counts["trajectories"],
        "round_count": counts["rounds"],
        "candidates_per_round": counts["candidates"],
        "transition_count": expected_transitions,
    }


def pair_pi_execution_rows(
    incumbent: Mapping[str, Any], candidate: Mapping[str, Any], *, seed: int
) -> dict[str, Any]:
    """Create one paired deployment row from two Pi executions."""

    if str(incumbent.get("task_id")) != str(candidate.get("task_id")):
        raise ValueError("paired Pi executions must have the same task_id")
    if str(incumbent.get("task_hash")) != str(candidate.get("task_hash")):
        raise ValueError("paired Pi executions must have the same task_hash")
    incumbent_seed = incumbent.get("seed")
    candidate_seed = candidate.get("seed")
    if (incumbent_seed is None) != (candidate_seed is None):
        raise ValueError("paired Pi executions must both record their seed")
    if incumbent_seed is not None and candidate_seed is not None and int(incumbent_seed) != int(candidate_seed):
        raise ValueError("paired Pi executions must have the same seed")
    if incumbent_seed is not None and int(incumbent_seed) != int(seed):
        raise ValueError("paired Pi execution seed does not match requested seed")
    initial_left = incumbent.get("initial_tree_hash")
    initial_right = candidate.get("initial_tree_hash")
    if (initial_left is None) != (initial_right is None):
        raise ValueError("paired Pi executions must both record their initial sandbox hash")
    if initial_left is not None and initial_right is not None and str(initial_left) != str(initial_right):
        raise ValueError("paired Pi executions must have the same initial sandbox hash")
    contract_left = incumbent.get("pairing_conditions_hash")
    contract_right = candidate.get("pairing_conditions_hash")
    if contract_left is not None and contract_right is not None and str(contract_left) != str(contract_right):
        raise ValueError("paired Pi executions must share the same pairing conditions")
    incumbent_success = float(incumbent.get("success", 0.0))
    candidate_success = float(candidate.get("success", 0.0))
    return {
        "task_id": str(incumbent["task_id"]),
        "task_hash": str(incumbent["task_hash"]),
        "seed": int(seed),
        "incumbent_success": incumbent_success,
        "candidate_success": candidate_success,
        "delta_actor": candidate_success - incumbent_success,
        "incumbent_execution": incumbent.get("trajectory"),
        "candidate_execution": candidate.get("trajectory"),
        "incumbent_final_tree_path": incumbent.get("final_tree_path"),
        "candidate_final_tree_path": candidate.get("final_tree_path"),
        "incumbent_initial_tree_hash": initial_left,
        "candidate_initial_tree_hash": initial_right,
        "pairing_conditions_hash": contract_left or contract_right,
        "incumbent_resource_metrics": dict(incumbent.get("resource_metrics", {})),
        "candidate_resource_metrics": dict(candidate.get("resource_metrics", {})),
        "incumbent_trace": list(incumbent.get("trace", [])),
        "candidate_trace": list(candidate.get("trace", [])),
        "status": "COMPLETED"
        if incumbent.get("status") == candidate.get("status") == "COMPLETED"
        else "IMPLEMENTATION_FAILURE",
        "incumbent_status": incumbent.get("status"),
        "candidate_status": candidate.get("status"),
        "incumbent_error": incumbent.get("error"),
        "candidate_error": candidate.get("error"),
    }


def _pi_mean_success(records: Sequence[Mapping[str, Any]]) -> float:
    return sum(float(record.get("success", 0.0)) for record in records) / max(len(records), 1)


def _pi_family_success(
    tasks: Sequence[TaskSpec], records: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    family_by_id = {task.task_id: task.family for task in tasks}
    grouped: dict[str, list[float]] = {}
    for record in records:
        family = family_by_id.get(str(record.get("task_id")))
        if family is not None:
            grouped.setdefault(family, []).append(float(record.get("success", 0.0)))
    return {name: sum(values) / len(values) for name, values in sorted(grouped.items()) if values}


def _pi_family_delta(
    tasks: Sequence[TaskSpec], pairs: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    family_by_id = {task.task_id: task.family for task in tasks}
    grouped: dict[str, list[float]] = {}
    for pair in pairs:
        family = family_by_id.get(str(pair.get("task_id")))
        if family is not None:
            grouped.setdefault(family, []).append(float(pair.get("delta_actor", 0.0)))
    return {name: sum(values) / len(values) for name, values in sorted(grouped.items()) if values}


def _pi_feedback(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [str(record.get("task_id")) for record in records if float(record.get("success", 0.0)) < 0.5]
    return {
        "failed_tests": len(failed),
        "failed_task_ids": failed,
        "mean_success": _pi_mean_success(records),
        "resource_metrics": _pi_resource_metrics(records),
    }


def _pi_resource_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys = (
        "tokens",
        "tool_calls",
        "tests_executed",
        "files_read",
        "files_written",
        "context_peak",
        "wall_clock_seconds",
        "dependency_operations",
        "model_calls",
    )
    return {
        key: sum(float(dict(record.get("resource_metrics", {})).get(key, record.get(key, 0.0))) for record in records)
        / max(len(records), 1)
        for key in keys
    }


def _pi_model_calls(records: Sequence[Mapping[str, Any]]) -> int:
    return int(
        sum(
            float(dict(record.get("resource_metrics", {})).get("model_calls", record.get("model_calls", 0.0)))
            for record in records
        )
    )


def _pi_footprint(
    incumbent: AgentPolicy,
    candidate: AgentPolicy,
    incumbent_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Combine pre-deployment policy and paired behavioral footprint fields."""

    footprint = dict(incumbent.diff(candidate))
    left = _pi_resource_metrics(incumbent_records)
    right = _pi_resource_metrics(candidate_records)
    for name, key in (
        ("tool_call_distribution_shift", "tool_calls"),
        ("test_execution_shift", "tests_executed"),
        ("files_read_shift", "files_read"),
        ("files_written_shift", "files_written"),
        ("dependency_operation_shift", "dependency_operations"),
        ("token_usage_shift", "tokens"),
        ("context_peak_shift", "context_peak"),
        ("wall_clock_shift", "wall_clock_seconds"),
    ):
        footprint[name] = right[key] - left[key]
    footprint["shell_command_distribution_shift"] = footprint["tool_call_distribution_shift"]
    footprint["action_sequence_distance"] = float(
        any(record.get("trace", []) != other.get("trace", []) for record, other in zip(incumbent_records, candidate_records))
    )
    return footprint


def _pi_error_record(task: TaskSpec, policy: AgentPolicy, seed: int, error: Exception) -> dict[str, Any]:
    return {
        "status": "IMPLEMENTATION_FAILURE",
        "terminal_state": "IMPLEMENTATION_FAILURE",
        "task_id": task.task_id,
        "task_hash": task.task_hash,
        "policy_hash": policy.policy_hash,
        "seed": int(seed),
        "success": 0.0,
        "resource_metrics": {"model_calls": 0.0},
        "trace": [],
        "error": f"{type(error).__name__}: {error}",
    }


def _evaluate_pi_tasks(
    root: Path,
    tasks: Sequence[TaskSpec],
    policy: AgentPolicy,
    *,
    seed: int,
    output: Path,
    agent_steps: int | None,
    tool_limit: int | None,
    wall_limit: int | None,
    token_limit: int | None = None,
    settings: RuntimeSettings | None = None,
    phase: str = "pi",
    role: str = "audit",
    run_id: str = "pi",
) -> tuple[list[dict[str, Any]], list[str]]:
    if settings is not None:
        if token_limit is None:
            token_limit = settings.token_limit
        try:
            inspected_records = evaluate_pi_with_inspect(
                root,
                tasks,
                policy,
                settings,
                seed=seed,
                phase=phase,
                role=role,
                run_id=run_id,
                output=output,
                agent_steps=agent_steps,
                tool_limit=tool_limit,
                wall_limit=wall_limit,
                token_limit=token_limit,
            )
        except (
            FileNotFoundError,
            KeyError,
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            TypeError,
            ValueError,
        ) as error:
            return [], [f"Inspect Pi batch failed: {type(error).__name__}: {error}"]
        inspected_failures = [
            f"{record.get('task_id')}: {record.get('error')}"
            for record in inspected_records
            if record.get("status") != "COMPLETED"
        ]
        return inspected_records, inspected_failures
    direct_records: list[dict[str, Any]] = []
    direct_failures: list[str] = []
    for task in tasks:
        try:
            record = _execute_pi(
                root,
                task,
                policy,
                seed=seed,
                output=output,
                agent_steps=agent_steps,
                tool_limit=tool_limit,
                wall_limit=wall_limit,
                token_limit=token_limit,
            )
        except (
            FileNotFoundError,
            KeyError,
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            TypeError,
            ValueError,
        ) as error:
            record = _pi_error_record(task, policy, seed, error)
            direct_failures.append(str(record["error"]))
        direct_records.append(record)
    return direct_records, direct_failures


def _evaluate_pi_pairs(
    root: Path,
    tasks: Sequence[TaskSpec],
    incumbent: AgentPolicy,
    candidate: AgentPolicy,
    *,
    seed: int,
    output: Path,
    agent_steps: int | None,
    tool_limit: int | None,
    wall_limit: int | None,
    token_limit: int | None = None,
    settings: RuntimeSettings | None = None,
    phase: str = "pi_actor",
    role: str = "promotion",
    run_id: str = "pi-pair",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    if settings is not None and token_limit is None:
        token_limit = settings.token_limit
    incumbent_records, incumbent_failures = _evaluate_pi_tasks(
        root,
        tasks,
        incumbent,
        seed=seed,
        output=output / "incumbent",
        agent_steps=agent_steps,
        tool_limit=tool_limit,
        wall_limit=wall_limit,
        token_limit=token_limit,
        settings=settings,
        phase=f"{phase}/incumbent",
        role=role,
        run_id=f"{run_id}/incumbent",
    )
    candidate_records, candidate_failures = _evaluate_pi_tasks(
        root,
        tasks,
        candidate,
        seed=seed,
        output=output / "candidate",
        agent_steps=agent_steps,
        tool_limit=tool_limit,
        wall_limit=wall_limit,
        token_limit=token_limit,
        settings=settings,
        phase=f"{phase}/candidate",
        role=role,
        run_id=f"{run_id}/candidate",
    )
    candidate_by_id = {str(record["task_id"]): record for record in candidate_records}
    pairs: list[dict[str, Any]] = []
    failures = [*incumbent_failures, *candidate_failures]
    for incumbent_record in incumbent_records:
        candidate_record = candidate_by_id.get(str(incumbent_record["task_id"]))
        if candidate_record is None:
            failures.append(f"missing paired Pi record: {incumbent_record['task_id']}")
            continue
        pairs.append(pair_pi_execution_rows(incumbent_record, candidate_record, seed=seed))
    return incumbent_records, candidate_records, failures + [
        f"paired execution failed: {pair['task_id']}"
        for pair in pairs
        if pair["status"] != "COMPLETED"
    ]


def _run_registered_pi_audit(
    root: Path,
    *,
    config: Mapping[str, Any],
    planes: Any,
    settings: RuntimeSettings,
    output: Path,
    trajectories: int,
    rounds: int,
    candidates_count: int,
    confirmatory: bool,
    agent_steps: int | None,
    lock: Mapping[str, Any] | None,
    task_limit: int | None = None,
) -> dict[str, Any]:
    """Run the Pi transition audit with the same operator/archive contract.

    Proposal calls receive only proxy feedback.  Gate outcomes are consumed by
    the paired evaluator and are persisted only in transition rows; they never
    enter the next proposal context.  Assessment is never requested here.
    """

    proxy_tasks = planes.tasks("proxy", role="operator")
    gate_tasks = planes.tasks("gate", role="promotion")
    # Only task identifiers/hashes are used by the audit guard.  The complete
    # hidden task contents and all deployment outcomes remain inside their
    # sealed planes and are never serialized into a proposal context.
    hidden_task_descriptors = tuple(
        descriptor
        for plane_name in ("gate", "assessment")
        for descriptor in planes.manifest().get(plane_name, [])
    )
    if not confirmatory:
        bounded = max(1, int(task_limit or 1))
        proxy_tasks = proxy_tasks[:bounded]
        gate_tasks = gate_tasks[:bounded]
    operators = registered_operators(settings)
    seed_blocks = [
        int(value)
        for value in config.get("seed_registry", {}).get("seed_blocks", [10001, 10101])
    ]
    if not seed_blocks:
        raise ValueError("Pi seed registry must contain at least one seed block")
    transitions: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    failures: list[str] = []
    proposal_calls = 0
    execution_model_calls = 0
    execution_count = 0
    completed_execution_count = 0
    operator_input_checks = 0
    for operator_index, operator in enumerate(operators):
        for trajectory_index in range(trajectories):
            seed = seed_blocks[operator_index % len(seed_blocks)] + trajectory_index
            incumbent = AgentPolicy.minimal().with_updates(
                metadata={
                    "scaffold": "Pi",
                    "operator": operator.name,
                    "trajectory": str(trajectory_index),
                    "phase": "CONFIRMATORY_PI" if confirmatory else "DEV_PI",
                }
            )
            run_id = f"pi-{operator.name}-trajectory-{trajectory_index:03d}"
            trajectory_failed = False
            for round_index in range(rounds):
                proxy_records, proxy_failures = _evaluate_pi_tasks(
                    root,
                    proxy_tasks,
                    incumbent,
                    seed=seed + round_index,
                    output=output / "artifacts" / run_id / f"round-{round_index}" / "proxy-incumbent",
                    agent_steps=agent_steps,
                    tool_limit=settings.tool_calls,
                    wall_limit=settings.wall_clock_seconds,
                    settings=settings,
                    phase="pi_proxy",
                    role="proxy_evaluator",
                    run_id=f"{run_id}-round-{round_index}-incumbent",
                )
                execution_count += len(proxy_records)
                completed_execution_count += sum(record.get("status") == "COMPLETED" for record in proxy_records)
                execution_model_calls += _pi_model_calls(proxy_records)
                failures.extend(proxy_failures)
                if proxy_failures:
                    trajectory_failed = True
                    break
                context = ProposalContext(
                    proxy_score=_pi_mean_success(proxy_records),
                    proxy_feedback=_pi_feedback(proxy_records),
                    round_index=round_index,
                    seed=seed,
                    resource_budget=settings.tool_calls,
                )
                try:
                    assert_public_operator_input(incumbent, context, hidden_task_descriptors)
                    operator_input_checks += 1
                    proposed = operator.propose(incumbent, context, count=candidates_count)
                except (RuntimeError, ValueError, TypeError) as error:
                    failures.append(f"{run_id} round {round_index} proposal: {type(error).__name__}: {error}")
                    trajectory_failed = True
                    break
                proposal_calls += 1
                if len(proposed) != candidates_count:
                    failures.append(
                        f"{run_id} round {round_index} proposal count={len(proposed)} expected={candidates_count}"
                    )
                    trajectory_failed = True
                    break
                incumbent_score = _pi_mean_success(proxy_records)
                proxy_scores: list[tuple[AgentPolicy, float]] = []
                for candidate_index, candidate in enumerate(proposed):
                    candidate_records, candidate_failures = _evaluate_pi_tasks(
                        root,
                        proxy_tasks,
                        candidate,
                        seed=seed + round_index,
                        output=output / "artifacts" / run_id / f"round-{round_index}" / f"candidate-{candidate_index}" / "proxy",
                        agent_steps=agent_steps,
                        tool_limit=settings.tool_calls,
                        wall_limit=settings.wall_clock_seconds,
                        settings=settings,
                        phase="pi_proxy",
                        role="proxy_evaluator",
                        run_id=f"{run_id}-round-{round_index}-candidate-{candidate_index}",
                    )
                    execution_count += len(candidate_records)
                    completed_execution_count += sum(record.get("status") == "COMPLETED" for record in candidate_records)
                    execution_model_calls += _pi_model_calls(candidate_records)
                    failures.extend(candidate_failures)
                    if candidate_failures:
                        trajectory_failed = True
                        break
                    candidate_score = _pi_mean_success(candidate_records)
                    proxy_scores.append((candidate, candidate_score))
                    incumbent_actor, candidate_actor, pair_failures = _evaluate_pi_pairs(
                        root,
                        gate_tasks,
                        incumbent,
                        candidate,
                        seed=seed + round_index,
                        output=output / "artifacts" / run_id / f"round-{round_index}" / f"candidate-{candidate_index}" / "actor",
                        agent_steps=agent_steps,
                        tool_limit=settings.tool_calls,
                        wall_limit=settings.wall_clock_seconds,
                        settings=settings,
                        phase="pi_actor",
                        role="promotion",
                        run_id=f"{run_id}-round-{round_index}-candidate-{candidate_index}",
                    )
                    execution_count += len(incumbent_actor) + len(candidate_actor)
                    completed_execution_count += sum(
                        record.get("status") == "COMPLETED" for record in (*incumbent_actor, *candidate_actor)
                    )
                    execution_model_calls += _pi_model_calls(incumbent_actor) + _pi_model_calls(candidate_actor)
                    failures.extend(pair_failures)
                    if pair_failures or len(incumbent_actor) != len(gate_tasks) or len(candidate_actor) != len(gate_tasks):
                        failures.append(
                            f"{run_id} round {round_index} candidate {candidate_index} has incomplete paired actor evidence"
                        )
                        trajectory_failed = True
                        break
                    paired_rows = [
                        pair_pi_execution_rows(left, right, seed=seed + round_index)
                        for left, right in zip(incumbent_actor, candidate_actor)
                    ]
                    paired_deltas = [float(pair["delta_actor"]) for pair in paired_rows]
                    actor_delta = sum(paired_deltas) / max(len(paired_deltas), 1)
                    footprint = _pi_footprint(incumbent, candidate, incumbent_actor, candidate_actor)
                    transition = TransitionRecord(
                        run_id=run_id,
                        scaffold="Pi",
                        operator=operator.name,
                        task_family="mixed",
                        round_index=round_index,
                        candidate_index=candidate_index,
                        incumbent=incumbent,
                        candidate=candidate,
                        delta_proxy=candidate_score - incumbent_score,
                        delta_actor=actor_delta,
                        proxy_incumbent_score=incumbent_score,
                        proxy_candidate_score=candidate_score,
                        actor_incumbent_score=_pi_mean_success(incumbent_actor),
                        actor_candidate_score=_pi_mean_success(candidate_actor),
                        footprint=footprint,
                        resource_metrics={
                            "proxy_incumbent": _pi_resource_metrics(proxy_records),
                            "proxy_candidate": _pi_resource_metrics(candidate_records),
                            "paired_tasks": len(paired_deltas),
                            "pairing_conditions_hashes": [
                                pair.get("pairing_conditions_hash") for pair in paired_rows
                            ],
                            "task_families": sorted({task.family for task in (*proxy_tasks, *gate_tasks)}),
                            "proxy_candidate_by_family": _pi_family_success(gate_tasks, candidate_records),
                            "proxy_incumbent_by_family": _pi_family_success(gate_tasks, proxy_records),
                            "actor_delta_by_family": _pi_family_delta(gate_tasks, paired_rows),
                            "scaffold": "Pi",
                        },
                        seed=seed + round_index,
                        paired_seed_ids=(seed + round_index,),
                        source_digest=content_hash(candidate.to_record()),
                        config_hash=file_hash(root / "configs/v15/confirmatory.yaml"),
                    )
                    transitions.append(transition.to_record())
                    candidates.append(
                        {
                            "run_id": run_id,
                            "round": round_index,
                            "candidate_index": candidate_index,
                            "candidate_id": transition.transition_id,
                            "candidate_hash": candidate.policy_hash,
                            "incumbent_hash": incumbent.policy_hash,
                            "proxy_delta": transition.delta_proxy,
                            "operator": operator.name,
                            "scaffold": "Pi",
                            "task_family": "mixed",
                            "candidate_policy": candidate.to_record(),
                            "incumbent_policy": incumbent.to_record(),
                            "footprint": dict(transition.footprint),
                            "seed": seed + round_index,
                        }
                    )
                if trajectory_failed:
                    break
                if proxy_scores:
                    incumbent = max(proxy_scores, key=lambda item: item[1])[0]
            if trajectory_failed:
                continue
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(transitions, output / "autonomous_transitions.jsonl")
    write_jsonl(candidates, output / "promotion_candidates.jsonl")
    write_table(
        transitions,
        output / "autonomous_transitions",
        columns=(
            "transition_id",
            "run_id",
            "scaffold",
            "operator",
            "task_family",
            "round",
            "candidate_index",
            "delta_proxy",
            "delta_actor",
            "proxy_incumbent_score",
            "proxy_candidate_score",
            "actor_incumbent_score",
            "actor_candidate_score",
            "actor_reversal",
            "footprint",
            "resource_metrics",
            "config_hash",
        ),
    )
    write_table(
        candidates,
        output / "promotion_candidates",
        columns=(
            "run_id",
            "round",
            "candidate_index",
            "candidate_id",
            "candidate_hash",
            "incumbent_hash",
            "proxy_delta",
            "operator",
            "scaffold",
            "task_family",
            "seed",
            "incumbent_policy",
            "candidate_policy",
            "footprint",
        ),
    )
    expected_transitions = trajectories * len(operators) * rounds * candidates_count
    status = "COMPLETED" if not failures and len(transitions) == expected_transitions else "IMPLEMENTATION_FAILURE"
    return _manifest(
        output,
        {
            "schema_version": "pivot-v15-pi-replication-2",
            "phase": "CONFIRMATORY" if confirmatory else "DEV",
            "confirmatory": confirmatory,
            "status": status,
            "terminal_state": (
                "IMPLEMENTATION_FAILURE"
                if status != "COMPLETED"
                else "UNDERPOWERED"
                if not confirmatory
                else None
            ),
            "execution_attempted": True,
            "design_status": "VALIDATED_DEV" if not confirmatory and status == "COMPLETED" else "PENDING_ANALYSIS",
            "leakage_detected": False,
            "scaffold": "Pi",
            "operator_count": len(operators),
            "trajectory_count": trajectories * len(operators),
            "round_count": rounds,
            "candidates_per_round": candidates_count,
            "expected_transition_count": expected_transitions,
            "record_count": len(transitions),
            "transition_count": len(transitions),
            "candidate_count": len(candidates),
            "task_count": len(proxy_tasks) + len(gate_tasks),
            "proxy_task_count": len(proxy_tasks),
            "gate_task_count": len(gate_tasks),
            "assessment_accessed": False,
            "gate_accessed": any(
                event.get("plane") == "gate" and event.get("outcome") == "granted"
                for event in planes.access_log_snapshot()
            ),
            "operator_input_audit": {
                "checks": operator_input_checks,
                "hidden_descriptor_count": len(hidden_task_descriptors),
                "sealed_outcomes_in_input": False,
            },
            "outcome_chasing": False,
            "proposal_calls_performed": proposal_calls,
            "model_calls_performed": proposal_calls + execution_model_calls,
            "proposal_model_calls": proposal_calls,
            "execution_model_calls": execution_model_calls,
            "agent_execution_count": execution_count,
            "completed_execution_count": completed_execution_count,
            "sandbox_executions": execution_count,
            "container_executions": 0,
            "errors": failures,
            "task_manifest_sha256": file_hash(root / "configs/v15/task_manifest.json"),
            "config_hash": file_hash(root / "configs/v15/confirmatory.yaml"),
            "runtime": pi_runtime_status(root),
            "lock_hash": lock.get("lock_hash") if lock else None,
            "access_log": list(planes.access_log_snapshot()),
            "role_access_log": list(planes.access_log_snapshot()),
            "note": "Pi proposals consume proxy feedback only; actor outcomes are paired and never returned to the proposal operator; assessment remains sealed.",
        },
    )


def run_pi_replication(
    root: Path,
    *,
    confirmatory: bool = False,
    task_limit: int | None = None,
    agent_steps: int | None = None,
) -> dict[str, Any]:
    """Run the registered Pi transition audit without opening assessment data."""

    root = Path(root).resolve()
    if confirmatory:
        if os.getenv("PIVOT_V15_CONFIRMATORY_ACK") != "I_ACCEPT_FROZEN_PROTOCOL":
            raise PermissionError("confirmatory execution requires PIVOT_V15_CONFIRMATORY_ACK")
        reject_confirmatory_overrides(confirmatory, task_limit=task_limit, agent_steps=agent_steps)
        if task_limit is not None or agent_steps is not None:
            raise ValueError("confirmatory Pi execution cannot use DEV limits")
        output = root / "results/v15/external-pi-replication"
        reject_existing_confirmatory_output(output, confirmatory)
        primary_manifest = root / "results/v15/external-transition-audit/manifest.json"
        if not primary_manifest.is_file():
            raise ValueError("Primary mini-SWE transition archive must be frozen before cross-scaffold confirmation")
        primary = json.loads(primary_manifest.read_text(encoding="utf-8"))
        config = yaml.safe_load((root / "configs/v15/confirmatory.yaml").read_text(encoding="utf-8"))
        if not isinstance(config, Mapping) or not isinstance(primary, Mapping):
            raise TypeError("confirmatory Pi inputs must be mappings")
        build_pi_confirmatory_plan(config, primary)
        if not pi_cli_path(root).is_file():
            raise FileNotFoundError(f"Pi CLI is not built: {pi_cli_path(root)}")
        if shutil.which("node") is None:
            raise RuntimeError("Node.js is required for confirmatory Pi execution")
        if shutil.which("bwrap") is None:
            raise RuntimeError("bubblewrap is required for filesystem-isolated confirmatory Pi execution")
        if not (root / "experiments/v15/pi_gateway_extension.ts").is_file():
            raise FileNotFoundError("Pi gateway extension is missing")
        if not (os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY")):
            raise RuntimeError("an authorized Anthropic credential is required for confirmatory Pi execution")
        # Resolve every task and runtime input before opening the lock.  The
        # opening event is the irreversible boundary before the first model call.
        lock = _lock_protocol_inputs(root)
        planes = load_task_planes(root / "configs/v15/task_manifest.json")
        settings = resolve_runtime_settings(
            root,
            artifact_root=output / "artifacts",
            log_root=output / "inspect-logs",
            verify_image=False,
        )
        counts = build_pi_confirmatory_plan(config, primary)
        lock = _lock_protocol_inputs(root, phase="pi_replication")
        return _run_registered_pi_audit(
            root,
            config=config,
            planes=planes,
            settings=settings,
            output=output,
            trajectories=counts["trajectory_count"] // counts["operator_count"],
            rounds=counts["round_count"],
            candidates_count=counts["candidates_per_round"],
            confirmatory=True,
            agent_steps=None,
            lock=lock,
        )
    output = root / "results/v15/dev-pi-replication"
    config = yaml.safe_load((root / "configs/v15/confirmatory.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise TypeError("Pi DEV config must be a mapping")
    planes = load_task_planes(root / "configs/v15/task_manifest.json")
    settings = resolve_runtime_settings(
        root,
        artifact_root=output / "artifacts",
        log_root=output / "inspect-logs",
        verify_image=False,
    )
    return _run_registered_pi_audit(
        root,
        config=config,
        planes=planes,
        settings=settings,
        output=output,
        trajectories=1,
        rounds=1,
        candidates_count=1,
        confirmatory=False,
        agent_steps=agent_steps,
        lock=None,
        task_limit=task_limit,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the predeclared Pi replication")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--task-limit", type=int, default=1)
    parser.add_argument("--agent-steps", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run_pi_replication(args.root.resolve(), task_limit=args.task_limit, agent_steps=args.agent_steps), sort_keys=True))


__all__ = [
    "_pi_token_count",
    "build_pi_confirmatory_plan",
    "evaluate_pi_with_inspect",
    "pair_pi_execution_rows",
    "pi_budget_violation",
    "pi_sandbox_command",
    "run_pi_replication",
]
