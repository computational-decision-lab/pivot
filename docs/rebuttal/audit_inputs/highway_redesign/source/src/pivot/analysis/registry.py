from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RegisteredRun:
    run_id: str
    seeds: tuple[int, ...]


def load_registry(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("registry must be a mapping")
    experiment = payload.get("experiment")
    if experiment not in {"p2", "e4", "e5", "e6", "e7", "e8", "e9", "ablations"}:
        raise ValueError("registry experiment must be one of p2, e4, e5, e6, e7, e8, e9, ablations")
    seed_sets = payload.get("seed_sets")
    if not isinstance(seed_sets, list) or not seed_sets:
        raise ValueError("registry seed_sets must not be empty")
    seen: set[int] = set()
    run_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for item in seed_sets:
        if not isinstance(item, dict):
            raise TypeError("each seed set must be a mapping")
        run_id = str(item.get("run_id", ""))
        seeds = tuple(int(seed) for seed in item.get("seeds", []))
        if not run_id or run_id in run_ids:
            raise ValueError("registry run IDs must be unique and non-empty")
        if not seeds:
            raise ValueError("registry seed sets must be non-empty")
        if seen.intersection(seeds):
            raise ValueError("registry seed sets must be disjoint")
        if any(seed < 0 for seed in seeds):
            raise ValueError("registry seeds must be non-negative")
        run_ids.add(run_id)
        seen.update(seeds)
        normalized.append({"run_id": run_id, "seeds": list(seeds)})
    result = dict(payload)
    result["seed_sets"] = normalized
    result["registry_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return result


def materialize_seed_config(base_config: Path, seeds: list[int] | tuple[int, ...], output: Path) -> str:
    payload = yaml.safe_load(Path(base_config).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("base config must be a mapping")
    payload = dict(payload)
    payload["seeds"] = [int(seed) for seed in seeds]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return hashlib.sha256(output.read_bytes()).hexdigest()


def run_registered(
    registry_path: Path,
    output_root: Path,
    *,
    project_root: Path | None = None,
    python_executable: str | None = None,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Execute a frozen registry without overwriting any existing run."""

    registry_path = Path(registry_path).resolve()
    registry = load_registry(registry_path)
    root = Path(project_root or Path.cwd()).resolve()
    base_config = _resolve_base_config(registry_path, str(registry["base_config"]), root)
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    executable = python_executable or sys.executable
    script = {
        "p2": "scripts/run_sweep.py",
        "e4": "experiments/e4_global_vs_local.py",
        "e5": "experiments/e5_budget_frontier.py",
        "e6": "experiments/e6_finance_actor.py",
        "e7": "experiments/e7_strategic_reversal.py",
        "e8": "experiments/e8_competition.py",
        "e9": "experiments/e9_closed_loop.py",
        "ablations": "experiments/ablation_suite.py",
    }[str(registry["experiment"])]
    results: list[dict[str, Any]] = []
    for item in registry["seed_sets"]:
        run_id = str(item["run_id"])
        run_dir = output_root / run_id
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite registered run: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.yaml"
        config_sha256 = materialize_seed_config(base_config, item["seeds"], config_path)
        command = [executable, script, "--config", str(config_path), "--output", str(run_dir)]
        started = datetime.now(timezone.utc).isoformat()
        status = "ok"
        exit_code = 0
        with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (run_dir / "stderr.log").open(
            "w", encoding="utf-8"
        ) as stderr:
            completed = subprocess.run(command, cwd=root, stdout=stdout, stderr=stderr, check=False)
            exit_code = int(completed.returncode)
            if exit_code != 0:
                status = "failed"
        manifest = {
            "run_id": run_id,
            "status": status,
            "exit_code": exit_code,
            "experiment": registry["experiment"],
            "seeds": list(item["seeds"]),
            "command": command,
            "registry": str(registry_path),
            "registry_sha256": registry["registry_sha256"],
            "base_config": str(base_config),
            "config_sha256": config_sha256,
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        dump_json(run_dir / "run_manifest.json", manifest)
        results.append(manifest)
        if status == "failed" and fail_fast:
            break
    return {
        "registry": str(registry_path),
        "experiment": registry["experiment"],
        "n_runs": len(results),
        "n_ok": sum(result["status"] == "ok" for result in results),
        "n_failed": sum(result["status"] != "ok" for result in results),
        "runs": results,
    }


def _resolve_base_config(registry_path: Path, value: str, project_root: Path) -> Path:
    candidate = Path(value)
    options = [
        candidate if candidate.is_absolute() else Path.cwd() / candidate,
        registry_path.parent / candidate,
        project_root / candidate,
    ]
    for option in options:
        if option.exists():
            return option.resolve()
    raise FileNotFoundError(f"base config does not exist: {value}")


def dump_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
