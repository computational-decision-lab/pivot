"""Run unchanged upstream CPU experiments and verify their saved artifacts."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "colin_pivot"
COMMIT = "9e3be723dbe895101c13dd3a6782d3e058b91558"


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def verify(directory: Path) -> dict:
    decision = json.loads((directory / "scientific_decision.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    errors = []
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            errors.append(name)
    failure_ledger = directory / "failure_ledger.jsonl"
    if failure_ledger.exists() and failure_ledger.read_text().strip():
        errors.append("failure_ledger contains failures")
    provenance = json.loads((directory / "provenance.json").read_text())
    if provenance["source_commit"] != COMMIT:
        errors.append("source commit mismatch")
    status = decision["status"]
    valid = not errors and status in {"UNDERPOWERED", "HYPOTHESIS_SUPPORTED", "HYPOTHESIS_NOT_SUPPORTED"}
    return {"artifacts_valid": valid, "scientific_status": status,
            "powered": decision["powered"], "errors": errors,
            "verified_files": len(manifest["files"])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "dev", "confirmatory"], default="smoke")
    parser.add_argument("--experiments", nargs="+", choices=["e3c", "e5c", "e7c"], default=["e3c", "e5c", "e7c"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    if git("rev-parse", "HEAD") != COMMIT or git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Upstream commit differs or tracked files were modified")
    for name in ["state.json", "metrics.json"]:
        if not (REPO / "results/v7/e3b-confirmatory" / name).is_file():
            raise FileNotFoundError(f"Missing frozen V7 reference: {name}")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = (args.output or HERE / "runs" / stamp).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(REPO / "src") + os.pathsep + str(REPO),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    report = {"source_commit": COMMIT, "profile": args.profile,
              "python": platform.python_version(), "platform": platform.platform(),
              "cpu_count": os.cpu_count(),
              "packages": {name: importlib.metadata.version(name) for name in ["numpy", "PyYAML", "pytest"]},
              "gpu_used": False, "model_api_used": False, "runs": []}

    def save() -> None:
        (out / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    test_cmd = [sys.executable, "-m", "pytest", "-q", "tests/unit/test_pivot_voi.py",
                "tests/integration/test_pivot_voi_round.py"]
    with (out / "core_tests.log").open("w") as log:
        tested = subprocess.run(test_cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    report["core_test_returncode"] = tested.returncode
    save()
    if tested.returncode or args.check_only:
        print(json.dumps({"core_test_returncode": tested.returncode, "output": str(out)}))
        return tested.returncode
    for experiment in args.experiments:
        start = time.monotonic()
        target = out / experiment
        command = [sys.executable, "-m", "experiments.v9.run", "--experiment", experiment,
                   "--profile", args.profile, "--output", str(target), "--root", str(REPO)]
        record = {"experiment": experiment}
        with (out / f"{experiment}.log").open("w") as log:
            try:
                result = subprocess.run(command, cwd=REPO, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=args.timeout_seconds)
                record["returncode"] = result.returncode
                if result.returncode == 0:
                    record.update(verify(target))
            except subprocess.TimeoutExpired:
                record.update(returncode=124, timed_out=True)
        record["wall_seconds"] = round(time.monotonic() - start, 3)
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        record["cumulative_child_peak_rss_mib"] = peak / (1024 ** 2 if sys.platform == "darwin" else 1024)
        report["runs"].append(record)
        save()
        print(json.dumps(record), flush=True)
        if record["returncode"] or not record.get("artifacts_valid"):
            return 1
    print(json.dumps({"completed": True, "output": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
