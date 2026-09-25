"""Run a reduced-seed author replication without modifying exported source.

Only provenance and per-seed persistence are instrumented. The public author
algorithms, candidate generation, statistics, and stopping rules are unchanged.
This is an engineering sanity gate, not a new confirmatory experiment.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
from datetime import datetime, timezone

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

BUNDLE = Path(__file__).resolve().parent
AUTHOR = BUNDLE / "author"
COMMIT = "693585913c17ff8359739e7dada633b65e41f0ba"
sys.path[:0] = [str(AUTHOR / "src"), str(AUTHOR)]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_source():
    source = read(BUNDLE / "SOURCE_MANIFEST.json")
    if source["source_commit"] != COMMIT:
        raise RuntimeError("Unexpected author commit")
    for relative, record in source["files"].items():
        if digest(AUTHOR / relative) != record["sha256"]:
            raise RuntimeError(f"Author source or frozen reference changed: {relative}")
    references = read(BUNDLE / "sanity_reference/REFERENCE_MANIFEST.json")
    for relative, record in references["files"].items():
        if digest(BUNDLE / "sanity_reference" / relative) != record["sha256"]:
            raise RuntimeError(f"Frozen reference changed: {relative}")
    return source


def install_export_provenance():
    # A GitHub archive has no .git. Explicitly identify its hash-verified source
    # instead of accidentally reporting a surrounding workspace's Git commit.
    import pivot.v9.artifacts as artifacts
    import experiments.v9.common as common

    def export_commit(root):
        if Path(root).resolve() != AUTHOR:
            raise ValueError("Unexpected author root for exported-source provenance")
        return COMMIT

    artifacts.current_commit = export_commit
    common.current_commit = export_commit


class SeedRecorder:
    def __init__(self, root, expected):
        self.root, self.expected = root, expected
        self.counts = {}
        self.lock = threading.Lock()

    def record(self, seed, label, rows, trajectory=None):
        from pivot.v9.artifacts import write_jsonl_gz
        seed_root = self.root / f"seed_{seed}"
        with self.lock:
            seed_root.mkdir(parents=True, exist_ok=True)
            write_jsonl_gz(seed_root / f"{label}.jsonl.gz", rows)
            if trajectory is not None:
                write(seed_root / f"{label}.metrics.json", trajectory)
            self.counts[seed] = self.counts.get(seed, 0) + 1
            complete = self.counts[seed] == self.expected
            write(seed_root / "status.json", {
                "seed": seed, "completed_units": self.counts[seed],
                "expected_units": self.expected,
                "status": "COMPLETE" if complete else "RUNNING", "updated_at": now(),
                "source_commit": COMMIT,
            })
            if complete:
                print(json.dumps({"seed_completed": seed, "output": str(seed_root)}), flush=True)


def row_identity(row):
    return tuple(row.get(k) for k in (
        "environment_id", "response_strength", "operator_family", "operator_shift",
        "seed", "round_id", "candidate_id", "method",
    ))


def rows_from(path):
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def reference_check(experiment, output, config, profile):
    import numpy as np
    seeds = set(range(config["seed_start"], config["seed_start"] + profile["seeds"]))
    environments = {item["id"] for item in config["environments"]}
    reference_root = BUNDLE / "sanity_reference"
    expected = {
        row_identity(row): row for row in rows_from(reference_root / f"{experiment}_rows.jsonl.gz")
        if row["seed"] in seeds and row["environment_id"] in environments
    }
    actual = {row_identity(row): row for row in rows_from(output / "transition_rows.jsonl.gz")}
    identity_match = actual.keys() == expected.keys()
    fields = ["delta_proxy", "delta_true", "actor_candidate_value", "proxy_candidate_value"]
    if experiment == "e3c":
        fields += ["estimated_delta", "CISR", "CTI"]
    numeric_mismatches, decision_mismatches, max_absolute_difference = 0, 0, 0.0
    examples = []
    for key in actual.keys() & expected.keys():
        left, right = actual[key], expected[key]
        for field in fields:
            a, b = float(left[field]), float(right[field])
            max_absolute_difference = max(max_absolute_difference, abs(a - b))
            if not np.isclose(a, b, rtol=1e-8, atol=1e-10):
                numeric_mismatches += 1
                if len(examples) < 8:
                    examples.append({"identity": key, "field": field, "actual": a, "reference": b})
        if experiment == "e3c":
            for field in ("selected", "hf_queried", "true_best"):
                decision_mismatches += left[field] != right[field]
    result = {
        "reference_type": "same frozen candidate/trajectory slice, not full-paper aggregate",
        "reference_commit": COMMIT, "seed_list": sorted(seeds),
        "environments": sorted(environments), "candidate_count": profile["candidates_per_round"],
        "rounds": profile["rounds"] if experiment == "e3c" else 1,
        "expected_rows": len(expected), "actual_rows": len(actual),
        "identity_match": identity_match, "numeric_mismatches": numeric_mismatches,
        "decision_mismatches": int(decision_mismatches),
        "max_absolute_difference": max_absolute_difference,
        "tolerance": {"rtol": 1e-8, "atol": 1e-10}, "examples": examples,
        "status": "PASS" if identity_match and not numeric_mismatches and not decision_mismatches else "REVIEW_REQUIRED",
        "policy": "Stop new benchmark work and inspect any mismatch; do not tune seeds or delete negative outcomes.",
    }
    if experiment == "e3c":
        reference_trajectories = [row for row in read(reference_root / "e3c_trajectory_metrics.json")
                                  if row["seed"] in seeds and row["environment_id"] in environments]
        lookup = {(row["environment_id"], row["seed"], row["method"]): row for row in reference_trajectories}
        gains = [lookup[(env, seed, "proxy_only")]["CISR"] - lookup[(env, seed, "pivot_voi")]["CISR"]
                 for env in sorted(environments) for seed in sorted(seeds)]
        result["matched_reference_proxy_minus_pivot_cisr"] = float(np.mean(gains))
        result["reference_calibration_seeds"] = 16
    write(output / "sanity_reference_check.json", result)
    return result


def run_one(experiment, output, profile):
    import yaml
    from experiments.v9 import e2c_operator_shift, e3c_closed_loop
    from pivot.v9.artifacts import build_manifest
    config_path = BUNDLE / "sanity_configs" / f"{experiment}.yaml"
    config = yaml.safe_load(config_path.read_text())
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    expected = (sum(len(x["response_strengths"]) for x in config["environments"])
                * len(config["operator_families"]) * len(config["shift_levels"])) if experiment == "e2c" else len(config["environments"]) * len(config["methods"])
    recorder = SeedRecorder(output.parent / f"{experiment}_per_seed", expected)
    module = e2c_operator_shift if experiment == "e2c" else e3c_closed_loop
    if experiment == "e2c":
        original = module.paired_transition_rows

        def recorded_batch(**kwargs):
            rows = original(**kwargs)
            label = hashlib.sha256(str((kwargs["environment_id"], kwargs["response_strength"], kwargs["operator_family"], kwargs["operator_shift"])).encode()).hexdigest()[:20]
            recorder.record(kwargs["seed"], label, rows)
            return rows

        module.paired_transition_rows = recorded_batch
    else:
        original = module._run_trajectory

        def recorded_trajectory(**kwargs):
            rows, trajectory = original(**kwargs)
            recorder.record(kwargs["seed"], f'{kwargs["environment_id"]}_{kwargs["method"]}', rows, trajectory)
            return rows, trajectory

        module._run_trajectory = recorded_trajectory
    started = time.monotonic()
    try:
        decision = module.run(config_path, profile=profile, output=output, root=AUTHOR)
    finally:
        if experiment == "e2c":
            module.paired_transition_rows = original
        else:
            module._run_trajectory = original
    gate = reference_check(experiment, output, config, profile)
    write(output / "deployment_manifest.json", {
        "experiment_id": f"{experiment.upper()}_TENCENT_SANITY_20260916",
        "git_commit": COMMIT, "source_is_git_export": True,
        "wrapper_sha256": digest(__file__), "source_manifest_sha256": digest(BUNDLE / "SOURCE_MANIFEST.json"),
        "config_sha256": digest(config_path), "profile": profile,
        "benchmark": "author controlled-world sanity replication",
        "environments": config["environments"], "methods": config.get("methods"),
        "candidate_count": profile["candidates_per_round"],
        "hf_budget": config.get("hf_budget_per_round"), "adaptation_horizon": "not applicable to controlled sanity",
        "seed_list": list(range(config["seed_start"], config["seed_start"] + profile["seeds"])),
        "completed_at": now(), "elapsed_seconds": time.monotonic() - started,
        "python": platform.python_version(), "machine": platform.machine(),
        "author_scientific_status": decision["status"], "sanity_status": gate["status"],
        "note": "UNDERPOWERED is expected for 8 seeds; it is not an engineering failure or confirmatory evidence.",
    })
    build_manifest(output, experiment_id=f"{experiment.upper()}_SANITY", status=gate["status"])
    print(json.dumps({"experiment": experiment, "author_status": decision["status"], "sanity_status": gate["status"], "seconds": time.monotonic() - started}), flush=True)
    return gate["status"] == "PASS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["e2c", "e3c", "all"], default="all")
    parser.add_argument("--output", type=Path, default=BUNDLE / "runs/sanity_8seeds")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = verify_source()
    profile = read(BUNDLE / "sanity_configs/profile.json")
    if profile["seeds"] != 8 or profile["rounds"] != 40 or profile["candidates_per_round"] != 8:
        raise RuntimeError("Unexpected sanity protocol dimensions")
    if args.dry_run:
        print(json.dumps({"status": "SOURCE_AND_CONFIG_PASS", "source_commit": COMMIT,
                          "source_files": len(source["files"]), "profile": profile,
                          "e2c_expected_rows": 7680, "e3c_expected_rows": 51200,
                          "training_or_model_calls": 0}, indent=2))
        return
    install_export_provenance()
    experiments = ["e2c", "e3c"] if args.experiment == "all" else [args.experiment]
    for experiment in experiments:
        if not run_one(experiment, args.output.resolve() / experiment, profile):
            raise SystemExit(3)


if __name__ == "__main__":
    main()
