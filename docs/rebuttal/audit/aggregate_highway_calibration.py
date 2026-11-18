#!/usr/bin/env python3
"""Combine disjoint calibration shards and fit one sealed correction model."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import sys


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    sys.path[:0] = [str(source), str(source / "src")]
    from experiments.revision.highway_calibration import fit_calibration_model
    from pivot.cloud.archive import atomic_json

    expected_seeds = [3_700_003 + 43 * index for index in range(40)]
    proxy: dict[int, list[dict[str, Any]]] = defaultdict(list)
    truth: dict[int, dict[str, float]] = defaultdict(dict)
    study_hashes: set[str] = set()
    source_identities: set[str] = set()
    origin_protocols: set[str] = set()
    for shard_path in args.shards:
        shard = shard_path.resolve()
        summary = json.loads((shard / "summary.json").read_text())
        if summary.get("status") != "CALIBRATION_READY":
            raise ValueError(f"calibration shard is incomplete: {shard}")
        shard_manifest = json.loads((shard / "content-manifest.json").read_text())
        for name, item in shard_manifest["files"].items():
            path = shard / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"calibration shard hash mismatch: {path}")
        identity = json.loads((shard / "identity.json").read_text())
        study_hashes.add(identity["config"]["study_config_hash"])
        origin_protocols.add(str(identity["config"]["protocol_id"]))
        source_identities.add(identity["source"]["historical_source_hash"])
        for row in json.loads((shard / "calibration-proxy-rows.json").read_text())["rows"]:
            proxy[int(row["seed"])].append({key: value for key, value in row.items() if key != "seed"})
        for row in json.loads((shard / "calibration-truth.json").read_text())["rows"]:
            truth[int(row["seed"])][str(row["candidate_id"])] = float(row["paired_delta"])

    if len(study_hashes) != 1 or len(source_identities) != 1:
        raise ValueError("calibration shards do not share a single study/source identity")
    if sorted(proxy) != expected_seeds or sorted(truth) != expected_seeds:
        raise ValueError("calibration shards do not cover the exact frozen 40-root cohort")
    if any(len(rows) != 8 for rows in proxy.values()) or any(len(rows) != 8 for rows in truth.values()):
        raise ValueError("each calibration seed must contain exactly eight candidate records")

    model = fit_calibration_model(proxy, truth, covariance_shrinkage=0.5)
    model_record = model.to_record()
    base_config_hash = next(iter(study_hashes))
    decision_config_hash = _digest({"base_config_hash": base_config_hash, "model": model_record})
    atomic_json(args.output / "calibration-proxy-rows.json", {
        "rows": [{"seed": seed, **row} for seed in expected_seeds for row in proxy[seed]]
    })
    atomic_json(args.output / "calibration-truth.json", {
        "rows": [{"seed": seed, "candidate_id": candidate, "paired_delta": value}
                 for seed in expected_seeds for candidate, value in truth[seed].items()]
    })
    atomic_json(args.output / "calibration-model.json", model_record)
    atomic_json(args.output / "decision-identity.json", {
        "base_config_hash": base_config_hash,
        "decision_config_hash": decision_config_hash,
        "calibration_model": model_record,
    })
    summary = {
        "protocol_id": "highway-physical-reactive-matched-targeted-v2",
        "calibration_source_protocol_ids": sorted(origin_protocols),
        "status": "CALIBRATION_READY",
        "calibration_seed_count": len(proxy),
        "calibration_seeds": expected_seeds,
        "candidate_count": 8,
        "study_config_hash": base_config_hash,
        "source_hash": next(iter(source_identities)),
        "calibration_model_sha256": _digest(model_record),
        "failures": [],
        "scope": "merged independent calibration cohort; test outcomes unopened",
    }
    atomic_json(args.output / "summary.json", summary)
    files = {}
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "content-manifest.json":
            data = path.read_bytes()
            files[str(path.relative_to(args.output))] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
    atomic_json(args.output / "content-manifest.json", {"files": files})
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
