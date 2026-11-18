#!/usr/bin/env python3
"""Run the frozen-state matched selector diagnostic.

The protocol keeps the state, incumbent, candidate panel, proxy values, and
evaluation streams identical across selectors. It is separate from v3c and
does not mutate any v3c result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from run_multiround_loop import ResponseWorld, deployment_delta, make_row, propose, sha256_bytes, stable_seed
from run_multiround_loop_v3b import (
    RESPONSE_REGIMES,
    calibration_cv,
    condition_noisy_observation,
    fit_contextual_model,
    make_calibration_samples,
    select_query,
)


METHODS = ("PIVOT-KG-Loop-v3c", "IVR-Online-v3c", "Top-Proxy-Online-v3c")
BASE_METHOD = {
    "PIVOT-KG-Loop-v3c": "PIVOT-KG-Loop-v3b",
    "IVR-Online-v3c": "IVR-Online",
    "Top-Proxy-Online-v3c": "Top-Proxy-Online",
}
ROOT_START = {"low": 990000, "medium": 1000000, "high": 1010000}


def run(*, source: Path, output: Path, roots_per_regime: int = 120) -> None:
    if output.exists():
        raise SystemExit(f"output exists; choose a new directory: {output}")
    if roots_per_regime != 120:
        raise ValueError("the locked auxiliary protocol uses 120 roots per regime")
    sys.path[:0] = [str(source), str(source / "src")]
    from experiments.revision.highway_calibration import CalibrationModel

    samples, calibration_records, _ = make_calibration_samples(
        source, calibration_roots=240, rounds=5
    )
    cv = calibration_cv(samples, folds=5, ridge=0.5)
    model = fit_contextual_model(
        samples,
        ridge=0.5,
        observation_variance=np.asarray(cv["observation_variance"], dtype=float),
        residual_variance=float(cv["model_residual_variance"]),
    )
    model_record = model.to_record()
    calibration_hash = sha256_bytes(json.dumps(model_record, sort_keys=True).encode())
    ledger: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []

    for regime, strength in RESPONSE_REGIMES.items():
        for offset in range(roots_per_regime):
            root = ROOT_START[regime] + offset
            pressure = 0.10 + 0.015 * (offset % 7)
            incumbent = {"intensity": 0.18, "bias": 0.0}
            world = ResponseWorld(pressure=pressure, response_strength=strength)
            candidates = propose(incumbent, root=root, round_id=0)
            rows = [
                make_row(world, incumbent, candidate, root=root, round_id=0, index=index)
                for index, candidate in enumerate(candidates)
            ]
            panel_hash = sha256_bytes(json.dumps(rows, sort_keys=True).encode())
            eval_seed = stable_seed("frozen-state-v3c-independent-eval", root)
            deployment_seed = stable_seed("frozen-state-v3c-deployment-return", root)
            eval_deltas = [
                deployment_delta(world, incumbent, candidate, seed=eval_seed + index)
                for index, candidate in enumerate(candidates)
            ]
            for method in METHODS:
                posterior = model.clone().predict(
                    rows,
                    pressure=pressure,
                    response_strength=strength,
                    incumbent=incumbent,
                    round_id=0,
                    rounds=5,
                    calibration_model_type=CalibrationModel,
                )
                query_details: list[dict[str, Any]] = []
                queried: list[str] = []
                for query_order in range(2):
                    candidate_id, acquisition = select_query(
                        BASE_METHOD[method], posterior, rows, set(queried)
                    )
                    index = posterior._index[candidate_id]
                    query_seed = stable_seed("frozen-state-v3c-hf", root, candidate_id)
                    value = deployment_delta(
                        world, incumbent, rows[index]["candidate_parameters"], seed=query_seed
                    )
                    condition_noisy_observation(posterior, candidate_id, value)
                    queried.append(candidate_id)
                    query_details.append(
                        {
                            "candidate_id": candidate_id,
                            "candidate_index": index,
                            "query_order": query_order,
                            "query_seed": query_seed,
                            "observation": float(value),
                            "acquisition": acquisition,
                            "observation_variance": float(posterior.observation_variance[index]),
                            "posterior_latent_variance": float(posterior.covariance[index, index]),
                        }
                    )
                selected = posterior.select()
                selected_delta = 0.0 if selected == "incumbent" else float(eval_deltas[int(selected[1:])])
                best_delta = max([0.0, *eval_deltas])
                isr = best_delta - selected_delta
                chosen = incumbent if selected == "incumbent" else candidates[int(selected[1:])]
                deployment_return = world.value(chosen, seed=deployment_seed, deployment=True)
                ledger.append(
                    {
                        "protocol_id": "FROZEN-STATE-V3C-2026-11-10",
                        "regime": regime,
                        "response_strength": strength,
                        "root_id": root,
                        "method": method,
                        "round_id": 0,
                        "pressure_before": pressure,
                        "candidate_panel": rows,
                        "panel_hash": panel_hash,
                        "calibration_hash": calibration_hash,
                        "evaluation_seed": eval_seed,
                        "deployment_seed": deployment_seed,
                        "query_ids": queried,
                        "query_count": len(queried),
                        "query_details": query_details,
                        "selected_policy": selected,
                        "terminal_selection_rule": "posterior mean over incumbent plus four candidates",
                        "deployment_outcome": {
                            "independent_deltas": [float(value) for value in eval_deltas],
                            "selected_delta": selected_delta,
                            "best_delta": best_delta,
                            "ISR": float(isr),
                            "deployment_return": float(deployment_return),
                        },
                        "initial_state_hash": world.state_hash(),
                    }
                )
                trajectories.append(
                    {
                        "protocol_id": "FROZEN-STATE-V3C-2026-11-10",
                        "regime": regime,
                        "root_id": root,
                        "method": method,
                        "ISR": float(isr),
                        "deployment_return": float(deployment_return),
                        "selected_policy": selected,
                        "query_ids": queried,
                    }
                )

    output.mkdir(parents=True)
    ledger_blob = json.dumps(ledger, sort_keys=True, separators=(",", ":")).encode()
    (output / "trajectory_ledger.json").write_bytes(ledger_blob)
    (output / "trajectory_ledger.sha256").write_text(sha256_bytes(ledger_blob) + "\n")
    (output / "trajectories.json").write_text(json.dumps(trajectories, indent=2, sort_keys=True) + "\n")
    (output / "calibration_model.json").write_text(json.dumps(model_record, indent=2, sort_keys=True) + "\n")
    (output / "calibration_training_samples.json").write_text(json.dumps(samples, indent=2, sort_keys=True) + "\n")
    (output / "calibration_rows.json").write_text(json.dumps(calibration_records, indent=2, sort_keys=True) + "\n")
    protocol = {
        "protocol_id": "FROZEN-STATE-V3C-2026-11-10",
        "split": "post_review_auxiliary",
        "roots_per_regime": roots_per_regime,
        "root_blocks": ROOT_START,
        "calibration_roots": 240,
        "rounds": 1,
        "candidate_count": 4,
        "hf_budget_per_root": 2,
        "response_regimes": RESPONSE_REGIMES,
        "initial_pressure": "0.10 + 0.015 * (root_offset mod 7)",
        "state_rule": "fixed initial state and incumbent; no deployment carries between roots",
        "shared_across_methods": ["incumbent", "candidate_panel", "proxy_values", "calibration_model", "seeds", "budget", "terminal_rule"],
        "acquisition": {
            "PIVOT-KG-Loop-v3c": "exact one-step Gaussian KG, sequential posterior",
            "IVR-Online-v3c": "integrated variance reduction, sequential posterior",
            "Top-Proxy-Online-v3c": "descending proxy delta, sequential posterior",
        },
        "cross_round_update": "none",
        "physical_simulator_called": False,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    summary = {
        "status": "PASS",
        "protocol": protocol,
        "protocol_sha256": sha256_bytes(json.dumps(protocol, sort_keys=True).encode()),
        "methods": list(METHODS),
        "ledger_sha256": sha256_bytes(ledger_blob),
        "calibration_model_hash": calibration_hash,
        "calibration": {"five_fold_cv": cv},
        "counts": {"roots_per_regime": roots_per_regime, "regimes": 3, "methods": 3, "ledger_rows": len(ledger), "queries": sum(row["query_count"] for row in ledger)},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(output), "ledger_rows": len(ledger)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(source=args.source, output=args.output)
