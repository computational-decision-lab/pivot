"""Native adapter smoke: deterministic repeated rollout and an ego-only change."""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
from native_adapter import NativeEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=60000)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--num-agents", type=int, default=8)
    args = parser.parse_args()
    started = time.monotonic()
    # Explicit smoke settings, not implicit adapter/scientific defaults.
    base = {"speed": 20.0, "headway": 1.5, "distance": 5.0, "lane_change_distance": 15.0}
    alternate = {**base, "speed": 30.0}
    with NativeEvaluator({"horizon": args.horizon, "num_agents": args.num_agents}) as evaluator:
        first = evaluator.run_episode(args.seed, base, base)
        repeat = evaluator.run_episode(args.seed, base, base)
        changed = evaluator.run_episode(args.seed, alternate, base)
        ids = first["initial_agent_ids"]
        same_positions = first["initial_positions"] == repeat["initial_positions"] == changed["initial_positions"]
        maximum_return_difference = max(abs(first["per_agent_returns"][key] - repeat["per_agent_returns"][key]) for key in ids)
        repeat_returns_match = bool(np.allclose(
            [first["per_agent_returns"][key] for key in ids],
            [repeat["per_agent_returns"][key] for key in ids], rtol=0.0, atol=1e-9))
        repeat_trajectory_match = first["trajectory_sha256"] == repeat["trajectory_sha256"]
        changed_trajectory = first["trajectory_sha256"] != changed["trajectory_sha256"]
        report = {
            "smoke_ok": same_positions and repeat_returns_match and repeat_trajectory_match and changed_trajectory,
            "same_initial_positions_all_three": same_positions,
            "repeat_max_absolute_return_difference": maximum_return_difference,
            "repeat_returns_match_atol_1e_9": repeat_returns_match,
            "repeat_trajectory_sha256_match": repeat_trajectory_match,
            "changed_ego_speed_changes_trajectory": changed_trajectory,
            "seconds": time.monotonic() - started,
            "effective_native_config": evaluator.config_metadata,
            "runs": [first, repeat, changed],
        }
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    if not report["smoke_ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
