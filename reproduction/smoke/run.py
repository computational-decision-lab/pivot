#!/usr/bin/env python3
"""Small source and native-environment checks; never launches a study cohort."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "evidence" / "paper"


def require(module: str) -> None:
    if importlib.util.find_spec(module) is None:
        raise RuntimeError(f"native smoke requires missing Python module {module!r}")


def smoke(experiment: str) -> dict:
    if experiment == "controlled":
        source = PAPER / "source" / "controlled" / "src"
        old_path = list(sys.path)
        sys.path.insert(0, str(source))
        try:
            from pivot.core.policy import Policy
            from pivot.v9.environments import environment_for
            world = environment_for("performative_control", 0.8)
            policy = Policy.from_mapping({"intensity": 0.18, "bias": 0.0})
            rollout = world.evaluate(policy, seed=7, mode="actor")
            if rollout.environment_steps <= 0:
                raise RuntimeError("controlled rollout recorded no environment steps")
            return {"check": "frozen controlled world rollout", "native_environment_step": False,
                    "environment_steps": rollout.environment_steps}
        finally:
            sys.path[:] = old_path
    if experiment in ("kuhn", "leduc", "leduc_v4"):
        require("pyspiel")
        import pyspiel
        game_name = "kuhn_poker" if experiment == "kuhn" else "leduc_poker"
        game = pyspiel.load_game(game_name)
        state = game.new_initial_state()
        actions = 0
        while not state.is_terminal():
            state.apply_action(state.legal_actions()[0])
            actions += 1
        assert actions > 0
        return {"check": "OpenSpiel native game episode", "game": game_name, "actions": actions,
                "native_environment_step": True}
    if experiment == "meltingpot":
        require("meltingpot")
        from meltingpot import substrate
        config = substrate.get_config("stag_hunt_in_the_matrix__repeated")
        env = substrate.build("stag_hunt_in_the_matrix__repeated",
                              roles=config.default_player_roles)
        try:
            env.reset()
            actions = [spec.generate_value() for spec in env.action_spec()]
            env.step(actions)
        finally:
            env.close()
        return {"check": "Melting Pot native substrate reset and step", "native_environment_step": True}
    if experiment in ("highway", "highway_redesign"):
        require("highway_env")
        import gymnasium as gym
        import highway_env  # noqa: F401 - registers the native environment
        env = gym.make("highway-v0")
        try:
            env.reset(seed=0)
            env.step(env.action_space.sample())
        finally:
            env.close()
        launcher = ROOT / "reproduction" / experiment / "run.py"
        subprocess.run([sys.executable, str(launcher), "--check-only"], check=True,
                       stdout=subprocess.DEVNULL,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        return {"check": "HighwayEnv native step and frozen source check",
                "native_environment_step": True}
    if experiment == "metadrive":
        require("metadrive")
        code = PAPER / "metadrive" / "code"
        completed = subprocess.run([sys.executable, str(code / "smoke_native.py"),
                                    "--horizon", "20", "--num-agents", "3"],
                                   cwd=code, check=True, text=True, capture_output=True,
                                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        native = json.loads(completed.stdout.strip().splitlines()[-1])
        return {"check": "frozen MetaDrive adapter repeated native rollout",
                "native_environment_step": True, "smoke_ok": native["smoke_ok"]}
    raise ValueError(experiment)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=(
        "controlled", "kuhn", "leduc", "leduc_v4", "meltingpot",
        "highway", "highway_redesign", "metadrive"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        details = smoke(args.experiment)
        result = {"experiment": args.experiment, "status": "passed", **details}
    except Exception as exc:
        result = {"experiment": args.experiment, "status": "failed", "error": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
