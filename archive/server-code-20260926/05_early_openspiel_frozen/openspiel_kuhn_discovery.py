#!/usr/bin/env python3
"""Discovery-only OpenSpiel benchmark for response-induced rank reversal.

This script does not run PIVOT or consume a confirmatory seed.  It constructs
candidate updates in Kuhn poker, evaluates them against a fixed proxy opponent,
then lets the opponent move a fixed number of steps toward its exact best
response to each candidate.  Exact game-tree values are used so the discovery
gate measures benchmark geometry rather than rollout noise.

The output is suitable for deciding whether a fresh, fixed-budget method study
is scientifically identified.  It must not be reported as confirmatory
PIVOT-vs-baseline evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import best_response, cfr, expected_game_score


VERSION = "openspiel_kuhn_fixed_br_discovery_v1"
DEFAULT_ALPHAS = tuple(float(value) for value in np.linspace(0.0, 1.0, 9))


def _clone(game: pyspiel.Game, source: policy.TabularPolicy) -> policy.TabularPolicy:
    copied = policy.TabularPolicy(game)
    if copied.action_probability_array.shape != source.action_probability_array.shape:
        raise ValueError("tabular policy shapes differ")
    copied.action_probability_array[:] = source.action_probability_array
    return copied


def _player_rows(tabular: policy.TabularPolicy) -> np.ndarray:
    return np.asarray([state.current_player() for state in tabular.states], dtype=int)


def _merge_players(
    game: pyspiel.Game,
    player_zero: policy.TabularPolicy,
    player_one: policy.TabularPolicy,
    rows: np.ndarray,
) -> policy.TabularPolicy:
    joint = _clone(game, player_zero)
    joint.action_probability_array[rows == 1] = player_one.action_probability_array[rows == 1]
    return joint


def _mix_player(
    game: pyspiel.Game,
    left: policy.TabularPolicy,
    right: policy.TabularPolicy,
    player_id: int,
    weight: float,
    rows: np.ndarray,
) -> policy.TabularPolicy:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("mixture weight must be in [0, 1]")
    mixed = _clone(game, left)
    mask = rows == player_id
    mixed.action_probability_array[mask] = (
        (1.0 - weight) * left.action_probability_array[mask]
        + weight * right.action_probability_array[mask]
    )
    return mixed


def _best_response_policy(
    game: pyspiel.Game,
    joint: policy.TabularPolicy,
    player_id: int,
) -> policy.TabularPolicy:
    """Materialize OpenSpiel's exact BR at the requested player's infosets."""
    responder = best_response.BestResponsePolicy(game, player_id, joint)
    materialized = _clone(game, joint)
    for index, state in enumerate(materialized.states):
        if state.current_player() != player_id:
            continue
        probabilities = responder.action_probabilities(state, player_id)
        materialized.action_probability_array[index, :] = 0.0
        for action, probability_value in probabilities.items():
            materialized.action_probability_array[index, int(action)] = float(probability_value)
    return materialized


def _value(
    game: pyspiel.Game,
    player_zero: policy.TabularPolicy,
    player_one: policy.TabularPolicy,
) -> float:
    values = expected_game_score.policy_value(
        game.new_initial_state(), [player_zero, player_one]
    )
    return float(values[0])


def _cfr_incumbent(game: pyspiel.Game, iterations: int) -> policy.TabularPolicy:
    solver = cfr.CFRPlusSolver(game)
    for _ in range(iterations):
        solver.evaluate_and_update_policy()
    return solver.average_policy()


def _perturbed_opponent(
    game: pyspiel.Game,
    incumbent: policy.TabularPolicy,
    rows: np.ndarray,
    rng: np.random.Generator,
    beta: float,
    concentration: float,
) -> policy.TabularPolicy:
    opponent = _clone(game, incumbent)
    for index, state in enumerate(opponent.states):
        if state.current_player() != 1:
            continue
        legal = np.flatnonzero(opponent.legal_actions_mask[index])
        random_behavior = rng.dirichlet(np.full(len(legal), concentration, dtype=float))
        row = np.zeros(opponent.action_probability_array.shape[1], dtype=float)
        row[legal] = random_behavior
        opponent.action_probability_array[index] = (
            (1.0 - beta) * incumbent.action_probability_array[index] + beta * row
        )
    opponent.action_probability_array[rows != 1] = incumbent.action_probability_array[rows != 1]
    return opponent


def _rank(values: Sequence[float]) -> list[int]:
    return sorted(range(len(values)), key=lambda index: (-float(values[index]), index))


def _spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    n = len(values_a)
    if n < 2:
        return 1.0
    rank_a = np.empty(n, dtype=float)
    rank_b = np.empty(n, dtype=float)
    for position, index in enumerate(_rank(values_a)):
        rank_a[index] = position
    for position, index in enumerate(_rank(values_b)):
        rank_b[index] = position
    if float(np.std(rank_a)) == 0.0 or float(np.std(rank_b)) == 0.0:
        return 1.0
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def _evaluate_root(
    *,
    game: pyspiel.Game,
    incumbent: policy.TabularPolicy,
    rows: np.ndarray,
    root_seed: int,
    beta: float,
    concentration: float,
    eta: float,
    short_steps: int,
    long_steps: int,
    alphas: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(root_seed)
    proxy_opponent = _perturbed_opponent(
        game, incumbent, rows, rng, beta=beta, concentration=concentration
    )
    joint_incumbent = _merge_players(game, incumbent, proxy_opponent, rows)
    focal_br = _best_response_policy(game, joint_incumbent, player_id=0)
    candidates = [
        _mix_player(game, incumbent, focal_br, 0, alpha, rows) for alpha in alphas
    ]
    proxy_baseline = _value(game, incumbent, proxy_opponent)
    proxy_improvements = [
        _value(game, candidate, proxy_opponent) - proxy_baseline for candidate in candidates
    ]
    root_rows: list[dict[str, Any]] = []
    horizon_summaries: dict[str, Any] = {}
    for horizon_name, steps in (("short", short_steps), ("long", long_steps)):
        response_weight = 1.0 - (1.0 - eta) ** steps
        incumbent_br = _best_response_policy(game, joint_incumbent, player_id=1)
        incumbent_responder = _mix_player(
            game, proxy_opponent, incumbent_br, 1, response_weight, rows
        )
        deployment_baseline = _value(game, incumbent, incumbent_responder)
        deployment_improvements: list[float] = []
        for candidate_index, (alpha, candidate, proxy_improvement) in enumerate(
            zip(alphas, candidates, proxy_improvements)
        ):
            joint_candidate = _merge_players(game, candidate, proxy_opponent, rows)
            opponent_br = _best_response_policy(game, joint_candidate, player_id=1)
            deployed_opponent = _mix_player(
                game, proxy_opponent, opponent_br, 1, response_weight, rows
            )
            deployment_improvement = (
                _value(game, candidate, deployed_opponent) - deployment_baseline
            )
            deployment_improvements.append(deployment_improvement)
            root_rows.append(
                {
                    "root_seed": root_seed,
                    "beta": beta,
                    "concentration": concentration,
                    "horizon": horizon_name,
                    "adaptation_steps": steps,
                    "eta": eta,
                    "response_weight": response_weight,
                    "candidate_index": candidate_index,
                    "alpha": float(alpha),
                    "proxy_improvement": float(proxy_improvement),
                    "deployment_improvement": float(deployment_improvement),
                    "proxy_deployment_gap": float(
                        proxy_improvement - deployment_improvement
                    ),
                }
            )
        proxy_best = int(np.argmax(proxy_improvements))
        deployment_best = int(np.argmax(deployment_improvements))
        ordered = np.sort(np.asarray(deployment_improvements, dtype=float))
        oracle_margin = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else 0.0
        horizon_summaries[horizon_name] = {
            "adaptation_steps": steps,
            "response_weight": response_weight,
            "proxy_best_index": proxy_best,
            "deployment_best_index": deployment_best,
            "proxy_best_alpha": float(alphas[proxy_best]),
            "deployment_best_alpha": float(alphas[deployment_best]),
            "proxy_best_deployment_regret": float(
                max(deployment_improvements) - deployment_improvements[proxy_best]
            ),
            "winner_changed": bool(proxy_best != deployment_best),
            "rank_spearman": _spearman(proxy_improvements, deployment_improvements),
            "mean_abs_gap": float(
                np.mean(np.abs(np.asarray(proxy_improvements) - deployment_improvements))
            ),
            "mean_squared_gap": float(
                np.mean(
                    (np.asarray(proxy_improvements) - deployment_improvements) ** 2
                )
            ),
            "oracle_margin": oracle_margin,
        }
    return root_rows, {
        "root_seed": root_seed,
        "beta": beta,
        "proxy_baseline": proxy_baseline,
        "horizons": horizon_summaries,
    }


def _bootstrap_paired(
    values: Sequence[float], *, rng: np.random.Generator, draws: int
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("bootstrap input must be non-empty")
    indices = rng.integers(0, len(array), size=(draws, len(array)))
    means = np.mean(array[indices], axis=1)
    return {
        "mean": float(np.mean(array)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def _aggregate(
    root_summaries: Sequence[dict[str, Any]], *, bootstrap_draws: int, seed: int
) -> dict[str, Any]:
    output: dict[str, Any] = {"by_horizon": {}, "paired_contrasts": {}}
    for horizon in ("short", "long"):
        items = [row["horizons"][horizon] for row in root_summaries]
        winners = Counter(str(row["deployment_best_index"]) for row in items)
        output["by_horizon"][horizon] = {
            "n_roots": len(items),
            "mean_abs_gap": float(np.mean([row["mean_abs_gap"] for row in items])),
            "mean_squared_gap": float(
                np.mean([row["mean_squared_gap"] for row in items])
            ),
            "proxy_deployment_mismatch_rate": float(
                np.mean([row["winner_changed"] for row in items])
            ),
            "mean_proxy_best_selection_regret": float(
                np.mean([row["proxy_best_deployment_regret"] for row in items])
            ),
            "mean_rank_spearman": float(np.mean([row["rank_spearman"] for row in items])),
            "median_oracle_margin": float(np.median([row["oracle_margin"] for row in items])),
            "winner_counts": dict(sorted(winners.items())),
            "max_winner_share": max(winners.values()) / len(items),
        }
    rng = np.random.default_rng(seed + 99173)
    contrasts = {
        "mean_abs_gap_long_minus_short": [
            row["horizons"]["long"]["mean_abs_gap"]
            - row["horizons"]["short"]["mean_abs_gap"]
            for row in root_summaries
        ],
        "mismatch_long_minus_short": [
            float(row["horizons"]["long"]["winner_changed"])
            - float(row["horizons"]["short"]["winner_changed"])
            for row in root_summaries
        ],
        "proxy_regret_long_minus_short": [
            row["horizons"]["long"]["proxy_best_deployment_regret"]
            - row["horizons"]["short"]["proxy_best_deployment_regret"]
            for row in root_summaries
        ],
    }
    for name, values in contrasts.items():
        output["paired_contrasts"][name] = _bootstrap_paired(
            values, rng=rng, draws=bootstrap_draws
        )
    long_summary = output["by_horizon"]["long"]
    gap_contrast = output["paired_contrasts"]["mean_abs_gap_long_minus_short"]
    regret_contrast = output["paired_contrasts"]["proxy_regret_long_minus_short"]
    output["gates"] = {
        "mechanism_gap_growth": bool(gap_contrast["ci_low"] > 0.0),
        "decision_effect_growth": bool(regret_contrast["ci_low"] > 0.0),
        "long_response_has_rank_reversal": bool(
            long_summary["proxy_deployment_mismatch_rate"] >= 0.25
        ),
        "long_response_has_selection_opportunity": bool(
            long_summary["max_winner_share"] <= 0.85
            and long_summary["median_oracle_margin"] >= 0.002
        ),
    }
    output["gates"]["eligible_for_method_pilot"] = all(output["gates"].values())
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_alphas(raw: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in raw.split(","))
    if not values or values[0] != 0.0 or len(set(values)) != len(values):
        raise ValueError("alphas must be unique and include 0 as the first candidate")
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("alphas must lie in [0, 1]")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "discovery"), default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=61000)
    parser.add_argument("--roots", type=int, default=48)
    parser.add_argument("--cfr-iterations", type=int, default=5000)
    parser.add_argument("--eta", type=float, default=0.062)
    parser.add_argument("--short-steps", type=int, default=1)
    parser.add_argument("--long-steps", type=int, default=8)
    parser.add_argument("--beta-min", type=float, default=0.35)
    parser.add_argument("--beta-max", type=float, default=0.95)
    parser.add_argument("--smoke-beta", type=float, default=0.60)
    parser.add_argument("--dirichlet-concentration", type=float, default=0.35)
    parser.add_argument("--alphas", default=",".join(str(x) for x in DEFAULT_ALPHAS))
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    args = parser.parse_args()
    if args.roots <= 0 or args.cfr_iterations <= 0 or args.bootstrap_draws <= 0:
        raise ValueError("roots, CFR iterations, and bootstrap draws must be positive")
    if not 0.0 < args.eta < 1.0 or args.short_steps < 0 or args.long_steps <= args.short_steps:
        raise ValueError("require eta in (0,1) and 0 <= short_steps < long_steps")
    if not 0.0 <= args.beta_min <= args.beta_max <= 1.0:
        raise ValueError("invalid beta range")
    if args.dirichlet_concentration <= 0.0:
        raise ValueError("Dirichlet concentration must be positive")
    alphas = _parse_alphas(args.alphas)

    game = pyspiel.load_game("kuhn_poker")
    incumbent = _cfr_incumbent(game, args.cfr_iterations)
    rows = _player_rows(incumbent)
    root_count = 1 if args.mode == "smoke" else args.roots
    candidate_rows: list[dict[str, Any]] = []
    root_summaries: list[dict[str, Any]] = []
    for offset in range(root_count):
        root_seed = args.seed_start + offset
        # Derive every world parameter from the root itself so roots can be
        # generated independently and parallel execution cannot change them.
        beta_rng = np.random.default_rng(root_seed + 314159)
        beta = args.smoke_beta if args.mode == "smoke" else float(
            beta_rng.uniform(args.beta_min, args.beta_max)
        )
        new_rows, root_summary = _evaluate_root(
            game=game,
            incumbent=incumbent,
            rows=rows,
            root_seed=root_seed,
            beta=beta,
            concentration=args.dirichlet_concentration,
            eta=args.eta,
            short_steps=args.short_steps,
            long_steps=args.long_steps,
            alphas=alphas,
        )
        candidate_rows.extend(new_rows)
        root_summaries.append(root_summary)

    aggregate = _aggregate(
        root_summaries, bootstrap_draws=args.bootstrap_draws, seed=args.seed_start
    )
    protocol = {
        "version": VERSION,
        "status": "development_discovery_only_not_confirmatory",
        "game": "kuhn_poker",
        "open_spiel_version": getattr(pyspiel, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "mode": args.mode,
        "seed_start": args.seed_start,
        "roots": root_count,
        "cfr_iterations": args.cfr_iterations,
        "eta": args.eta,
        "short_steps": args.short_steps,
        "long_steps": args.long_steps,
        "short_response_weight": 1.0 - (1.0 - args.eta) ** args.short_steps,
        "long_response_weight": 1.0 - (1.0 - args.eta) ** args.long_steps,
        "beta_min": args.beta_min,
        "beta_max": args.beta_max,
        "smoke_beta": args.smoke_beta,
        "dirichlet_concentration": args.dirichlet_concentration,
        "candidate_alphas": alphas,
        "bootstrap_draws": args.bootstrap_draws,
        "candidate_generator": "behavioral mixture of CFR+ incumbent and exact BR to fixed proxy opponent",
        "response_mechanism": "behavioral mixture of proxy opponent and exact BR to each frozen focal candidate",
        "deployment_baseline": "incumbent evaluated against its own matched responder",
        "selection_or_audit_data_used": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "argv": sys.argv,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "candidate_values.csv", candidate_rows)
    (args.output_dir / "root_summaries.json").write_text(
        json.dumps(root_summaries, indent=2, sort_keys=True), encoding="utf-8"
    )
    result = {"protocol": protocol, "aggregate": aggregate}
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
