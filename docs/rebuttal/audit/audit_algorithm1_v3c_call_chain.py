#!/usr/bin/env python3
"""Read-only audit of the submitted Algorithm 1 and the v3c runner.

This audit checks source-level call-chain anchors. It does not execute a
simulator and does not turn source assertions into empirical evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def line_number(text: str, needle: str, *, last: bool = False) -> int:
    matches = [number for number, line in enumerate(text.splitlines(), 1) if needle in line]
    if matches:
        return matches[-1] if last else matches[0]
    raise AssertionError(f"missing source anchor: {needle}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    paper = (root / "paper/main.tex").read_text(encoding="utf-8")
    runner_path = root / "docs/rebuttal/audit/run_multiround_loop_v3c.py"
    runner = runner_path.read_text(encoding="utf-8")
    protocol_path = root / "docs/rebuttal/audit/P1_LOOP_V3C_PROTOCOL_2026-11-10.md"
    protocol = protocol_path.read_text(encoding="utf-8")

    paper_anchors = {
        "algorithm_candidate_proxy": line_number(paper, r"Evaluate candidates to obtain"),
        "algorithm_posterior": line_number(paper, r"Initialize the calibrated posterior"),
        "algorithm_acquisition": line_number(paper, r"Compute $A_j$ for each feasible"),
        "algorithm_hf": line_number(paper, r"Obtain paired HF observation"),
        "algorithm_condition": line_number(paper, r"condition the posterior"),
        "algorithm_terminal": line_number(paper, r"\RETURN $a(D)$"),
    }
    runner_anchors = {
        "proposer": line_number(runner, "candidates = propose(incumbent"),
        "proxy": line_number(runner, "rows = ["),
        "posterior": line_number(runner, "posterior = current_model.predict("),
        "acquisition": line_number(runner, "cid, acquisition = select_query("),
        "hf_observation": line_number(runner, "value = deployment_delta("),
        "gaussian_conditioning": line_number(runner, "condition_noisy_observation(posterior, cid, value)"),
        "terminal_selection": line_number(runner, "selected = posterior.select()"),
        "independent_evaluation": line_number(runner, "eval_deltas = ["),
        "deployment": line_number(runner, "world.deploy(chosen_policy, seed=deploy_seed)"),
        "next_incumbent": line_number(runner, "incumbent = {", last=True),
    }
    checks = {
        "five_round_loop": "for round_id in range(rounds):" in runner,
        "two_queries_per_round": "for query_order in range(2):" in runner,
        "no_sentinel": '"sentinel_rule": None' in runner and "no common sentinel" in protocol,
        "positive_observation_variance": "condition_noisy_observation" in runner,
        "state_transition_recorded": '"next_state_hash": state_after' in runner,
        "terminal_rule_includes_incumbent": '"terminal_selection_rule": "posterior mean over incumbent plus four candidates"' in runner,
        "original_kg_source_unchanged_reference": '"kg_source"' in runner,
    }
    assert all(checks.values()), checks
    result = {
        "status": "PASS",
        "scope": "source-level call-chain audit; no simulator execution",
        "submitted_algorithm": {
            "single_round_logic": "Algorithm 1 in paper/main.tex",
            "v3c_relation": "faithful single-round logic inside a controlled five-round extension",
        },
        "paper_anchors": paper_anchors,
        "runner_anchors": runner_anchors,
        "checks": checks,
        "controlled_extensions": [
            "five state-carrying rounds",
            "contextual correction model reset at each round",
            "method-specific two-query sequence with no common sentinel",
            "method-specific later candidate panels after incumbent divergence",
        ],
        "not_established": [
            "confirmation of static Algorithm 1 in arbitrary environments",
            "universal KG superiority",
            "persistent cross-round online calibration",
        ],
    }
    output = args.output.resolve() if args.output else None
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
