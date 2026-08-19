#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.analysis.registered import (
    evaluate_gate_a_b,
    evaluate_gate_c,
    evaluate_gate_d,
    evaluate_gate_e,
    evaluate_gate_f,
    summarize_e4_runs,
    summarize_e5_runs,
    summarize_e6_runs,
    summarize_e7_runs,
    summarize_e8_runs,
    summarize_e9_runs,
    summarize_p2_runs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate registered PIVOT evidence")
    parser.add_argument(
        "--experiment", choices=("p2", "e4", "e5", "e6", "e7", "e8", "e9", "f"), required=True
    )
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-budget", type=int, default=1)
    parser.add_argument("--target-participation", type=float, default=0.05)
    parser.add_argument("--mode", default="adaptive")
    parser.add_argument("--e8-inputs", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    if args.experiment == "p2":
        summary = summarize_p2_runs(args.inputs)
        gates: dict[str, Any] = evaluate_gate_a_b(summary)
    elif args.experiment == "e4":
        summary = summarize_e4_runs(args.inputs)
        gates = evaluate_gate_c(summary)
    elif args.experiment == "e5":
        summary = summarize_e5_runs(args.inputs, target_budget=args.target_budget)
        gates = evaluate_gate_d(summary)
    elif args.experiment == "e6":
        summary = summarize_e6_runs(args.inputs, target_participation=args.target_participation)
        gates = evaluate_gate_e(summary)
    elif args.experiment == "e7":
        summary = summarize_e7_runs(args.inputs)
        gates = {"F": "Not run", "note": "Use experiment f for the combined E7/E8 gate."}
    elif args.experiment == "e8":
        summary = summarize_e8_runs(args.inputs, mode=args.mode)
        gates = {"F": "Not run", "note": "Use experiment f for the combined E7/E8 gate."}
    elif args.experiment == "e9":
        summary = summarize_e9_runs(args.inputs)
        gates = {"F": "Not run", "note": "E9 is a closed-loop artifact summary, not a new gate."}
    else:
        if not args.e8_inputs:
            raise ValueError("--e8-inputs is required for experiment f")
        e7_summary = summarize_e7_runs(args.inputs)
        e8_summary = summarize_e8_runs(args.e8_inputs, mode=args.mode)
        summary = {"e7": e7_summary, "e8": e8_summary}
        gates = evaluate_gate_f(e7_summary, e8_summary)
    payload = {
        "experiment": args.experiment,
        "inputs": [str(path) for path in args.inputs],
        "summary": summary,
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"experiment": args.experiment, "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
