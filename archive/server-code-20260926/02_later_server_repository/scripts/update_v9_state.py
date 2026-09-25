"""Materialize the V9 research-state ledger from scientific decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pivot.v9.artifacts import write_json


def update(root: Path) -> dict[str, Any]:
    state_path = root / "research/research_state_v9.json"
    raw_state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(raw_state, dict):
        raise TypeError("research state must be an object")
    state: dict[str, Any] = raw_state
    statuses: dict[str, str] = {}
    for experiment in ("e2c", "e3c", "e4c", "e5c", "e7c"):
        decision = root / f"results/v9/{experiment}-confirmatory/scientific_decision.json"
        if decision.is_file():
            payload = json.loads(decision.read_text(encoding="utf-8"))
            status = str(payload.get("status", "IMPLEMENTATION_FAILURE"))
            key = experiment.upper()
            statuses[key] = status
            state[key] = status
    state["P0"] = "PASSED"
    state["paper"] = "LOCAL_PDF_BUILT_CONDITIONAL_GO"
    state["claim_policy"] = "Claims are scoped by research/claims_v9.yaml and powered scientific_decision.json artifacts; universal superiority remains forbidden."
    state["confirmatory_statuses"] = statuses
    write_json(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Update V9 research state")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    state = update(args.root.resolve())
    print(json.dumps({key: state[key] for key in ("P0", "E2C", "E3C", "E4C", "E5C", "E7C", "paper")}, sort_keys=True))


if __name__ == "__main__":
    main()
