#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pivot.core.policy import Policy
from pivot.core.result import RolloutContext
from pivot.core.transition import PolicyTransition
from pivot.environments.execution_replay import ExecutionReplayWorld
from pivot.environments.finance_backtest.config import FinanceConfig
from pivot.environments.finance_backtest.world import HistoricalBacktestWorld
from pivot.environments.interactive_market.config import InteractiveMarketConfig
from pivot.environments.interactive_market.world import InteractiveMarketWorld
from pivot.evaluation.paired import PairedEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="PIVOT F0/F1/F2 virtual finance fixture")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/raw/e6-finance"))
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    finance = FinanceConfig(**payload.get("finance", {}))
    seeds = [int(seed) for seed in payload.get("seeds", [1, 2, 3])]
    incumbent = Policy.from_mapping({"intensity": 0.2, "position_size": 0.2})
    candidate = Policy.from_mapping({"intensity": 0.6, "position_size": 0.6})
    transition = PolicyTransition(
        incumbent=incumbent,
        candidate=candidate,
        round_id=0,
        candidate_index=0,
        improvement_operator="typed_finance_fixture",
        edit_type="intensity_and_position_size",
        proxy_world_id="F0-backtest",
        high_fidelity_world_id="F2-interactive-market",
        config_id=finance.fixture_version,
    )
    rows: list[dict[str, object]] = []
    f0 = HistoricalBacktestWorld(finance)
    f1 = ExecutionReplayWorld(finance)
    for seed in seeds:
        contexts = [RolloutContext(seed=seed, scenario_id=f"finance-{seed}")]
        f0_delta = PairedEvaluator(f0).evaluate(transition, contexts).delta
        f1_delta = PairedEvaluator(f1).evaluate(transition, contexts).delta
        for participation in payload.get("participation_rates", [0.0]):
            f2 = InteractiveMarketWorld(InteractiveMarketConfig(finance=finance, participation_rate=float(participation)))
            f2_delta = PairedEvaluator(f2, mode="actor").evaluate(transition, contexts).delta
            rows.append({"seed": seed, "participation_rate": participation, "delta_f0": f0_delta, "delta_f1": f1_delta, "delta_f2": f2_delta, "virtual_fills": True})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "finance_actor.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    (args.output / "provenance.json").write_text(json.dumps({"finance_config": asdict(finance), "seeds": seeds, "virtual_fills": True}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"E6 rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
