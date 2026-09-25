#!/usr/bin/env python3
"""Replay the Table 1 selectors using their saved posterior and observation bank."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def audit(evidence: Path) -> dict:
    source = evidence / "source/core"
    sys.path.insert(0, str(source))
    import pivot_v2
    spec = importlib.util.spec_from_file_location("comparison_analysis", source / "analyze_v4.py")
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    report = {"scope": "First-cohort Table 1; recomputed decisions, not new native simulator runs", "cohorts": {}}
    for environment, horizon in [("leduc", 8), ("kuhn", 8), ("melting", 32)]:
        base = evidence / "core" / environment
        summary = json.loads((base / "analysis/summary.json").read_text())
        rows = json.loads((base / "analysis/scored_decisions.json").read_text())
        fits = json.loads((base / "analysis/posterior_v2_spec.json").read_text())
        posterior_spec = pivot_v2.PosteriorV2Spec(**fits[str(horizon)])
        protocol_path = base / "confirm/protocol.json"
        if not protocol_path.exists():
            protocol_path = base / "calibration/protocol.json"
        protocol = json.loads(protocol_path.read_text())
        analysis.QUERY_COST = None
        analysis.CAPS = (96, 192, 384)
        analysis.PRIMARY_CAP = 192
        analysis.set_candidate_geometry(protocol)
        # The compact evidence retains summaries but omits native runner status
        # markers. Read the same fields as the historical load_cohort function.
        worlds = {}
        for path in sorted((base / "confirm").glob("seed_*/summary.json")):
            saved = json.loads(path.read_text())
            count = len(saved["proxy_deltas"])
            values = lambda field: np.array([saved[field][str(horizon)][str(i)] for i in range(count)])
            worlds[saved["seed"]] = {
                "root": saved["seed"],
                "proxy": np.array([saved["proxy_deltas"][str(i)] for i in range(count)]),
                "per_h": {horizon: {"S": values("selection_bank"),
                                    "cost": float(horizon + 32) if analysis.QUERY_COST is None else analysis.QUERY_COST}},
                "_audit": {horizon: {"label": values("audit_gains")}},
            }
        primary = [r for r in rows if r["adaptation"] == horizon and r["cap"] == summary["primary_cap"] and r["method"] in {"pivot_kg", "uniform_v2"}]
        if len(primary) != 60:
            raise ValueError(f"Expected 60 matched decision rows for {environment}, got {len(primary)}")
        worst = 0.0
        selected = {}
        for row in primary:
            world = worlds[row["root"]]
            ph = world["per_h"][horizon]
            post = pivot_v2.make_posterior(posterior_spec, world["proxy"])
            got = pivot_v2.run_selector(post, world["proxy"], lambda j: ph["S"][j],
                                        method=row["method"], budget=2,
                                        costs=np.full(len(world["proxy"]), ph["cost"]), seed=world["root"])
            if got["selected"] != row["selected"] or got["queried"] != row["queried"]:
                raise ValueError(f"Decision mismatch: {environment} {row['root']} {row['method']}")
            if got["hf_queries"] != row["hf_queries"] or got["hf_queries"] != 2:
                raise ValueError("Unequal query budget")
            error = float(np.max(np.abs(np.asarray(got["estimates"]) - row["estimates"])))
            worst = max(worst, error)
            truth = np.r_[world["_audit"][horizon]["label"], 0.0]
            gain = float(truth[got["selected"]])
            if abs(gain - row["gain"]) > 1e-12 or abs(float(truth.max()) - gain - row["isr"]) > 1e-12:
                raise ValueError("Saved audit score mismatch")
            selected.setdefault(row["root"], {})[row["method"]] = row
        if worst > 1e-12 or len(selected) != 30:
            raise ValueError("Posterior estimate or root-count mismatch")
        contrast = float(np.mean([r["uniform_v2"]["isr"] - r["pivot_kg"]["isr"] for r in selected.values()]))
        expected = summary["hypotheses"]["H2_long_primary_gain_minus_comparator_cap192"]
        if abs(contrast - expected["mean"]) > 1e-12:
            raise ValueError("Reported contrast mismatch")
        report["cohorts"][environment] = {
            "roots": 30, "replayed_decisions": 60, "queries_each": 2,
            "max_posterior_estimate_error": worst,
            "mean_ISR": {m: float(np.mean([r[m]["isr"] for r in selected.values()])) for m in ["pivot_kg", "uniform_v2"]},
            "uniform_minus_pivot_ISR": contrast,
            "saved_root_bootstrap_CI": [expected["lo"], expected["hi"]],
            "same_selected_update_roots": sum(r["pivot_kg"]["selected"] == r["uniform_v2"]["selected"] for r in selected.values()),
            "comparison": "Same fitted posterior, observation update and terminal selection; query allocation differs.",
        }
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, default=ROOT / "evidence/paper")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = audit(args.evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
