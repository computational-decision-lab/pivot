#!/usr/bin/env python3
"""Trace V10 metrics to code and document aggregation/scale contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from .audit_utils import load_json, sha256, write_json, write_markdown


def _method(metrics: dict[str, Any], environment: str, method: str) -> dict[str, Any]:
    for row in metrics.get("by_method_environment", []):
        if row.get("environment_id") == environment and row.get("method") == method:
            return cast(dict[str, Any], row)
    raise KeyError(f"missing {environment}/{method}")


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    code_paths = {
        "canonical_metrics": root / "src/pivot/v9/statistics.py",
        "closed_loop": root / "experiments/v9/e3c_closed_loop.py",
        "budget_frontier": root / "experiments/v9/e5c_efficiency.py",
        "manuscript": root / "paper/iclr2027/main.tex",
    }
    errors: list[str] = []
    for name, path in code_paths.items():
        if not path.is_file():
            errors.append(f"missing {name}: {path.relative_to(root)}")
    snippets = {
        "canonical_metrics": ("abs(proxy - true)", "max(available) - selected[0]"),
        "closed_loop": ('row["CISR"] =', 'row["CTI"] =', "cti +=", "cisr +="),
        "budget_frontier": ('"CISR": true_best - selected_true', '"CTI": selected_true'),
        "manuscript": ("CISR}_T=\\sum_t", "native", "fraction of excess regret removed"),
    }
    for name, required in snippets.items():
        if not code_paths[name].is_file():
            continue
        text = code_paths[name].read_text(encoding="utf-8")
        for token in required:
            if token not in text:
                errors.append(f"{name}: missing metric contract token {token}")
    e3 = load_json(root / "results/v9/e3c-confirmatory/scientific_decision.json")["metrics"]
    congestion_proxy = float(_method(e3, "congestion_resource", "proxy_only")["CISR"]["mean"])
    performative_proxy = float(_method(e3, "performative_control", "proxy_only")["CISR"]["mean"])
    scale_ratio = performative_proxy / max(congestion_proxy, 1e-12)
    if scale_ratio < 10:
        errors.append("controlled worlds do not exhibit the expected native-scale separation")
    report = {
        "valid": not errors,
        "errors": errors,
        "metric_contracts": {
            "IDE": {
                "definition": "mean absolute difference |Delta_V - Delta_*|",
                "aggregation": "transition rows within the declared operator law",
                "unit": "native environment reward",
            },
            "ISC": {
                "definition": "sign agreement after excluding proxy/true ties at tau",
                "aggregation": "comparable transition rows",
                "unit": "probability",
            },
            "IRR": {
                "definition": "P(Delta_* < -tau | Delta_V > tau)",
                "aggregation": "proxy-positive transition rows",
                "unit": "conditional probability",
            },
            "ISR": {
                "definition": "max_j Delta_*j - Delta_*selected",
                "aggregation": "one incumbent/candidate decision set",
                "unit": "native environment reward",
            },
            "CTI": {
                "definition": "sum_t Delta_*selected,t",
                "aggregation": "trajectory in the closed loop; one selected delta when T=1",
                "unit": "native environment reward",
            },
            "CISR": {
                "definition": "sum_t ISR_t",
                "aggregation": "trajectory for repeated improvement; equals ISR in the one-set budget study",
                "unit": "native environment reward",
            },
            "FER": {
                "definition": "1 - (CISR_method-CISR_allHF)/(CISR_proxy-CISR_allHF)",
                "aggregation": "only matched environment, K, horizon, and cost protocol",
                "unit": "dimensionless fraction of excess regret removed",
                "withhold_when": "denominator is nonpositive/unstable or all-HF is not comparable",
            },
        },
        "scale_diagnostic": {
            "congestion_proxy_cisr": congestion_proxy,
            "performative_proxy_cisr": performative_proxy,
            "native_scale_ratio": scale_ratio,
            "cross_environment_raw_comparison_allowed": False,
        },
        "inference_units": {
            "operator_shift": "independent seed; cell summaries preserve transition counts",
            "closed_loop": "trajectory seed",
            "ood": "registered held-out unit and fitted family; descriptive split spans are not pseudo-CIs",
            "efficiency": "paired environment x K x seed decision set",
            "strategic": "opponent-seed cluster",
            "finance": "session-level observational diagnostic",
        },
        "candidate_budget_contract": {
            "closed_loop": {"K": 8, "rounds": 40, "hf_budget_per_round": 2},
            "frontier": {"K": [4, 8, 16], "budgets": [0, 1, 2, 4, 8, 16]},
            "connected_lines": "only fixed K and environment; heterogeneous cells are never connected",
        },
        "source_sha256": {
            name: sha256(path) for name, path in code_paths.items() if path.is_file()
        },
        "source_locations": {
            "IDE_ISC_IRR_ISR": "src/pivot/v9/statistics.py:58",
            "closed_loop_CTI_CISR": "experiments/v9/e3c_closed_loop.py:249",
            "budget_set_CTI_CISR": "experiments/v9/e5c_efficiency.py:67",
        },
    }
    write_json(root, "artifacts/v10/metric_scale_audit.json", report)
    write_markdown(root, "V10_METRIC_SCALE_AUDIT.md", _markdown(report))
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V10 Metric and Scale Audit",
        "",
        f"Status: **{'PASS' if report['valid'] else 'FAIL'}**",
        "",
        "| Metric | Definition | Aggregation | Unit |",
        "| --- | --- | --- | --- |",
    ]
    for name, row in report.get("metric_contracts", {}).items():
        lines.append(f"| {name} | {row['definition']} | {row['aggregation']} | {row['unit']} |")
    scale = report.get("scale_diagnostic", {})
    lines.extend(
        [
            "",
            "## Scale boundary",
            "",
            f"The Proxy Only cumulative regret means are `{scale.get('congestion_proxy_cisr')}` and `{scale.get('performative_proxy_cisr')}` in two different native reward systems (ratio `{scale.get('native_scale_ratio'):.1f}` where available). Their raw magnitudes are not compared across environments.",
            "",
            "Closed-loop CTI/CISR sum 40 replacement decisions. The efficiency frontier uses one candidate set, so its stored CISR is one-set ISR. FER is computed only within a matched cell and is withheld when its oracle denominator is not meaningful.",
            "",
            "## Inference units",
            "",
        ]
    )
    for name, unit in report.get("inference_units", {}).items():
        lines.append(f"- `{name}`: {unit}")
    lines.extend(["", "## Code trace", ""])
    for name, location in report.get("source_locations", {}).items():
        lines.append(f"- `{name}`: `{location}`")
    if report.get("errors"):
        lines.extend(["", "## Errors", "", *[f"- {item}" for item in report["errors"]]])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V10 metric scales")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = audit(args.root)
    print(json.dumps({"valid": report["valid"], "errors": len(report["errors"])}, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
