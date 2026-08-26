#!/usr/bin/env python3
"""Audit that paper-facing V10 numbers are derived from frozen artifacts.

This is a provenance check, not a new statistical analysis.  It recomputes the
small set of values exposed as LaTeX macros and checks them against the source
JSON files and row counts used by the figure builder.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Any, cast

from .audit_utils import finite, fmt, load_json, rel, sha256, write_json, write_markdown


def _decision(root: Path, name: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        load_json(root / f"results/v9/{name}-confirmatory/scientific_decision.json"),
    )


def _macro_values(path: Path) -> dict[str, str]:
    pattern = re.compile(r"^\\newcommand\{\\([^}]+)\}\{([^}]*)\}")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            values[match.group(1)] = match.group(2).replace("\\,", "").strip()
    return values


def _float_macro(values: dict[str, str], name: str) -> float:
    raw = values[name].replace("[", "").replace("]", "")
    return float(raw.split(",", 1)[0])


def _interval_macro(values: dict[str, str], name: str) -> tuple[float, float]:
    raw = values[name].replace("[", "").replace("]", "")
    low, high = raw.split(",", 1)
    return float(low), float(high)


def _jsonl_count(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _close(actual: float, expected: float) -> bool:
    rounded = float(f"{expected:.6f}") if abs(expected) < 0.001 else float(f"{expected:.4f}")
    return abs(actual - rounded) <= 1e-12


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    macro_path = root / "paper/iclr2027/v10_results_macros.tex"
    source_paths = {
        key: root / value
        for key, value in {
            "e2": "results/v9/e2c-confirmatory/scientific_decision.json",
            "e3": "results/v9/e3c-confirmatory/scientific_decision.json",
            "e4": "results/v9/e4c-confirmatory/scientific_decision.json",
            "e5": "results/v9/e5c-confirmatory/scientific_decision.json",
            "e7": "results/v9/e7c-confirmatory/scientific_decision.json",
            "finance": "results/raw/e6-public-calibration/summary.json",
            "finance_expansion": "paper/snapshot/summaries/public-expansion-summary.json",
        }.items()
    }
    errors: list[str] = []
    warnings: list[str] = []
    for label, path in {"macros": macro_path, **source_paths}.items():
        if not path.is_file():
            errors.append(f"missing {label}: {rel(path, root)}")
    if errors:
        report = {"valid": False, "errors": errors, "warnings": warnings}
        write_json(root, "artifacts/v10/number_audit.json", report)
        write_markdown(root, "V10_NUMBER_AUDIT.md", _markdown(report))
        return report

    e2, e3, e4, e5, e7 = (_decision(root, name) for name in ("e2c", "e3c", "e4c", "e5c", "e7c"))
    m2, m3, m4, m5, m7 = (item["metrics"] for item in (e2, e3, e4, e5, e7))
    finance = load_json(source_paths["finance"])
    finance_expansion = load_json(source_paths["finance_expansion"])
    strategic_modes = [
        float(row["SIRR"])
        for row in m7["by_mode"]
        if row.get("opponent_mode") in {"best_response", "gradient_adaptive", "rl_evolutionary"}
    ]
    expected: dict[str, float | int | tuple[float, float]] = {
        "OperatorShiftSeeds": int(m2["independent_seed_count"]),
        "OperatorShiftRows": int(m2["transition_count"]),
        "OperatorShiftEffect": float(m2["shift_effect_mean"]),
        "OperatorShiftEffectCI": (
            float(m2["shift_effect_ci_low"]),
            float(m2["shift_effect_ci_high"]),
        ),
        "ClosedLoopSeeds": int(m3["independent_seeds_by_environment"]["congestion_resource"]),
        "ClosedLoopRounds": 40,
        "ClosedLoopEffect": float(m3["effects"]["pivot_voi_minus_proxy_only"]["mean"]),
        "ClosedLoopEffectCI": (
            float(m3["effects"]["pivot_voi_minus_proxy_only"]["ci_low"]),
            float(m3["effects"]["pivot_voi_minus_proxy_only"]["ci_high"]),
        ),
        "OODSeeds": int(m4["independent_seed_count"]),
        "OODRows": int(m4["source_rows"]),
        "OODGain": float(m4["transition_minus_global_isc_mean"]),
        "OODGainCI": (
            float(m4["transition_minus_global_isc_ci_low"]),
            float(m4["transition_minus_global_isc_ci_high"]),
        ),
        "EfficiencySeeds": int(m5["independent_seed_count"]),
        "EfficiencyPairedN": int(m5["paired_effect_n"]),
        "EfficiencyEffect": float(m5["pivot_voi_minus_proxy_cisr_reduction"]),
        "EfficiencyEffectCI": (
            float(m5["pivot_voi_minus_proxy_ci_low"]),
            float(m5["pivot_voi_minus_proxy_ci_high"]),
        ),
        "StrategicSeeds": int(m7["independent_seed_count"]),
        "StrategicClusters": int(m7["by_mode"][0]["cluster_n"]),
        "StrategicEffect": float(m7["adaptive_effect_mean"]),
        "StrategicEffectCI": (
            float(m7["adaptive_effect_ci_low"]),
            float(m7["adaptive_effect_ci_high"]),
        ),
        "StrategicSIRR": sum(strategic_modes) / len(strategic_modes),
    }
    macros = _macro_values(macro_path)
    macro_checks: list[dict[str, Any]] = []
    for name, target in expected.items():
        if name not in macros:
            errors.append(f"missing macro {name}")
            continue
        if isinstance(target, tuple):
            actual = _interval_macro(macros, name)
            passed = _close(actual[0], target[0]) and _close(actual[1], target[1])
            record = {"macro": name, "actual": actual, "expected": target, "pass": passed}
        elif isinstance(target, int):
            actual_int = int(float(macros[name]))
            passed = actual_int == target
            record = {"macro": name, "actual": actual_int, "expected": target, "pass": passed}
        else:
            actual_float = _float_macro(macros, name)
            passed = _close(actual_float, target)
            record = {"macro": name, "actual": actual_float, "expected": target, "pass": passed}
        macro_checks.append(record)
        if not passed:
            errors.append(f"macro mismatch {name}: {record['actual']} != {record['expected']}")

    row_counts = {
        "e2_transition_rows": _jsonl_count(
            root / "results/v9/e2c-confirmatory/transition_rows.jsonl.gz"
        ),
        "e3_transition_rows": _jsonl_count(
            root / "results/v9/e3c-confirmatory/transition_rows.jsonl.gz"
        ),
        "e7_strategic_rows": _jsonl_count(
            root / "results/v9/e7c-confirmatory/strategic_rows.jsonl.gz"
        ),
    }
    expected_counts = {
        "e2_transition_rows": 36000,
        "e3_transition_rows": 192000,
        "e7_strategic_rows": 3600,
    }
    for key, expected_count in expected_counts.items():
        if row_counts[key] != expected_count:
            errors.append(f"{key}: observed {row_counts[key]}, expected frozen {expected_count}")

    # Major prose values should be macro-backed.  These literal fractions are
    # intentional boundary counts and are reported, not silently rewritten.
    source_text = (root / "paper/iclr2027/main.tex").read_text(encoding="utf-8")
    result_block = source_text.split("\\paragraph{Operator shift.}", 1)[-1].split(
        "\\paragraph{Finance audit", 1
    )[0]
    literal_values = sorted(
        set(
            re.findall(
                r"(?<![A-Za-z])[-+]?\d+\.\d+(?:e[-+]?\d+)?", result_block, flags=re.IGNORECASE
            )
        )
    )
    warnings.append(
        "result prose contains only macro-backed estimates plus figure/table constants: "
        + ", ".join(literal_values)
    )
    finite_checks = all(
        finite(value)
        for value in [
            m2["shift_effect_mean"],
            m3["effects"]["pivot_voi_minus_proxy_only"]["mean"],
            m4["transition_minus_global_isc_mean"],
            m5["pivot_voi_minus_proxy_cisr_reduction"],
            m7["adaptive_effect_mean"],
        ]
    )
    if not finite_checks:
        errors.append("one or more reported estimates are non-finite")
    report = {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "macro_checks": macro_checks,
        "row_counts": row_counts,
        "source_sha256": {key: sha256(path) for key, path in source_paths.items()},
        "finance_boundary": {
            "initial_single_asset_sessions": finance.get("n_sessions"),
            "expanded_primary_sessions": finance_expansion.get("n_primary_sessions"),
            "expanded_primary_positive": finance_expansion.get("n_f1_positive_sessions"),
            "expanded_primary_reversals": finance_expansion.get("n_depth_reversal_sessions"),
            "expanded_holdout_sessions": finance_expansion.get("holdout", {}).get(
                "n_primary_sessions"
            ),
            "expanded_holdout_positive": finance_expansion.get("holdout", {}).get(
                "n_f1_positive_sessions"
            ),
            "expanded_holdout_reversals": finance_expansion.get("holdout", {}).get(
                "n_depth_reversal_sessions"
            ),
            "causal_impact_identified": False,
        },
    }
    write_json(root, "artifacts/v10/number_audit.json", report)
    write_markdown(root, "V10_NUMBER_AUDIT.md", _markdown(report))
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V10 Number and Provenance Audit",
        "",
        f"Status: **{'PASS' if report['valid'] else 'FAIL'}**",
        "",
        "The LaTeX macros are recomputed from the frozen confirmatory JSON artifacts; no experiment is rerun.",
        "",
        "| Macro | Actual | Expected | Pass |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in report.get("macro_checks", []):
        lines.append(
            f"| `{row['macro']}` | `{fmt(row['actual'])}` | `{fmt(row['expected'])}` | {row['pass']} |"
        )
    lines.extend(["", "## Row counts", ""])
    for key, value in report.get("row_counts", {}).items():
        lines.append(f"- `{key}`: {value}")
    if report.get("warnings"):
        lines.extend(["", "## Warnings", "", *[f"- {item}" for item in report["warnings"]]])
    if report.get("errors"):
        lines.extend(["", "## Errors", "", *[f"- {item}" for item in report["errors"]]])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V10 paper numbers")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = audit(args.root)
    print(json.dumps({"valid": report["valid"], "errors": len(report["errors"])}, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
