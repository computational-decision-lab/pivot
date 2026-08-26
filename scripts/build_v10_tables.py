#!/usr/bin/env python3
"""Build compact, semantic LaTeX tables from the frozen evidence.

The table writer intentionally reads result artifacts and never reruns an
experiment.  Internal run identifiers stay in the machine-readable source;
the paper-facing tables use the scientific names of the estimands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected object in {path}")
    return payload


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "--"
    number = float(value)
    if abs(number) < 1e-3 and number != 0:
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def _interval(low: Any, high: Any) -> str:
    return f"[{_fmt(low)}, {_fmt(high)}]".replace("[ ", "[")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def build(root: Path, output: Path) -> list[Path]:
    e2 = _load(root / "results/v9/e2c-confirmatory/scientific_decision.json")["metrics"]
    e3 = _load(root / "results/v9/e3c-confirmatory/scientific_decision.json")["metrics"]
    e4 = _load(root / "results/v9/e4c-confirmatory/scientific_decision.json")["metrics"]
    e7 = _load(root / "results/v9/e7c-confirmatory/scientific_decision.json")["metrics"]
    finance = _load(root / "results/raw/e6-public-calibration/summary.json")
    finance_expansion = _load(root / "paper/snapshot/summaries/public-expansion-summary.json")

    rows = [
        (
            "Operator shift",
            _fmt(e2["shift_effect_mean"]),
            _interval(e2["shift_effect_ci_low"], e2["shift_effect_ci_high"]),
            "IDE contrast; paired transition",
        ),
        (
            "Closed-loop validation",
            _fmt(e3["effects"]["pivot_voi_minus_proxy_only"]["mean"]),
            _interval(
                e3["effects"]["pivot_voi_minus_proxy_only"]["ci_low"],
                e3["effects"]["pivot_voi_minus_proxy_only"]["ci_high"],
            ),
            "CISR reduction; trajectory seed",
        ),
        (
            "OOD evaluator contrast",
            _fmt(e4["transition_minus_global_isc_mean"]),
            _interval(
                e4["transition_minus_global_isc_ci_low"], e4["transition_minus_global_isc_ci_high"]
            ),
            "transition ISC minus global ISC; null",
        ),
        (
            "Strategic adaptation",
            _fmt(e7["adaptive_effect_mean"]),
            _interval(e7["adaptive_effect_ci_low"], e7["adaptive_effect_ci_high"]),
            "strategic minus actor; opponent cluster",
        ),
    ]
    body = "\n".join(
        f"{name} & {estimate} & {interval} & {note} \\\\" for name, estimate, interval, note in rows
    )
    controlled = (
        r"""\begin{tabular}{llll}
\toprule
Evidence layer & Estimate & 95\% interval & Unit / interpretation \\
\midrule
"""
        + body
        + r"""
\bottomrule
\end{tabular}"""
    )
    controlled_path = output / "controlled_results.tex"
    _write(controlled_path, controlled)

    public_rows = [
        ("Initial single-asset sessions", str(finance.get("n_sessions", "--"))),
        ("Expanded primary sessions", str(finance_expansion.get("n_primary_sessions", "--"))),
        (
            "Expanded primary reversals / eligible",
            f"0/{finance_expansion.get('n_f1_positive_sessions', '--')}",
        ),
        (
            "Expanded holdout reversals / eligible",
            f"0/{finance_expansion.get('holdout', {}).get('n_f1_positive_sessions', '--')}",
        ),
        (
            "Expanded pooled depth effect",
            _fmt(finance_expansion.get("pooled_depth_mechanical_effect", {}).get("estimate"), 2),
        ),
        ("Causal impact identified", "No"),
    ]
    public_body = "\n".join(f"{name} & {value} \\\\" for name, value in public_rows)
    public = (
        r"""\begin{tabular}{lr}
\toprule
Public-data boundary & Value \\
\midrule
"""
        + public_body
        + r"""
\bottomrule
\end{tabular}"""
    )
    public_path = output / "public_results.tex"
    _write(public_path, public)

    # A small table of registered comparison boundaries; detailed cells remain
    # in the CSV/JSON artifacts and generated figures.
    e3_effects = e3["effects"]
    ablation_rows = [
        (
            "PIVOT-VOI vs Proxy",
            _fmt(e3_effects["pivot_voi_minus_proxy_only"]["mean"]),
            "CISR reduction",
        ),
        (
            "PIVOT-VOI vs Global-VOI",
            _fmt(e3_effects["pivot_voi_minus_global_voi"]["mean"]),
            "CISR reduction",
        ),
        (
            "PIVOT-VOI vs Paired LUCB",
            _fmt(e3_effects["pivot_voi_minus_paired_lucb"]["mean"]),
            "CISR reduction",
        ),
        (
            "OOD transition minus global",
            _fmt(e4["transition_minus_global_isc_mean"]),
            "ISC; powered null",
        ),
        (
            "Finance participation boundary",
            _fmt(finance.get("target_f2_depth_minus_f1_mean"), 2),
            "observational effect",
        ),
    ]
    ablation_body = "\n".join(
        f"{name} & {estimate} & {metric} \\\\" for name, estimate, metric in ablation_rows
    )
    ablation = (
        r"""\begin{tabular}{lll}
\toprule
Registered comparison & Estimate & Metric / scope \\
\midrule
"""
        + ablation_body
        + r"""
\bottomrule
\end{tabular}"""
    )
    ablation_path = output / "ablation_results.tex"
    _write(ablation_path, ablation)
    manifest = {
        "schema_version": "v10",
        "source_paths": [
            "results/v9/e2c-confirmatory/scientific_decision.json",
            "results/v9/e3c-confirmatory/scientific_decision.json",
            "results/v9/e4c-confirmatory/scientific_decision.json",
            "results/v9/e7c-confirmatory/scientific_decision.json",
            "results/raw/e6-public-calibration/summary.json",
            "paper/snapshot/summaries/public-expansion-summary.json",
        ],
        "outputs": [
            str(path.relative_to(root)) for path in (controlled_path, public_path, ablation_path)
        ],
    }
    _write(output / "v10_table_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    return [controlled_path, public_path, ablation_path, output / "v10_table_manifest.json"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V10 semantic paper tables")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("paper/tables"))
    args = parser.parse_args()
    paths = build(
        args.root.resolve(),
        (args.root / args.output).resolve()
        if not args.output.is_absolute()
        else args.output.resolve(),
    )
    print(json.dumps({"tables": [str(path) for path in paths]}, sort_keys=True))


if __name__ == "__main__":
    main()
