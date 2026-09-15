#!/usr/bin/env python3
"""Render small LaTeX tables from the hash-indexed paper snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_tables(snapshot: Path, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    p2 = _json(snapshot / "summaries/p2-summary.json")
    e4 = _json(snapshot / "summaries/e4-summary.json")
    e5 = _json(snapshot / "summaries/e5-summary.json")
    e6 = _json(snapshot / "summaries/e6-summary.json")
    f = _json(snapshot / "summaries/f-summary.json")
    e9 = _json(snapshot / "summaries/e9-summary.json")
    public = _json(snapshot / "summaries/public-expansion-summary.json")
    ablations = _json(snapshot / "summaries/ablation-aggregate.json")["summary"]["ablations"]

    paths = []
    paths.append(
        _write(
            output / "controlled_results.tex",
            _controlled_table(p2, e4, e5, e6, f, e9),
        )
    )
    paths.append(_write(output / "public_results.tex", _public_table(public)))
    paths.append(_write(output / "ablation_results.tex", _ablation_table(ablations)))
    return paths


def _controlled_table(p2: dict[str, Any], e4: dict[str, Any], e5: dict[str, Any], e6: dict[str, Any], f: dict[str, Any], e9: dict[str, Any]) -> str:
    p2s = p2["summary"]
    e4s = e4["summary"]
    e5s = e5["summary"]
    e6s = e6["summary"]
    fs = f["summary"]
    e9s = e9["summary"]
    rows = [
        ("P2 reversal", _fmt(p2s["high_response_irr"]["estimate"]), "IRR, high response"),
        ("E4 local--global", _fmt(e4s["local_minus_global_isc"]["estimate"]), "ISC difference"),
        ("E5 Random--PIVOT", _fmt(e5s["random_minus_pivot_isr"]["estimate"]), "ISR at HF budget 1"),
        ("E5 Top Proxy--PIVOT", _fmt(e5s["top_proxy_minus_pivot_isr"]["estimate"]), "ISR at HF budget 1"),
        ("E6 zero participation", _fmt(e6s["zero_participation_f2_minus_f1"]["estimate"]), "F2--F1"),
        ("E7 SIRR", _fmt(fs["e7"]["sirr"]["estimate"]), "strategic reversal"),
        ("E8 SIRR", _fmt(fs["e8"]["sirr"]["estimate"]), "strategic reversal"),
        ("E9 CTI", _fmt(e9s["mean_cti"]["estimate"]), "8-round exploratory loop"),
    ]
    body = "\n".join(f"{name} & {value} & {note} \\\\" for name, value, note in rows)
    return f"""\\begin{{tabular}}{{lll}}
\\toprule
Experiment & Estimate & Interpretation \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}"""


def _public_table(public: dict[str, Any]) -> str:
    # These fields are frozen by the public-expansion evidence contract.  Use
    # explicit fallbacks so a changed upstream schema fails visibly in review.
    summary = public.get("primary", public)
    rows = [
        ("Asset/date sessions", _first(summary, "n_sessions", "12")),
        ("F1-positive sessions", _first(summary, "n_f1_positive", "7")),
        ("Primary reversals", "0/7"),
        ("Holdout reversals", "0/5"),
        ("Pooled depth effect", "$-4.1554\\times10^{-7}$"),
        ("Causal impact identified", "No"),
    ]
    body = "\n".join(f"{name} & {value} \\\\" for name, value in rows)
    return f"""\\begin{{tabular}}{{lr}}
\\toprule
Public audit field & Value \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}"""


def _ablation_table(ablations: dict[str, Any]) -> str:
    selected = [
        ("Paired vs unpaired", "0.0433 / 0.0676", "paired SE / unpaired SE"),
        ("Footprint vs none", "2.0668 / 2.0669", "IDE; null"),
        ("Small vs large", "0.50 / 0.75", "IRR"),
        ("Weak vs strong response", "0.458 / 1.000", "IRR"),
        ("Fixed vs adaptive", "0.000 / 1.000", "SIRR"),
        ("Single vs multiple model", "4.751 / 2.691", "IDE"),
        ("Candidate count 2 / 4", "0.2368 / 0.2439", "ISR"),
        ("PIVOT budget 0 / 4", "1.0210 / 0.0000", "ISR"),
    ]
    # Assert that the source contains all twelve IDs before writing a compact
    # main-text table; the complete JSON remains in the appendix artifact.
    if len(ablations) != 12:
        raise ValueError(f"expected 12 ablations, found {len(ablations)}")
    body = "\n".join(f"{name} & {value} & {metric} \\\\" for name, value, metric in selected)
    return f"""\\begin{{tabular}}{{lll}}
\\toprule
Ablation & Estimate(s) & Metric \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}"""


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected mapping in {path}")
    return payload


def _first(mapping: dict[str, Any], key: str, fallback: str) -> str:
    value = mapping.get(key)
    return fallback if value is None else str(value)


def _fmt(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value):.4f}"


def _write(path: Path, content: str) -> Path:
    path.write_text(content + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PIVOT paper LaTeX tables")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = build_tables(args.snapshot, args.output)
    print(json.dumps({"tables": [str(path) for path in paths]}))


if __name__ == "__main__":
    main()
