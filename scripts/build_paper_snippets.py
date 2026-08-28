"""Paper-facing wrapper for the frozen semantic result macro generator."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# When invoked as ``python scripts/build_paper_snippets.py``, Python places the
# scripts directory (rather than the project root) first on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_v10_paper_snippets import build as _build


def _neutralize_macro_text(root: Path, text: str) -> str:
    """Remove historical iteration labels from the paper-facing macro file."""

    text = text.replace("scripts/build_v10_paper_snippets.py", "scripts/build_paper_snippets.py")
    strategic_path = root / "results/v9/e7c-confirmatory/strategic_summary.json"
    if strategic_path.is_file():
        payload = json.loads(strategic_path.read_text(encoding="utf-8"))
        rows = payload.get("by_mode", []) if isinstance(payload, dict) else []
        adaptive = {"best_response", "gradient_adaptive", "rl_evolutionary"}
        adaptive_traces = sum(
            int(row["cluster_n"])
            for row in rows
            if row.get("opponent_mode") in adaptive
        )
        all_traces = sum(int(row["cluster_n"]) for row in rows)
        text = re.sub(
            r"(\\newcommand\{\\StrategicFamilySeedTraces\}\{)[^}]+(\})",
            rf"\g<1>{adaptive_traces}\g<2>",
            text,
        )
        if r"\StrategicAllFamilySeedTraces" in text:
            text = re.sub(
                r"(\\newcommand\{\\StrategicAllFamilySeedTraces\}\{)[^}]+(\})",
                rf"\g<1>{all_traces}\g<2>",
                text,
            )
        else:
            text += f"\\newcommand{{\\StrategicAllFamilySeedTraces}}{{{all_traces}}}\n"
    return text


def build(root: Path) -> Path:
    generated = _build(root)
    neutral = root / "paper/iclr2027/results_macros.tex"
    neutral.write_text(_neutralize_macro_text(root, generated.read_text(encoding="utf-8")), encoding="utf-8")
    return neutral


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build paper-facing result macros")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps({"output": str(build(args.root.resolve()))}, sort_keys=True))
