#!/usr/bin/env python3
"""Materialize the V15 publication figure aliases used by paper/main.tex."""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

FIGURES = (
    "fig1_improvement_reversal", "fig2_operator_shift", "fig3_pivot_voi",
    "fig4_evidence_efficiency", "fig5_closed_loop", "figA_response_footprint",
    "figB_learned_ood_null", "figC_posterior_robustness",
    "figD_strategic_distribution", "figE_finance_boundary",
)

def build(root: Path) -> dict[str, object]:
    root = root.resolve()
    source = root / "figures/v15"
    target = root / "paper/figures/release"
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for figure in FIGURES:
        directory = source / figure
        for suffix in ("pdf", "png"):
            src = directory / f"figure.{suffix}"
            if not src.is_file():
                raise FileNotFoundError(src)
            dst = target / f"{figure}.{suffix}"
            shutil.copyfile(src, dst)
            copied.append(dst.relative_to(root).as_posix())
    architecture = root / "paper/snapshot/figures/fig3_pivot_architecture.pdf"
    if architecture.is_file():
        dst = target / "fig3_pivot_architecture.pdf"
        shutil.copyfile(architecture, dst)
        copied.append(dst.relative_to(root).as_posix())
    return {"copied": copied, "count": len(copied)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(build(args.root))
