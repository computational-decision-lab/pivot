"""Copy the verified V9 figure bundle into the self-contained paper package."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

FIGURES = (
    "fig1_v9_improvement_reversal",
    "fig2_e2c_operator_shift",
    "fig4_e3c_closed_loop",
    "fig5_e5c_budget_frontier",
    "fig6_e4c_ood",
    "fig7_e7c_strategic",
    "fig8_e5c_calibration",
)
FORMATS = ("pdf", "svg", "png", "csv", "parquet", "meta.json")


def sync(root: Path) -> list[Path]:
    source = root / "figures/v9"
    destination = root / "paper/iclr2027/figures/v9"
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for stem in FIGURES:
        for suffix in FORMATS:
            path = source / f"{stem}.{suffix}"
            if not path.is_file():
                raise FileNotFoundError(path)
            target = destination / path.name
            shutil.copyfile(path, target)
            copied.append(target)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync V9 figures into paper package")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(f"copied={len(sync(args.root.resolve()))}")


if __name__ == "__main__":
    main()
