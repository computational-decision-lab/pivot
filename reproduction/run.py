#!/usr/bin/env python3
"""Bounded reproduction entry point for the paper artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "paper"
EXPERIMENTS = (
    "controlled", "kuhn", "leduc", "leduc_v4", "meltingpot",
    "highway", "highway_redesign", "metadrive",
)
REPORT_KEYS = {
    "controlled": ("all_hf_reference",),
    "kuhn": ("core",),
    "leduc": ("core",),
    "leduc_v4": ("leduc_v4",),
    "meltingpot": ("core",),
    "highway": (),
    "highway_redesign": ("highway_redesign",),
    "metadrive": ("metadrive",),
}


def run(command: list[str], *, env_extra: dict[str, str] | None = None,
        cwd: Path = ROOT) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True,
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env_extra or {})})


def core_cohort_summary(experiment: str) -> dict:
    name = "melting" if experiment == "meltingpot" else experiment
    horizon = "32" if name == "melting" else "8"
    files = sorted((EVIDENCE / "core" / name / "confirm").glob("seed_*/summary.json"))
    if len(files) != 30:
        raise ValueError(f"expected 30 saved {name} roots, found {len(files)}")
    disjoint = tied = 0
    for path in files:
        row = json.loads(path.read_text())
        ids = sorted(row["proxy_deltas"], key=int)
        proxy = [0.0] + [row["proxy_deltas"][index] for index in ids]
        truth = [0.0] + [row["audit_gains"][horizon][index] for index in ids]
        pmax, tmax = max(proxy), max(truth)
        proxy_best = {i for i, x in enumerate(proxy) if abs(x - pmax) <= 1e-12}
        truth_best = {i for i, x in enumerate(truth) if abs(x - tmax) <= 1e-12}
        disjoint += not bool(proxy_best & truth_best)
        tied += len(truth_best) > 1
    return {"roots": len(files), "disjoint_optimal_sets": disjoint,
            "tied_deployment_optima": tied, "horizon": int(horizon)}


def run_full(experiment: str, python: str, output: Path) -> None:
    """Execute a full cohort from frozen protocol and source."""
    if experiment in ("kuhn", "leduc", "leduc_v4"):
        if experiment == "leduc_v4":
            source = EVIDENCE / "source" / "leduc_v4" / "code"
            cohort = EVIDENCE / "leduc_v4"
            panel = source / "openspiel_v4_panel.py"
            analyzer = source / "analyze_v5.py"
        else:
            source = EVIDENCE / "source" / "core"
            cohort = EVIDENCE / "core" / experiment
            panel = source / "openspiel_v2" / "openspiel_v2_panel.py"
            analyzer = source / "analyze_v4.py"
        for phase in ("calibration", "confirm"):
            protocol = cohort / phase / "protocol.json"
            if not protocol.is_file():
                raise FileNotFoundError(f"frozen protocol missing: {protocol}")
            run([python, str(panel), "--protocol", str(protocol), "--output",
                 str(output / phase), "--workers", "1"])
        run([python, str(analyzer), "--mode", "confirm", "--protocol",
             str(cohort / "confirm" / "protocol.json"), "--calibration",
             str(output / "calibration"), "--test", str(output / "confirm"),
             "--output", str(output / "analysis")])
        return
    if experiment in ("highway", "highway_redesign"):
        launcher = EVIDENCE / "source" / experiment / "run.py"
        if not launcher.is_file():
            raise FileNotFoundError(f"frozen Highway launcher missing: {launcher}")
        command = [python, str(launcher), "--output", str(output)]
        if experiment == "highway":
            command.extend(["--cohort", "replication"])
        run(command)
        return
    if experiment == "metadrive":
        protocols = EVIDENCE / "metadrive" / "formal_protocols"
        runner = EVIDENCE / "metadrive" / "code" / "run_benchmark.py"
        analyzer = EVIDENCE / "metadrive" / "analysis_tools" / "analyze_formal.py"
        env = {"PYTHONPATH": os.pathsep.join((
            str(EVIDENCE / "metadrive" / "code"),
            str(EVIDENCE / "metadrive" / "analysis_tools"),
            os.environ.get("PYTHONPATH", "")))}
        for phase in ("calibration", "confirmation"):
            protocol = protocols / f"{phase}_protocol.json"
            run([python, str(runner), "--protocol", str(protocol), "--out",
                 str(output / phase), "--workers", "1", "--one-episode-per-process"])
            run([python, str(analyzer), "--mode", phase, "--calibration",
                 str(output / "calibration"), "--out", str(output / phase),
                 "--protocol", str(protocol)], env_extra=env)
        return
    if experiment == "meltingpot":
        raise RuntimeError(
            "full Melting Pot rerun needs the frozen specialist model and crossbench "
            "root required by evidence/paper/source/core/run_melting_v4.py, plus the "
            "original frozen confirmation protocol; these are not packaged as portable "
            "inputs. See docs/protocol-chronology.md for recorded hashes."
        )
    if experiment == "controlled":
        source = EVIDENCE / "source" / "controlled"
        profiles = ROOT / "configs" / "v9" / "profiles.yaml"
        if not profiles.is_file():
            raise FileNotFoundError(f"controlled confirmatory profiles missing: {profiles}")
        for study in ("e2c", "e3c", "e4c", "e5c", "e7c"):
            packaged = EVIDENCE / "controlled" / "configs" / f"{study}.yaml"
            used = ROOT / "configs" / "v9" / f"{study}.yaml"
            if not used.is_file() or hashlib.sha256(used.read_bytes()).digest() != hashlib.sha256(packaged.read_bytes()).digest():
                raise ValueError(f"controlled config differs from packaged input: {study}")
        env = {"PYTHONPATH": os.pathsep.join((str(source), str(source / "src"),
                                               os.environ.get("PYTHONPATH", "")))}
        for study in ("e2c", "e3c", "e4c", "e5c", "e7c"):
            run([python, "-m", "experiments.v9.run", "--experiment", study,
                 "--profile", "confirmatory", "--output", str(output / study),
                 "--root", str(ROOT)], env_extra=env, cwd=source)
        return
    raise ValueError(experiment)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "analyze", "figures", "smoke", "full"))
    parser.add_argument("--experiment", choices=EXPERIMENTS,
                        help="select one cohort for analysis, smoke, or full (figures always covers all)")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter for each reproduction step")
    parser.add_argument("--output", type=Path, required=True,
                        help="directory for newly generated reports and figures")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == EVIDENCE or EVIDENCE in output.parents:
        parser.error("--output must not be inside immutable evidence/paper")
    if args.command == "full" and not args.experiment:
        parser.error("full requires --experiment to identify a registered cohort")
    if args.command == "full" and output.exists():
        parser.error("full requires a new output directory to preserve existing runs")
    if args.command != "full":
        output.mkdir(parents=True, exist_ok=True)

    try:
        if args.command == "full":
            run_full(args.experiment, args.python, output)
            return 0
        if args.command in ("verify", "analyze"):
            report_path = output / "saved-evidence-verification.json"
            run([args.python, str(ROOT / "reproduction" / "verify_saved_evidence.py"),
                 "--evidence", str(EVIDENCE), "--output", str(report_path)])
            if args.command == "analyze":
                report = json.loads(report_path.read_text())
                comparison_path = output / "table1-comparison-audit.json"
                run([args.python, str(ROOT / "reproduction" / "audit_comparisons.py"),
                     "--evidence", str(EVIDENCE), "--output", str(comparison_path)])
                comparisons = json.loads(comparison_path.read_text())["cohorts"]
                keys = REPORT_KEYS[args.experiment] if args.experiment else (
                    "core", "leduc_v4", "metadrive", "highway_redesign", "all_hf_reference")
                selected = {key: report[key] for key in keys}
                if args.experiment in ("kuhn", "leduc", "meltingpot"):
                    selected["selected_core_cohort"] = core_cohort_summary(args.experiment)
                    name = "melting" if args.experiment == "meltingpot" else args.experiment
                    selected["table1_comparison"] = comparisons[name]
                elif args.experiment is None:
                    selected["table1_comparisons"] = comparisons
                selected["scope"] = (
                    "Recomputed from saved labels and artifacts after all manifest hashes were checked; "
                    "no simulator or historical rollout was re-executed. The core aggregate pools "
                    "Kuhn, Leduc, and Melting Pot. selected_core_cohort isolates the requested "
                    "30-root tie-aware disagreement count."
                )
                if args.experiment == "highway":
                    selected["availability"] = (
                        "The supplied verifier does not recompute an original Highway aggregate; "
                        "its evidence files were hash-checked only."
                    )
                (output / "saved-analysis.json").write_text(json.dumps(
                    {"experiment": args.experiment or "all", "results": selected}, indent=2) + "\n")

        if args.command == "figures":
            figures = ROOT / "reproduction" / "figures" / "run.py"
            if not figures.is_file():
                raise FileNotFoundError(f"figure launcher missing: {figures}")
            command = [args.python, str(figures), "--output", str(output)]
            run(command)

        if args.command == "smoke":
            experiments = [args.experiment] if args.experiment else list(EXPERIMENTS)
            for experiment in experiments:
                run([args.python, str(ROOT / "reproduction" / "smoke" / "run.py"),
                     "--experiment", experiment, "--output", str(output / "smoke" / f"{experiment}.json")])
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Reproduction failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
