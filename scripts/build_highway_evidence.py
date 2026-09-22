#!/usr/bin/env python3
"""Recompute HighwayEnv paper results from complete, hash-checked seed records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper.figures.v10_style import COLORS, TEXT_SIZES, apply, figure_size

COMPARATORS = ("Uniform HF", "Calibrated PIVOT-KG")
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED_BASE = 20260922


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def paired_contrast(
    records: list[dict[str, Any]], seeds: list[int], *, budget: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pairs: dict[int, dict[str, float]] = {}
    for row in records:
        if row["budget"] != budget or row["method"] not in COMPARATORS:
            continue
        seed, method = int(row["seed"]), row["method"]
        pair = pairs.setdefault(seed, {})
        if method in pair:
            raise ValueError("duplicate method/seed/budget observation")
        value = float(row["ISR"])
        if not np.isfinite(value) or value < -1e-12:
            raise ValueError("regret must be finite and nonnegative")
        pair[method] = value
    if len(set(seeds)) != len(seeds) or set(pairs) != set(seeds) or any(
        set(pair) != set(COMPARATORS) for pair in pairs.values()
    ):
        raise ValueError("registered seed pairs are incomplete or unexpected")
    rows = [
        {
            "seed": seed,
            "budget": budget,
            "uniform_ISR": pairs[seed][COMPARATORS[0]],
            "pivot_ISR": pairs[seed][COMPARATORS[1]],
            "uniform_minus_pivot": pairs[seed][COMPARATORS[0]] - pairs[seed][COMPARATORS[1]],
        }
        for seed in seeds
    ]
    values = np.asarray([row["uniform_minus_pivot"] for row in rows])
    rng = np.random.default_rng(BOOTSTRAP_SEED_BASE + budget)
    means = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "budget": budget,
        "n_seed_pairs": len(seeds),
        "mean": float(values.mean()),
        "ci95": [float(lo), float(hi)],
        "uniform_ISR": float(np.mean([row["uniform_ISR"] for row in rows])),
        "pivot_ISR": float(np.mean([row["pivot_ISR"] for row in rows])),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED_BASE + budget,
        "conclusion": "supported" if lo > 0 else "not_supported" if hi < 0 else "unresolved",
    }, rows


def audit_cohort(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text())
    for name, record in manifest["files"].items():
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("invalid manifest path")
        data = path.read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Highway evidence checksum mismatch: {root.name}/{name}")
    def load(name: str) -> Any:
        if name not in manifest["files"]:
            raise ValueError(f"unbound evidence file: {name}")
        return json.loads((root / name).read_text())
    identity = load("identity.json")
    config = identity["config"]
    seeds = config["test_seeds"]
    calibration = config["calibration_seeds"]
    if len(seeds) != 60 or len(calibration) != 20 or len(set(calibration)) != 20 or set(seeds) & set(calibration):
        raise ValueError("expected 20 calibration and 60 disjoint test seeds")
    model = load("calibration-model.json")
    decision = load("decision-identity.json")
    if (decision["base_config_hash"] != _digest(config)
            or decision["calibration_model"] != model
            or decision["decision_config_hash"] != _digest({"config": config, "calibration_model": model})
            or identity["source"]["source_hash"] != _digest(identity["source"]["files"])):
        raise ValueError("source or calibrated decision identity does not match its seal")
    summary = load("summary.json")
    if summary["failures"] or "SMOKE" in summary["status"]:
        raise ValueError("failed or smoke run cannot supply paper evidence")
    records = load("promotion-results.json")["rows"]
    cells = {(method, budget) for method in config["methods"] for budget in (
        [config["all_hf_budget"]] if method == "All-HF Oracle" else config["budgets"]
    )}
    expected = {(seed, method, budget) for seed in seeds for method, budget in cells}
    actual = [(row["seed"], row["method"], row["budget"]) for row in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("promotion cells are incomplete or duplicated")
    for row in records:
        if not np.isclose(row["ISR"], row["actor_best_delta"] - row["selected_actor_delta"], atol=1e-12, rtol=0):
            raise ValueError("regret does not match audited deployment values")
    truth = load("truth-audit.json")["rows"]
    truth_map = {(r["seed"], r["method"], r["budget"], r["candidate_id"]): r for r in truth}
    candidates = set(model["candidate_ids"])
    expected_truth = {(*key, candidate) for key in expected for candidate in candidates}
    if len(truth_map) != len(truth) or set(truth_map) != expected_truth:
        raise ValueError("truth audit is incomplete or duplicated")
    for row in records:
        key = (row["seed"], row["method"], row["budget"])
        values = {c: truth_map[(*key, c)]["paired_delta"] for c in candidates}
        values["incumbent"] = 0.0
        if (not all(np.isfinite(v) for v in values.values())
                or row["selected_candidate"] not in values
                or not np.isclose(row["selected_actor_delta"], values[row["selected_candidate"]], atol=1e-12, rtol=0)
                or not np.isclose(row["actor_best_delta"], max(values.values()), atol=1e-12, rtol=0)):
            raise ValueError("promotion values differ from candidate truth audit")
        if row["hf_queries"] != key[2] or row["candidate_count"] != len(candidates):
            raise ValueError("promotion budget or candidate count mismatch")
    queries = load("query-ledger.json")["rows"]
    groups: dict[tuple, list[dict]] = {}
    for row in queries:
        key = (row["seed"], row["method"], row["budget"])
        groups.setdefault(key, []).append(row)
        audited = truth_map[(*key, row["candidate_id"])]
        if not row["logical_hf_query"] or not audited["logical_hf_query"] or not np.isclose(
            row["paired_delta"], audited["paired_delta"], atol=1e-12, rtol=0
        ):
            raise ValueError("query ledger differs from truth audit")
    if set(groups) != expected:
        raise ValueError("query cells differ from registered methods and budgets")
    for key, group in groups.items():
        budget = key[2]
        if len(group) != budget or sorted(r["query_index"] for r in group) != list(range(budget)):
            raise ValueError("logical query budget mismatch")
        if len({r["candidate_id"] for r in group}) != budget:
            raise ValueError("duplicate queried candidate")
        queried = {r["candidate_id"] for r in group}
        for candidate in candidates:
            audited = truth_map[(*key, candidate)]
            if audited["logical_hf_query"] != (candidate in queried) or audited["audit_only"] != (candidate not in queried):
                raise ValueError("truth query/audit partition mismatch")
    costs = load("cost-summary.json")
    expected_evaluations = 20 * 2 * (len(candidates) + 1) + 60 * (len(cells) + 1) * (len(candidates) + 1)
    if (not costs["cost_complete"] or costs["failed_evaluations"] or costs["unresolved_evaluations"]
            or costs["unmeasured_evaluations"] or costs["completed_evaluations"] != expected_evaluations):
        raise ValueError("physical evaluation accounting is incomplete")
    contrasts, paired = [], []
    for budget in config["budgets"]:
        contrast, rows = paired_contrast(records, seeds, budget=budget)
        recorded = next(r for r in summary["budget_contrasts"] if r["budget"] == budget)
        for name in ("mean", "ci95", "n_seed_pairs"):
            if not np.allclose(contrast[name], recorded[name], atol=1e-12, rtol=0):
                raise ValueError(f"summary differs from paired seed recomputation: {budget}/{name}")
        contrasts.append(contrast)
        paired.extend(rows)
    return {
        "valid": True, "protocol_id": config["protocol_id"], "config": config,
        "source": identity["source"], "dependencies": identity["dependencies"],
        "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "contrasts": contrasts, "paired_rows": paired,
        "promotion_rows": len(records), "query_rows": len(queries), "truth_rows": len(truth),
        "physical_evaluations": costs["completed_evaluations"],
    }


def audit_redesign(root: Path) -> dict[str, Any]:
    """Audit the eight-candidate external redesign without reusing old schema."""

    manifest = json.loads((root / "manifest.json").read_text())
    for name, record in manifest["files"].items():
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("invalid redesign manifest path")
        data = path.read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Highway redesign checksum mismatch: {root.name}/{name}")

    def load(name: str) -> Any:
        if name not in manifest["files"]:
            raise ValueError(f"unbound redesign evidence file: {name}")
        return json.loads((root / name).read_text())

    identity = load("identity.json")
    config = identity["config"]
    calibration = [int(seed) for seed in config["calibration_seeds"]]
    seeds = [int(seed) for seed in config["test_seeds"]]
    if (
        len(calibration) != 40
        or len(set(calibration)) != 40
        or len(seeds) != 120
        or len(set(seeds)) != 120
        or set(calibration) & set(seeds)
    ):
        raise ValueError("redesign requires 40 calibration and 120 disjoint test seeds")
    if config["candidate_count"] != 8 or config["budgets"] != [2, 4] or config["primary_budget"] != 4:
        raise ValueError("redesign candidate or budget contract failed")
    if config["subset_counts"] != {"2": 28, "4": 70}:
        raise ValueError("redesign subset contract failed")

    model = load("calibration-model.json")
    candidates = tuple(model["candidate_ids"])
    if len(candidates) != 8 or len(set(candidates)) != 8:
        raise ValueError("redesign calibration model does not define eight unique candidates")
    decision = load("decision-identity.json")
    if (
        decision["base_config_hash"] != _digest(config)
        or decision["calibration_model"] != model
        or decision["decision_config_hash"] != _digest({"config": config, "calibration_model": model})
        or identity["source"]["source_hash"] != _digest(identity["source"]["files"])
    ):
        raise ValueError("redesign source or decision identity does not match its seal")

    summary = load("summary.json")
    if summary["failures"] or not summary["convergence_ok"]:
        raise ValueError("failed redesign cannot supply paper evidence")
    if summary["status"] != "DEV_EXTERNAL_REDESIGN":
        raise ValueError("redesign evidence has an unexpected status")

    promotion = load("promotion-results.json")["rows"]
    methods = ("Uniform HF", "Calibrated PIVOT-KG", "All-HF Oracle")
    expected_cells = {
        (seed, method, budget)
        for seed in seeds
        for method, budgets in (
            ("Uniform HF", (2, 4)),
            ("Calibrated PIVOT-KG", (2, 4)),
            ("All-HF Oracle", (8,)),
        )
        for budget in budgets
    }
    actual_cells = [(int(row["seed"]), row["method"], int(row["budget"])) for row in promotion]
    if len(actual_cells) != len(expected_cells) or set(actual_cells) != expected_cells:
        raise ValueError("redesign promotion cells are incomplete or duplicated")
    if any(
        row["outcome_chasing"]
        or row.get("analysis_truth_reused", False)
        or int(row["candidate_count"]) != len(candidates)
        for row in promotion
    ):
        raise ValueError("redesign promotion rows violate the fixed protocol")

    truth = load("truth-audit.json")["rows"]
    truth_map = {(int(row["seed"]), row["candidate_id"]): row for row in truth}
    truth_candidates = (*candidates, "incumbent")
    expected_truth = {(seed, candidate) for seed in seeds for candidate in truth_candidates}
    if len(truth_map) != len(truth) or set(truth_map) != expected_truth:
        raise ValueError("redesign truth audit is incomplete or duplicated")
    if any(
        not row["audit_only"]
        or row["logical_hf_query"]
        or row["method"] != "Truth Audit"
        or row["outcome_chasing"]
        for row in truth
    ):
        raise ValueError("redesign truth audit contains logical query outcomes")

    promotion_by_cell = {
        (int(row["seed"]), row["method"], int(row["budget"])): row for row in promotion
    }
    for row in promotion:
        if row["method"] not in methods:
            raise ValueError("unexpected redesign promotion method")
        if row["method"] == "Calibrated PIVOT-KG":
            values = {
                candidate: float(truth_map[(int(row["seed"]), candidate)]["paired_delta"])
                for candidate in truth_candidates
            }
            if not np.isclose(row["actor_best_delta"], max(values.values()), atol=1e-12, rtol=0):
                raise ValueError("redesign PIVOT actor best differs from truth audit")
            if not np.isclose(
                row["selected_actor_delta"], values[row["selected_candidate"]], atol=1e-12, rtol=0
            ):
                raise ValueError("redesign PIVOT selected value differs from truth audit")
            if int(row["hf_queries"]) != int(row["budget"]):
                raise ValueError("redesign PIVOT budget mismatch")
        elif row["method"] == "All-HF Oracle":
            values = [
                float(truth_map[(int(row["seed"]), candidate)]["paired_delta"])
                for candidate in truth_candidates
            ]
            if not np.isclose(row["actor_best_delta"], max(values), atol=1e-12, rtol=0):
                raise ValueError("redesign oracle best differs from truth audit")

    rows = load("rows.json")["rows"]
    paired = {(int(row["seed"]), int(row["budget"])): row for row in rows}
    expected_pairs = {(seed, budget) for seed in seeds for budget in (2, 4)}
    if len(paired) != len(rows) or set(paired) != expected_pairs:
        raise ValueError("redesign paired rows are incomplete or duplicated")
    for key, row in paired.items():
        uniform = promotion_by_cell[(key[0], "Uniform HF", key[1])]
        pivot = promotion_by_cell[(key[0], "Calibrated PIVOT-KG", key[1])]
        if not np.isclose(row["uniform_expected_ISR"], uniform["ISR"], atol=1e-12, rtol=0):
            raise ValueError("redesign Uniform row differs from promotion record")
        if not np.isclose(row["pivot_ISR"], pivot["pivot_ISR"], atol=1e-12, rtol=0):
            raise ValueError("redesign PIVOT row differs from promotion record")

    queries = load("query-ledger.json")["rows"]
    query_groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in queries:
        if (
            row["method"] != "Calibrated PIVOT-KG"
            or not row["physical_pair_evaluation"]
            or row["analysis_truth_reused"]
            or not row["logical_hf_query"]
            or row["outcome_chasing"]
        ):
            raise ValueError("redesign query ledger contains a nonphysical or reused query")
        key = (int(row["seed"]), int(row["budget"]))
        query_groups.setdefault(key, []).append(row)
        audited = truth_map[(key[0], row["candidate_id"])]
        if not np.isclose(row["paired_delta"], audited["paired_delta"], atol=1e-12, rtol=0):
            raise ValueError("redesign query value differs from post-decision audit")
    if set(query_groups) != expected_pairs:
        raise ValueError("redesign query groups differ from registered cells")
    for key, group in query_groups.items():
        budget = key[1]
        if len(group) != budget or sorted(row["query_index"] for row in group) != list(range(budget)):
            raise ValueError("redesign logical query budget mismatch")
        if len({row["candidate_id"] for row in group}) != budget:
            raise ValueError("redesign query selected a candidate twice")

    subset_rows = load("subset-ledger.json")["rows"]
    subset_groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in subset_rows:
        key = (int(row["seed"]), len(row["queried_candidates"]))
        subset_groups.setdefault(key, []).append(row)
    for key, group in subset_groups.items():
        expected_count = {2: 28, 4: 70}.get(key[1])
        if expected_count is None or len(group) != expected_count:
            raise ValueError("redesign Uniform subset count mismatch")
        if {int(row["subset_index"]) for row in group} != set(range(expected_count)):
            raise ValueError("redesign Uniform subset indices are incomplete")
        expected_isr = paired[key]["uniform_expected_ISR"]
        if not np.isclose(
            np.mean([row["ISR"] for row in group]), expected_isr, atol=1e-12, rtol=0
        ):
            raise ValueError("redesign subset ledger differs from Uniform expectation")
    if set(subset_groups) != expected_pairs:
        raise ValueError("redesign subset groups are incomplete")

    costs = load("cost-summary.json")
    phase_counts = {
        "calibration_proxy": 360,
        "calibration_actor": 360,
        "proxy": 1080,
        "hf_query": 960,
        "truth_audit": 1080,
    }
    if (
        not costs["cost_complete"]
        or costs["failed_evaluations"]
        or costs["unresolved_evaluations"]
        or costs["unmeasured_evaluations"]
        or costs["completed_evaluations"] != sum(phase_counts.values())
        or costs["phase_counts"] != phase_counts
    ):
        raise ValueError("redesign physical evaluation accounting is incomplete")
    notes = load("execution-notes.json")
    if (
        not notes["truth_audit_after_all_decisions"]
        or notes["analysis_truth_reused"]
        or not notes["all_logical_queries_physical"]
    ):
        raise ValueError("redesign execution ordering or query provenance is not sealed")

    contrasts: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for budget in (2, 4):
        values = np.asarray(
            [paired[(seed, budget)]["uniform_expected_ISR"] - paired[(seed, budget)]["pivot_ISR"] for seed in seeds],
            dtype=float,
        )
        rng = np.random.default_rng(20260927 + budget)
        bootstrap = rng.choice(values, size=(4000, len(values)), replace=True).mean(axis=1)
        lo, hi = np.percentile(bootstrap, [2.5, 97.5])
        contrast = {
            "budget": budget,
            "n_seed_pairs": len(seeds),
            "mean": float(values.mean()),
            "ci95": [float(lo), float(hi)],
            "uniform_ISR": float(np.mean([paired[(seed, budget)]["uniform_expected_ISR"] for seed in seeds])),
            "pivot_ISR": float(np.mean([paired[(seed, budget)]["pivot_ISR"] for seed in seeds])),
            "bootstrap_draws": 4000,
            "bootstrap_seed": 20260927 + budget,
            "conclusion": "supported" if lo > 0 else "not_supported" if hi < 0 else "unresolved",
        }
        recorded = next(row for row in summary["budget_contrasts"] if row["budget"] == budget)
        for name in ("mean", "ci95", "n_seed_pairs"):
            if not np.allclose(contrast[name], recorded[name], atol=1e-12, rtol=0):
                raise ValueError(f"redesign summary differs from paired recomputation: {budget}/{name}")
        contrasts.append(contrast)
        paired_rows.extend(
            {
                "seed": seed,
                "budget": budget,
                "uniform_ISR": paired[(seed, budget)]["uniform_expected_ISR"],
                "pivot_ISR": paired[(seed, budget)]["pivot_ISR"],
                "uniform_minus_pivot": paired[(seed, budget)]["uniform_expected_ISR"] - paired[(seed, budget)]["pivot_ISR"],
            }
            for seed in seeds
        )
    return {
        "valid": True,
        "protocol_id": config["protocol_id"],
        "config": config,
        "source": identity["source"],
        "dependencies": identity["dependencies"],
        "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "contrasts": contrasts,
        "paired_rows": paired_rows,
        "promotion_rows": len(promotion),
        "query_rows": len(queries),
        "truth_rows": len(truth),
        "physical_evaluations": costs["completed_evaluations"],
    }


def build(root: Path) -> dict[str, Any]:
    audits = {name: audit_cohort(root / "evidence/highway" / name) for name in ("original", "replication")}
    redesign = audit_redesign(root / "evidence/highway" / "redesign")
    original, replication = audits.values()
    from reproduction.highway.run import verify_source
    verify_source(root / "reproduction/highway")
    for cohort, audit in audits.items():
        for name, expected_hash in audit["source"]["files"].items():
            exported = "scripts/run_highway_original.py" if cohort == "original" and name == "scripts/run_highway_calibrated.py" else name
            data = (root / "reproduction/highway/source" / exported).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected_hash:
                raise ValueError(f"frozen source differs from the executed {cohort} code")
    redesign_source_root = root / "reproduction/highway_redesign/source"
    for name, expected_hash in redesign["source"]["files"].items():
        data = (redesign_source_root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError(f"frozen source differs from the executed redesign code: {name}")
    original_seeds = set(original["config"]["test_seeds"] + original["config"]["calibration_seeds"])
    replication_seeds = set(replication["config"]["test_seeds"] + replication["config"]["calibration_seeds"])
    if original_seeds & replication_seeds or replication["config"]["budgets"] != [2, 4]:
        raise ValueError("replication seed or budget contract failed")
    macros = []
    word = {1: "One", 2: "Two", 4: "Four"}
    table = [r"\begin{tabular}{@{}llrrrrl@{}}", r"\toprule",
             r"Cohort & HF budget & $n$ & Uniform ISR & PIVOT-KG ISR & Gain & 95\% CI \\", r"\midrule"]
    pairs = []
    all_audits = {**audits, "redesign": redesign}
    for name, audit in all_audits.items():
        prefix = {"original": "Highway", "replication": "HighwayReplication", "redesign": "HighwayRedesign"}[name]
        seed_count = 60 if name != "redesign" else 120
        calibration_count = 20 if name != "redesign" else 40
        candidate_count = 5 if name != "redesign" else 8
        for key, value in {
            "CalibrationSeeds": calibration_count,
            "Seeds": seed_count,
            "CandidateCount": candidate_count,
            "QueryRows": audit["query_rows"],
            "TruthRows": audit["truth_rows"],
        }.items():
            macros.append(f"\\newcommand{{\\{prefix}{key}}}{{{value}}}")
        for c in audit["contrasts"]:
            budget, mean, (lo, hi) = c["budget"], c["mean"], c["ci95"]
            macros += [f"\\newcommand{{\\{prefix}Budget{word[budget]}Contrast}}{{${mean:.4f}$}}",
                       f"\\newcommand{{\\{prefix}Budget{word[budget]}CI}}{{$[{lo:.4f},{hi:.4f}]$}}"]
            label = {"original": "Original", "replication": "Replication", "redesign": "Redesign (8)"}[name]
            budget_label = ("4 (primary)" if name == "redesign" and budget == 4
                            else "2 (primary)" if name != "redesign" and budget == 2
                            else "2 (secondary)" if name == "redesign" and budget == 2
                            else str(budget))
            table.append(f"{label} & {budget_label} & {seed_count} & {c['uniform_ISR']:.4f} & {c['pivot_ISR']:.4f} & ${mean:.4f}$ & $[{lo:.4f},{hi:.4f}]$ " + r"\\")
        pairs.extend({"cohort": name, **r} for r in audit.pop("paired_rows"))
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (root / "paper/highway_results.tex").write_text("% Generated from hash-checked paired seed evidence.\n" + "\n".join(macros) + "\n")
    (root / "paper/tables/highway_results.tex").write_text("\n".join(table) + "\n")
    with (root / "paper/tables/highway_paired_contrasts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    _figure(root, all_audits)
    report = {"valid": True, "cohorts": all_audits, "no_seed_exclusions": True,
              "primary_budget_by_cohort": {"original": 2, "replication": 2, "redesign": 4},
              "secondary_budget_by_cohort": {"original": 4, "replication": 4, "redesign": 2},
              "bootstrap_draws": BOOTSTRAP_DRAWS, "redesign_bootstrap_draws": 4000,
              "bootstrap_seed_base": BOOTSTRAP_SEED_BASE,
              "scope": "external DEV replication and registered redesign; technical smoke exposed two retained redesign test roots",
              "interval_scope": "test-seed bootstrap conditional on fitted calibration model; no simultaneous guarantee"}
    (root / "paper/highway_reproduction_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _figure(root: Path, audits: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply()
    fig, ax = plt.subplots(figsize=figure_size("wide"), layout="constrained")
    styles = {
        "original": ("Original", COLORS["global"], "o", -0.08),
        "replication": ("Replication", COLORS["strategic"], "s", 0.0),
        "redesign": ("Redesign (8)", COLORS["pivot"], "^", 0.08),
    }
    for name, audit in audits.items():
        values = audit["contrasts"]
        y = np.array([c["mean"] for c in values])
        ci = np.array([c["ci95"] for c in values])
        label, color, marker, offset = styles[name]
        x = np.array([c["budget"] for c in values]) + offset
        ax.errorbar(x, y, yerr=[y-ci[:, 0], ci[:, 1]-y], label=label,
                    color=color, fmt=marker,
                    capsize=3, markersize=4, linewidth=1.2)
    ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--")
    ax.set(xticks=[1, 2, 4], xticklabels=["1", "2 (primary)", "4 (secondary)"],
           xlabel="Paired HF queries per decision", ylabel="Uniform ISR − PIVOT-KG ISR")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=TEXT_SIZES["legend"])
    targets = (
        root / "paper/figures/revision/fig4_highway_budget",
        root / "paper/figures/fig4_highway_budget",
    )
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(target.with_suffix(".png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    sys.path.insert(0, str(args.root.resolve()))
    report = build(args.root.resolve())
    print(json.dumps({"valid": report["valid"], "cohorts": list(report["cohorts"])}))
