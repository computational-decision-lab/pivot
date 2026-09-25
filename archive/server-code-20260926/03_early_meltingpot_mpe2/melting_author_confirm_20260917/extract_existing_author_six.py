"""Re-express the existing frozen 30-seed cohort; runs zero new simulations.

The author methods were predeclared secondary methods. Extracting their table
does not retrospectively promote them to the cohort's primary hypothesis.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import tarfile
from pathlib import Path
import numpy as np

METHODS = ("proxy_only", "author_random_hf", "author_paired_lucb",
           "author_global_voi", "author_pivot_voi", "all_hf_reference")
ROOT_SEEDS = list(range(44000, 44030))

def sha(data):
    return hashlib.sha256(data).hexdigest()

def interval(values):
    x = np.asarray(values, dtype=float)
    assert len(x) == 30 and np.isfinite(x).all()
    indices = np.random.default_rng(20260917).integers(30, size=(10000, 30))
    limits = np.quantile(x[indices].mean(axis=1), [.025, .975])
    return {"mean": float(x.mean()), "ci95_low": float(limits[0]),
            "ci95_high": float(limits[1]), "n_seeds": 30}

def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    protocol_path = source / "melting_method_v3_20260917_frozen.json"
    csv_path = source / "method_analysis/method_seed_results.csv"
    archive_path = source / "melting_method_verified_20260917.tar.gz"
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    assert protocol["test_seeds"] == ROOT_SEEDS
    assert protocol["primary_method"] == "pivot_sequential_adaptive_stop"
    assert set(METHODS) <= set(protocol["methods"])
    raw_rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    all_rows = []
    for raw in raw_rows:
        row = dict(raw)
        for key in ("seed", "adaptation", "hf_queries"):
            row[key] = int(raw[key])
        row["budget_cap"] = int(raw["budget_cap"]) if raw["budget_cap"] else None
        for key in ("hf_episode_cost", "audit_gain", "noisy_audit_isr"):
            row[key] = float(raw[key])
        all_rows.append(row)
    selected = [r for r in all_rows if r["method"] in METHODS]
    assert len(selected) == 840
    assert set(r["seed"] for r in selected) == set(ROOT_SEEDS)
    by_key = {(r["seed"], r["adaptation"], r["method"], r["budget_cap"]): r for r in all_rows}
    assert len(by_key) == len(all_rows)
    checks = []
    with tarfile.open(archive_path, "r:gz") as archive:
        prefix = "melting_method_confirm_20260917/"
        def raw_read(name):
            file = archive.extractfile(prefix + name)
            assert file is not None
            return file.read()
        assert raw_read("protocol.json") == protocol_bytes
        archive_status = json.loads(raw_read("status.json"))
        assert archive_status["status"] == "complete" and archive_status["frozen_inputs_unchanged"]
        assert raw_read("analysis/method_seed_results.csv") == csv_path.read_bytes()
        for seed in ROOT_SEEDS:
            root = f"seed_{seed}/"
            summary = json.loads(raw_read(root + "summary.json"))
            seal = json.loads(raw_read(root + "selection_seal.json"))
            decision_bytes = raw_read(root + "decisions_frozen.json")
            scored_bytes = raw_read(root + "scored_decisions.json")
            assert sha(decision_bytes) == seal["decisions_sha256"]
            assert sha(scored_bytes) == summary["scored_decisions_sha256"]
            assert summary["source_protocol_sha256"] == sha(protocol_bytes)
            assert summary["status"] == "complete" and summary["actual_native_episodes"] == 704
            assert seal["audit_generated"] is False
            for name, field in (("candidates.json", "candidate_sha256"),
                                ("features.json", "features_sha256"),
                                ("posterior.json", "posterior_sha256"),
                                ("training_seal.json", "training_seal_sha256")):
                assert sha(raw_read(root + name)) == seal[field]
            decisions, scored = json.loads(decision_bytes), json.loads(scored_bytes)
            assert len(decisions) == len(scored)
            for before, after in zip(decisions, scored):
                assert "audit_gain" not in before and "noisy_audit_isr" not in before
                assert all(after[k] == v for k, v in before.items())
                row = by_key[(seed, after["adaptation"], after["method"], after["budget_cap"])]
                for key in ("audit_gain", "noisy_audit_isr", "hf_episode_cost"):
                    assert math.isclose(row[key], after[key], rel_tol=1e-12, abs_tol=1e-12)
                assert row["selected_id"] == after["selected_id"]
            checks.append({"seed": seed, "protocol_and_selection_seals_valid": True,
                           "archived_decisions_match_csv": True, "native_episodes": 704,
                           "summary_sha256": sha(raw_read(root + "summary.json"))})
    def get(seed, horizon, method, cap):
        real_cap = 96 if method in ("proxy_only", "calibrated_no_hf") else None if method == "all_hf_reference" else cap
        return by_key[seed, horizon, method, real_cap]
    frontier = []
    for horizon, method, cap in sorted({(r["adaptation"], r["method"], r["budget_cap"]) for r in selected}, key=str):
        subset = [r for r in selected if (r["adaptation"], r["method"], r["budget_cap"]) == (horizon, method, cap)]
        stat = interval([r["noisy_audit_isr"] for r in subset])
        frontier.append({"adaptation": horizon, "method": method, "budget_cap": cap,
                         "mean_hf_episode_cost": float(np.mean([r["hf_episode_cost"] for r in subset])),
                         "mean_audit_isr": stat["mean"], "isr_ci_low": stat["ci95_low"],
                         "isr_ci_high": stat["ci95_high"], "n_seeds": 30,
                         "reference_only": method == "all_hf_reference"})
    contrasts, contrast_seeds, interactions = [], [], []
    comparators = [m for m in METHODS if m != "author_pivot_voi"] + ["uniform_random_matched", "calibrated_no_hf"]
    for cap in protocol["hf_episode_caps"]:
        for baseline in comparators:
            effects = {}
            for horizon in (4, 32):
                values = []
                for seed in ROOT_SEEDS:
                    pivot = get(seed, horizon, "author_pivot_voi", cap)
                    other = get(seed, horizon, baseline, cap)
                    effect = pivot["audit_gain"] - other["audit_gain"]
                    assert math.isclose(effect, other["noisy_audit_isr"] - pivot["noisy_audit_isr"], abs_tol=1e-10)
                    values.append(effect)
                    contrast_seeds.append({"seed": seed, "adaptation": horizon, "budget_cap": cap,
                                           "baseline": baseline, "baseline_minus_author_pivot_isr": effect})
                effects[horizon] = values
                contrasts.append({"adaptation": horizon, "budget_cap": cap, "baseline": baseline,
                                  "matched_hf_cost": baseline not in ("proxy_only", "calibrated_no_hf", "all_hf_reference"),
                                  "positive_favors": "author_pivot_voi", **interval(values)})
            interactions.append({"budget_cap": cap, "baseline": baseline,
                                 "definition": "(baseline_minus_author_pivot_ISR)_long_minus_short",
                                 **interval(np.asarray(effects[32]) - np.asarray(effects[4]))})
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "author_six_seed_results.csv", selected)
    write_csv(args.output / "author_six_frontier.csv", frontier)
    primary_table = [r for r in frontier if r["budget_cap"] == 192 or r["method"] in ("proxy_only", "all_hf_reference")]
    write_csv(args.output / "author_six_budget192.csv", primary_table)
    write_csv(args.output / "author_paired_contrasts.csv", contrasts)
    write_csv(args.output / "author_paired_contrast_seeds.csv", contrast_seeds)
    write_csv(args.output / "author_response_interactions.csv", interactions)
    result = {"status": "EXISTING_FORMAL_COHORT_REEXPRESSED", "new_native_episodes": 0,
              "native_episodes_in_source": 21120, "independent_seeds": 30,
              "methods": list(METHODS), "source_csv_sha256": sha(csv_path.read_bytes()),
              "source_archive_sha256": sha(archive_path.read_bytes()), "protocol_sha256": sha(protocol_bytes),
              "frozen_primary_method": protocol["primary_method"],
              "frozen_primary_comparator": protocol["primary_comparator"],
              "author_table_status": "predeclared secondary methods; newly tabulated pointwise descriptive contrasts, not a replacement primary test",
              "bootstrap": {"unit": "paired root seed", "draws": 10000, "seed": 20260917, "multiplicity_adjusted": False},
              "checks": checks, "budget192_table": primary_table,
              "budget192_contrasts": [r for r in contrasts if r["budget_cap"] == 192],
              "budget192_interactions": [r for r in interactions if r["budget_cap"] == 192],
              "scope": protocol["audit_target"], "limitations": protocol["claims_excluded"]}
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "verified_seed_count": len(checks), "new_native_episodes": 0,
                      "output": str(args.output.resolve()), "budget192_table": primary_table,
                      "budget192_contrasts": result["budget192_contrasts"]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
