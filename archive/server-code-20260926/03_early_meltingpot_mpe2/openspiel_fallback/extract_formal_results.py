#!/usr/bin/env python3
"""Extract a compact, auditable view of the sealed OpenSpiel formal result.

This post-audit utility never participates in selection.  It reduces the large
audit payload to root-level CSVs and small JSON summaries suitable for plotting
and review on a laptop.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BOOTSTRAP_DRAWS = 5000


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def bootstrap_mean(values: list[float], *seed_parts: object) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(stable_seed("compact_posthoc", *seed_parts))
    sampled = array[rng.integers(0, len(array), size=(BOOTSTRAP_DRAWS, len(array)))]
    means = sampled.mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "n_roots": int(len(array)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"row schemas differ for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = args.audit.read_bytes()
    audit = json.loads(raw)
    if audit.get("status") != "AUDITED_AFTER_SELECTION_SEAL":
        raise RuntimeError("input is not a sealed-and-audited formal result")
    if audit.get("selection_audit_seed_overlap") is not False:
        raise RuntimeError("selection/audit seed separation is not certified")

    scored = audit["scored_decisions"]
    exact_rows = audit["exact_candidate_values"]
    if len(scored) != 10200 or len(exact_rows) != 1800:
        raise RuntimeError(
            f"unexpected formal result cardinality: {len(scored)=}, {len(exact_rows)=}"
        )

    # Average allocation replicas within root before any across-root summary.
    grouped: dict[tuple[str, int, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in scored:
        grouped[
            (
                str(row["horizon"]),
                int(row["budget"]),
                str(row["method"]),
                int(row["root_seed"]),
            )
        ].append(row)

    root_method_rows: list[dict[str, object]] = []
    for (horizon, budget, method, root), rows in sorted(grouped.items()):
        root_method_rows.append(
            {
                "root_seed": root,
                "horizon": horizon,
                "budget_paired_hands": budget,
                "method": method,
                "allocation_replicates": len(rows),
                "selection_regret": float(np.mean([r["selection_regret"] for r in rows])),
                "selection_accuracy": float(np.mean([r["correct_selection"] for r in rows])),
                "harmful_promotion_rate": float(np.mean([r["harmful_promotion"] for r in rows])),
                "nominal_paired_hands": float(np.mean([r["nominal_paired_hands"] for r in rows])),
                "physical_profile_hands": float(np.mean([r["physical_profile_hands"] for r in rows])),
            }
        )

    summary_groups: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in root_method_rows:
        summary_groups[
            (
                str(row["horizon"]),
                int(row["budget_paired_hands"]),
                str(row["method"]),
            )
        ].append(row)

    method_summary_rows: list[dict[str, object]] = []
    for (horizon, budget, method), rows in sorted(summary_groups.items()):
        regrets = [float(row["selection_regret"]) for row in rows]
        interval = bootstrap_mean(regrets, horizon, budget, method, "regret")
        method_summary_rows.append(
            {
                "horizon": horizon,
                "budget_paired_hands": budget,
                "method": method,
                "mean_selection_regret": interval["mean"],
                "regret_ci_low": interval["ci_low"],
                "regret_ci_high": interval["ci_high"],
                "selection_accuracy": float(np.mean([r["selection_accuracy"] for r in rows])),
                "harmful_promotion_rate": float(np.mean([r["harmful_promotion_rate"] for r in rows])),
                "mean_physical_profile_hands": float(np.mean([r["physical_profile_hands"] for r in rows])),
                "n_roots": len(rows),
                "bootstrap_draws": BOOTSTRAP_DRAWS,
            }
        )

    exact_groups: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in exact_rows:
        exact_groups[(int(row["root_seed"]), str(row["horizon"]))].append(row)

    mechanism_rows: list[dict[str, object]] = []
    for (root, horizon), rows in sorted(exact_groups.items()):
        rows = sorted(rows, key=lambda row: int(row["candidate_index"]))
        proxy = np.asarray([row["proxy_improvement"] for row in rows], dtype=float)
        exact = np.asarray([row["exact_deployment_improvement"] for row in rows], dtype=float)
        proxy_winner = int(np.argmax(proxy))
        oracle_winner = int(np.argmax(exact))
        ordered = np.sort(exact)
        mechanism_rows.append(
            {
                "root_seed": root,
                "horizon": horizon,
                "mean_abs_proxy_deployment_gap": float(np.mean(np.abs(proxy - exact))),
                "proxy_winner_index": proxy_winner,
                "deployment_winner_index": oracle_winner,
                "winner_mismatch": int(proxy_winner != oracle_winner),
                "proxy_best_selection_regret": float(exact[oracle_winner] - exact[proxy_winner]),
                "deployment_top_two_margin": float(ordered[-1] - ordered[-2]),
                "oracle_deployment_improvement": float(exact[oracle_winner]),
                "proxy_best_deployment_improvement": float(exact[proxy_winner]),
            }
        )

    root_lookup = {
        (
            int(row["root_seed"]),
            str(row["horizon"]),
            int(row["budget_paired_hands"]),
            str(row["method"]),
        ): row
        for row in root_method_rows
    }
    primary_rows: list[dict[str, object]] = []
    primary_methods = [
        "pivot_cg_v2",
        "uniform_random",
        "global_ivr",
        "posterior_lucb",
        "author_global_voi_e5c_adapter",
        "calibrated_no_hf",
        "proxy_only",
    ]
    for root in range(71000, 71100):
        for horizon in ("short", "long"):
            values = {
                method: float(root_lookup[(root, horizon, 128, method)]["selection_regret"])
                for method in primary_methods
            }
            primary_rows.append(
                {
                    "root_seed": root,
                    "horizon": horizon,
                    **{f"{method}_regret": values[method] for method in primary_methods},
                    "random_minus_pivot": values["uniform_random"] - values["pivot_cg_v2"],
                    "no_hf_minus_pivot": values["calibrated_no_hf"] - values["pivot_cg_v2"],
                    "lucb_minus_pivot": values["posterior_lucb"] - values["pivot_cg_v2"],
                    "ivr_minus_pivot": values["global_ivr"] - values["pivot_cg_v2"],
                }
            )

    compact = {
        "source_audit_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol_lock_sha256": audit["protocol_lock_sha256"],
        "selection_seal_sha256": audit["selection_seal_sha256"],
        "status": audit["status"],
        "audit_target": audit["audit_target"],
        "selection_audit_seed_overlap": audit["selection_audit_seed_overlap"],
        "decision": audit["decision"],
        "predeclared_contrasts": audit["contrasts"],
        "mechanism": audit["mechanism"],
        "posthoc_summary_policy": {
            "unit": "root",
            "uniform_random_replicates": "averaged within root",
            "bootstrap": "root bootstrap with deterministic namespace",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "confirmatory_status": "descriptive only; predeclared contrasts remain authoritative",
        },
        "cardinality": {
            "scored_decisions": len(scored),
            "exact_candidate_values": len(exact_rows),
            "root_method_budget_rows": len(root_method_rows),
            "mechanism_root_horizon_rows": len(mechanism_rows),
            "primary_root_horizon_rows": len(primary_rows),
        },
        "method_budget_summary": method_summary_rows,
    }

    (args.output_dir / "formal_compact_summary.json").write_text(
        json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "method_budget_summary.csv", method_summary_rows)
    write_csv(args.output_dir / "root_method_budget.csv", root_method_rows)
    write_csv(args.output_dir / "mechanism_by_root.csv", mechanism_rows)
    write_csv(args.output_dir / "primary_budget_128_by_root.csv", primary_rows)

    manifest = {}
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "compact_manifest.json":
            manifest[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    (args.output_dir / "compact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
