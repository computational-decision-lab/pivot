#!/usr/bin/env python3
"""Machine-check the requested rebuttal evidence completion matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile


def sha256(path: Path) -> str:
    return hashlib.sha256(read_bytes(path)).hexdigest()


def load(path: Path):
    return json.loads(read_bytes(path).decode("utf-8"))


def read_bytes(path: Path) -> bytes:
    if path.is_file():
        return path.read_bytes()
    path = path.resolve()
    for root in path.parents:
        supplement = root / "paper/pivot_iclr2027_supplementary.zip"
        if not supplement.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        with ZipFile(supplement) as archive:
            if relative in archive.namelist():
                return archive.read(relative)
    raise FileNotFoundError(path)


def locate_root(explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        return explicit_root.resolve()
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "results/rebuttal/20261110-p1-loop-v3c-confirmation/summary.json").is_file():
            return candidate
    raise FileNotFoundError("could not locate a project or delivery root containing the rebuttal results")


def existing_path(root: Path, *relative_paths: str) -> Path:
    for relative in relative_paths:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(root / relative) for relative in relative_paths)
    raise FileNotFoundError(f"none of the expected files exists: {tried}")


def rebuttal_path(root: Path) -> Path:
    return existing_path(
        root,
        "docs/rebuttal/rebuttal_final_2026-11-10.md",
        "rebuttal/rebuttal_final_2026-11-10.md",
        "rebuttal_final_2026-11-10.md",
    )


def audit_path(root: Path, filename: str) -> Path:
    return existing_path(root, f"docs/rebuttal/audit/{filename}", f"rebuttal/audit/{filename}")


def verify_audit_manifest(root: Path) -> int:
    """Check actual bytes, including this verifier, against the audit manifest."""
    manifest = audit_path(root, "SHA256_MANIFEST.txt")
    seen: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        assert len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
        assert relative not in seen, f"duplicate manifest entry: {relative}"
        assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
        seen.add(relative)
        assert sha256(root / relative) == digest, f"stale audit manifest entry: {relative}"
    assert seen, "empty audit manifest"
    return len(seen)


def delivery_sha256(delivery: Path, relative_path: str) -> str:
    path = delivery / relative_path
    try:
        return sha256(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"delivery does not contain {relative_path} in its tree or supplementary archive"
        ) from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--delivery", type=Path, default=None)
    args = parser.parse_args()
    root = locate_root(args.root)
    audit_manifest_entries = verify_audit_manifest(root)

    p0 = load(root / "results/rebuttal/20261109-p0-offline-v2-500/summary.json")
    assert p0["roots"] == 120
    assert p0["calibration_bootstrap_B2"]["replicates"] == 500
    assert p0["calibration_bootstrap_B4"]["replicates"] == 500
    assert p0["calibration_bootstrap_B2"]["sign_stability"] == 1.0
    assert p0["calibration_bootstrap_B4"]["sign_stability"] == 1.0
    assert p0["physical_simulator_called"] is False
    assert p0["formal_lucb"]["status"] == "initialization_infeasible"
    assert sha256(root / "results/rebuttal/20261109-p0-offline-v2-500/root-rows.json") == sha256(
        root / "results/rebuttal/20261109-p0-offline-input-replay-v1/root-rows.json"
    )

    physical = load(
        root / "results/rebuttal/20261109-highway-matched-physical-v2/physical-aggregate-v2/summary.json"
    )
    assert physical["status"] == "DEV_INDEPENDENT_PHYSICAL_REPLICATION"
    assert physical["protocol_id"] == "highway-physical-reactive-matched-targeted-v2"
    assert (physical["calibration_roots"], physical["test_roots"]) == (40, 120)
    assert (physical["candidate_count"], physical["budget"]) == (8, 4)
    assert physical["failures"] == []
    journal = physical["physical_evaluation_journal"]
    assert journal["completed_evaluations"] == 3147
    assert journal["known_simulator_calls"] == 6294
    assert journal["failed_evaluations"] == 0
    assert journal["unresolved_evaluations"] == 0
    assert journal["unmeasured_evaluations"] == 0 and journal["cost_complete"]
    ivr = physical["primary_contrast"]
    assert ivr["n_root_pairs"] == 120
    assert math.isclose(ivr["mean"], -0.0029760954049873056, abs_tol=1e-12)
    assert ivr["ci95_root_bootstrap"][0] < 0 < ivr["ci95_root_bootstrap"][1]

    independent = load(
        root / "results/rebuttal/20261109-highway-independent-calibration-v1/aggregate/summary.json"
    )
    assert independent["status"] == "DEV_INDEPENDENT_PHYSICAL_REPLICATION"
    assert independent["protocol_id"] == "highway-independent-calibration-v1"
    assert (independent["calibration_roots"], independent["test_roots"]) == (40, 120)
    assert (independent["candidate_count"], independent["budget"]) == (8, 4)
    assert independent["failures"] == []
    independent_journal = independent["physical_evaluation_journal"]
    assert independent_journal["completed_evaluations"] == 3141
    assert independent_journal["known_simulator_calls"] == 6282
    assert independent_journal["failed_evaluations"] == 0
    assert independent_journal["unresolved_evaluations"] == 0
    assert independent_journal["unmeasured_evaluations"] == 0
    assert independent_journal["cost_complete"] is True
    independent_ivr = independent["primary_contrast"]
    assert independent_ivr["n_root_pairs"] == 120
    assert math.isclose(independent_ivr["mean"], 0.001604686108970511, abs_tol=1e-12)
    assert independent_ivr["ci95_root_bootstrap"][0] < 0 < independent_ivr["ci95_root_bootstrap"][1]
    independent_audit = load(audit_path(root, "HIGHWAY_INDEPENDENT_VERIFICATION_2026-11-09.json"))
    assert independent_audit["status"] == "PASS" and all(independent_audit["checks"].values())
    assert independent_audit["query_decisions_replayed"] == 1920
    assert independent_audit["terminal_decisions_replayed"] == 480
    assert independent_audit["uniform_subsets_replayed"] == 8400
    assert independent_audit["calibration_plus_test_evaluations"] == 3861
    assert independent_audit["calibration_plus_test_simulator_calls"] == 7722
    assert independent_audit["summary_sha256"] == sha256(
        root / "results/rebuttal/20261109-highway-independent-calibration-v1/aggregate/summary.json"
    )

    poker = load(root / "results/rebuttal/20261109-p0-poker-sensitivity-v3/summary.json")
    assert poker["calibration_roots"] == 12 and poker["test_roots"] == 30
    assert len(poker["grid"]["r_floor"]) * len(poker["grid"]["covariance_shrinkage"]) == 20
    assert poker["default_prior_reproduced"] is True
    poker_replay = load(root / "results/rebuttal/20261109-p0-poker-sensitivity-input-replay-v1/summary.json")
    assert poker_replay["decision_hash"] == poker["decision_hash"]
    assert poker_replay["default_prior_reproduced"] is True

    poker_bootstrap = load(
        root / "results/rebuttal/20261109-p0-poker-calibration-bootstrap-v1/summary.json"
    )
    assert poker_bootstrap["status"] == "PASS"
    assert poker_bootstrap["replicates"] == 500
    assert poker_bootstrap["calibration_roots"] == 12 and poker_bootstrap["test_roots"] == 30
    assert poker_bootstrap["decision_count"] == 75_000
    assert poker_bootstrap["default_prior_reproduced"] is True
    assert poker_bootstrap["physical_simulator_called"] is False
    assert poker_bootstrap["aggregate"]["uniform_v2"]["positive_fraction"] == 1.0
    assert poker_bootstrap["aggregate"]["ivr_v2"]["zero_fraction"] == 1.0
    poker_bootstrap_audit = load(
        audit_path(root, "POKER_CALIBRATION_BOOTSTRAP_VERIFICATION_2026-11-09.json")
    )
    assert poker_bootstrap_audit["status"] == "PASS"
    assert all(poker_bootstrap_audit["checks"].values())

    loop = load(root / "results/rebuttal/20261109-p1-loop-v1/summary.json")
    assert loop["protocol"]["roots"] == 30 and loop["protocol"]["rounds"] == 5
    assert all(loop["audit_checks"].values())
    loop_replay = load(root / "results/rebuttal/20261109-p1-loop-input-replay-v1/summary.json")
    assert loop_replay["ledger_sha256"] == loop["ledger_sha256"]
    assert all(loop_replay["audit_checks"].values())

    v2_dir = root / "results/rebuttal/20261110-p1-loop-v2-confirmation"
    v2 = load(v2_dir / "summary.json")
    v2_analysis = load(v2_dir / "paired_analysis.json")
    assert v2["protocol"]["protocol_id"] == "P1-LOOP-V2-2026-11-10"
    assert v2["protocol"]["roots_per_regime"] == 120
    assert v2["protocol"]["rounds"] == 5
    assert v2_analysis["split"] == "confirmation"
    expected_v2_cisr = {
        "low": {
            "PIVOT-KG-Loop-v2": 0.029577170367417638,
            "IVR-Online": 0.06691558781047137,
            "Top-Proxy-Online": 0.13309808778258256,
        },
        "medium": {
            "PIVOT-KG-Loop-v2": 0.34967972662266467,
            "IVR-Online": 0.5769978145764676,
            "Top-Proxy-Online": 0.15999426678311895,
        },
        "high": {
            "PIVOT-KG-Loop-v2": 0.27625804119708663,
            "IVR-Online": 0.35477742912651633,
            "Top-Proxy-Online": 0.30476305985925833,
        },
    }
    for regime, methods in expected_v2_cisr.items():
        for method, expected in methods.items():
            observed = v2["method_summary"][regime][method]["CISR"]["mean"]
            assert math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
    assert sha256(v2_dir / "trajectory_ledger.json") == v2["ledger_sha256"]

    # The v3c confirmation is the only multi-round result used for a positive
    # claim. Check its frozen protocol and paired decision statistics directly.
    v3c_dir = root / "results/rebuttal/20261110-p1-loop-v3c-confirmation"
    v3c = load(v3c_dir / "summary.json")
    v3c_analysis = load(v3c_dir / "paired_analysis.json")
    multidim = load(v3c_dir / "multidim_support_diagnostic.json")
    protocol = v3c["protocol"]
    assert protocol["protocol_id"] == "P1-LOOP-V3C-2026-11-10"
    assert (protocol["rounds"], protocol["candidate_count"], protocol["hf_budget_per_round"]) == (5, 4, 2)
    assert protocol["roots_per_regime"] == 120
    assert protocol["calibration_roots"] == 240
    assert protocol["sentinel_rule"] is None
    assert protocol["contextual_model"]["cross_round_update"] == "none"
    assert multidim["status"] == "PASS"
    assert multidim["counts"] == {
        "calibration_roots": 240,
        "calibration_rows": 960,
        "ledger_rows": 5400,
        "rounds": 5,
        "test_candidate_vectors": 21600,
        "test_states": 5400,
    }
    assert multidim["constant_calibration_features"]["round_fraction"] == [0.0]
    assert math.isclose(
        multidim["multivariate_support"]["gt_training_p95_rate"],
        0.8030555555555555,
        abs_tol=1e-12,
    )
    assert math.isclose(
        multidim["multivariate_support"]["gt_training_p99_rate"],
        0.8,
        abs_tol=1e-12,
    )
    assert multidim["test_feature_ranges"]["round_fraction"]["min"] == 0.0
    assert multidim["test_feature_ranges"]["round_fraction"]["max"] == 1.0
    assert math.isclose(
        multidim["calibration_root_heldout_residuals"]["overall"]["rmse"],
        1.0325327613353985,
        abs_tol=1e-12,
    )
    assert math.isclose(
        multidim["calibration_root_heldout_residuals"]["overall"]["working_95pct_prediction_interval_coverage"],
        0.9302083333333333,
        abs_tol=1e-12,
    )
    assert v3c["methods"] == [
        "PIVOT-KG-Loop-v3c",
        "IVR-Online-v3c",
        "Top-Proxy-Online-v3c",
    ]
    assert sha256(v3c_dir / "trajectory_ledger.json") == v3c["ledger_sha256"]
    pressure_support = load(v3c_dir / "pressure_support_diagnostic.json")
    lower = pressure_support["calibration_support"]["min"]
    upper = pressure_support["calibration_support"]["max"]
    pressure_counts: dict[str, list[int]] = {}
    for row in load(v3c_dir / "trajectory_ledger.json"):
        key = f"{row['regime']}:{row['method']}"
        counts = pressure_counts.setdefault(key, [0, 0])
        counts[0] += 1
        counts[1] += int(not lower <= row["pressure_before"] <= upper)
    for key, (total, outside) in pressure_counts.items():
        diagnostic = pressure_support["test_states_by_regime_method"][key]
        assert total == diagnostic["n_states"] == 600
        assert outside == diagnostic["outside_calibration_support"]
    assert pressure_support["input_hashes"]["ledger"] == v3c["ledger_sha256"]
    assert v3c_analysis["overall_success"] is True
    assert all(v3c_analysis["protocol_audit_checks"].values())
    assert v3c_analysis["replay_audit"]["trajectory_replay_all_pass"] is True
    assert v3c_analysis["replay_audit"]["failure_count"] == 0
    assert v3c_analysis["replay_audit"]["replay_counts"] == {
        "candidate_panels": 5400,
        "queries": 10800,
        "posterior_selections": 5400,
        "independent_evaluations": 5400,
        "deployment_returns": 5400,
        "state_transitions": 5400,
        "trajectory_aggregates": 1080,
    }
    expected_cisr = {
        "low": {
            "PIVOT-KG-Loop-v3c": 0.008881907515396218,
            "IVR-Online-v3c": 0.055742911780611484,
            "Top-Proxy-Online-v3c": 0.07095462632193868,
        },
        "medium": {
            "PIVOT-KG-Loop-v3c": 0.0014064706543051475,
            "IVR-Online-v3c": 0.44046197227728834,
            "Top-Proxy-Online-v3c": 0.23241443209483179,
        },
        "high": {
            "PIVOT-KG-Loop-v3c": 0.08885908030105018,
            "IVR-Online-v3c": 0.9853241677821174,
            "Top-Proxy-Online-v3c": 0.33346203030368154,
        },
    }
    for regime, methods in expected_cisr.items():
        for method, expected in methods.items():
            observed = v3c["method_summary"][regime][method]["CISR"]["mean"]
            assert math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
    contrasts = v3c_analysis["primary_paired_cumulative_isr_contrasts"]
    assert len(contrasts) == 6
    for contrast in contrasts.values():
        assert contrast["paired_root_bootstrap"]["ci95"][0] > 0
        assert contrast["two_sided_signflip_p"] < 0.05
        assert contrast["holm_adjusted_p_over_all_regime_baseline_contrasts"] < 0.05
        assert contrast["deployment_return_primary_minus_baseline"]["ci95"][0] >= 0

    frozen_dir = root / "results/rebuttal/20261110-frozen-state-matched-v1"
    frozen_summary = load(frozen_dir / "summary.json")
    frozen_analysis = load(frozen_dir / "paired_analysis.json")
    frozen_protocol = frozen_summary["protocol"]
    assert frozen_summary["status"] == "PASS"
    assert frozen_analysis["status"] == "PASS"
    assert all(frozen_analysis["checks"].values())
    assert frozen_protocol["protocol_id"] == "FROZEN-STATE-V3C-2026-11-10"
    assert frozen_protocol["cross_round_update"] == "none"
    assert frozen_protocol["roots_per_regime"] == 120
    assert frozen_protocol["hf_budget_per_root"] == 2
    assert frozen_summary["counts"] == {
        "roots_per_regime": 120,
        "regimes": 3,
        "methods": 3,
        "ledger_rows": 1080,
        "queries": 2160,
    }
    assert sha256(frozen_dir / "trajectory_ledger.json") == frozen_summary["ledger_sha256"]
    assert frozen_analysis["replay_counts"]["queries"] == 2160
    assert frozen_analysis["replay_counts"]["terminal_decisions"] == 1080
    frozen_contrasts = frozen_analysis["paired_contrasts"]
    assert len(frozen_contrasts) == 6
    assert frozen_contrasts["medium:IVR-Online-v3c_minus_PIVOT_ISR"][
        "ISR_baseline_minus_PIVOT"
    ]["ci95"][0] > 0
    assert frozen_contrasts["medium:Top-Proxy-Online-v3c_minus_PIVOT_ISR"][
        "ISR_baseline_minus_PIVOT"
    ]["ci95"][0] > 0
    assert frozen_contrasts["high:IVR-Online-v3c_minus_PIVOT_ISR"][
        "ISR_baseline_minus_PIVOT"
    ]["ci95"][0] < 0 < frozen_contrasts[
        "high:IVR-Online-v3c_minus_PIVOT_ISR"
    ]["ISR_baseline_minus_PIVOT"]["ci95"][1]
    assert all(
        value["ISR_baseline_minus_PIVOT"]["mean"] == 0.0
        for key, value in frozen_contrasts.items()
        if key.startswith("low:")
    )

    rebuttal = rebuttal_path(root).read_text()
    assert len(rebuttal) <= 5000 and all(ord(char) < 128 for char in rebuttal)
    assert "saved-truth replay" in rebuttal
    assert "full deployment transition or response map" in rebuttal
    assert "0/600, 76/600, and 72/600" in rebuttal
    assert "Kuhn" in rebuttal and "Melting Pot" in rebuttal and "MetaDrive" in rebuttal

    report = {
        "status": "PASS",
        "audit_manifest": {"entries_checked": audit_manifest_entries, "all_match": True},
        "highway": {
            "roots": p0["roots"],
            "bootstrap_replicates": p0["calibration_bootstrap_B4"]["replicates"],
            "b4_mean_contrast": p0["calibration_bootstrap_B4"]["mean_of_fitted_contrasts"],
        },
        "highway_physical": {
            "test_roots": physical["test_roots"],
            "completed_evaluations": journal["completed_evaluations"],
            "simulator_calls": journal["known_simulator_calls"],
            "ivr_minus_pivot": ivr["mean"],
            "ivr_ci95": ivr["ci95_root_bootstrap"],
        },
        "highway_independent_calibration": {
            "test_roots": independent["test_roots"],
            "completed_evaluations": independent_journal["completed_evaluations"],
            "simulator_calls": independent_journal["known_simulator_calls"],
            "ivr_minus_pivot": independent_ivr["mean"],
            "ivr_ci95": independent_ivr["ci95_root_bootstrap"],
            "uniform_minus_pivot": independent["contrasts"]["Uniform-Random"]["mean"],
        },
        "poker": {"decision_hash": poker["decision_hash"], "grid_points": 20},
        "poker_calibration_bootstrap": {
            "replicates": poker_bootstrap["replicates"],
            "uniform_positive_fraction": poker_bootstrap["aggregate"]["uniform_v2"]["positive_fraction"],
            "ivr_zero_fraction": poker_bootstrap["aggregate"]["ivr_v2"]["zero_fraction"],
        },
        "loop": {"ledger_sha256": loop["ledger_sha256"], "audit_all_pass": loop["audit_checks"]["all_pass"]},
        "multi_round_v2": {
            "protocol_id": v2["protocol"]["protocol_id"],
            "roots_per_regime": v2["protocol"]["roots_per_regime"],
            "ledger_sha256": v2["ledger_sha256"],
            "mixed_results_retained": True,
        },
        "multi_round_v3c": {
            "protocol_id": protocol["protocol_id"],
            "rounds": protocol["rounds"],
            "roots_per_regime": protocol["roots_per_regime"],
            "ledger_sha256": v3c["ledger_sha256"],
            "six_holm_tests_pass": all(
                item["holm_adjusted_p_over_all_regime_baseline_contrasts"] < 0.05
                for item in contrasts.values()
            ),
            "independent_replay_pass": v3c_analysis["replay_audit"]["trajectory_replay_all_pass"],
            "pressure_support_counts_recomputed": pressure_counts,
        },
        "frozen_state_matched": {
            "protocol_id": frozen_protocol["protocol_id"],
            "roots_per_regime": frozen_protocol["roots_per_regime"],
            "ledger_sha256": frozen_summary["ledger_sha256"],
            "independent_replay_pass": frozen_analysis["checks"]["all_replay_checks_pass"],
            "medium_pivot_advantage_over_ivr_and_top_proxy": True,
            "high_pivot_advantage_over_ivr": False,
            "low_response_tie": True,
        },
        "rebuttal_chars": len(rebuttal),
    }
    if args.delivery:
        delivery = args.delivery.resolve()
        assert rebuttal_path(delivery).read_text() == rebuttal
        assert delivery_sha256(delivery, "results/rebuttal/20261109-p1-loop-v1/trajectory_ledger.json") == sha256(
            root / "results/rebuttal/20261109-p1-loop-v1/trajectory_ledger.json"
        )
        for filename in ("summary.json", "paired_analysis.json", "trajectory_ledger.json", "trajectory_ledger.sha256", "protocol.json"):
            assert delivery_sha256(
                delivery, f"results/rebuttal/20261110-p1-loop-v2-confirmation/{filename}"
            ) == sha256(v2_dir / filename)
        assert delivery_sha256(
            delivery, "results/rebuttal/20261110-p1-loop-v3c-confirmation/trajectory_ledger.json"
        ) == sha256(root / "results/rebuttal/20261110-p1-loop-v3c-confirmation/trajectory_ledger.json")
        assert delivery_sha256(
            delivery, "results/rebuttal/20261110-p1-loop-v3c-confirmation/paired_analysis.json"
        ) == sha256(root / "results/rebuttal/20261110-p1-loop-v3c-confirmation/paired_analysis.json")
        assert delivery_sha256(
            delivery, "results/rebuttal/20261110-p1-loop-v3c-confirmation/multidim_support_diagnostic.json"
        ) == sha256(v3c_dir / "multidim_support_diagnostic.json")
        assert delivery_sha256(
            delivery, "results/rebuttal/20261110-frozen-state-matched-v1/trajectory_ledger.json"
        ) == sha256(frozen_dir / "trajectory_ledger.json")
        assert delivery_sha256(
            delivery, "results/rebuttal/20261110-frozen-state-matched-v1/paired_analysis.json"
        ) == sha256(frozen_dir / "paired_analysis.json")
        assert delivery_sha256(
            delivery,
            "results/rebuttal/20261109-highway-matched-physical-v2/physical-aggregate-v2/summary.json",
        ) == sha256(
            root
            / "results/rebuttal/20261109-highway-matched-physical-v2/physical-aggregate-v2/summary.json"
        )
        assert delivery_sha256(
            delivery, "results/rebuttal/20261109-p0-poker-calibration-bootstrap-v1/summary.json"
        ) == sha256(
            root / "results/rebuttal/20261109-p0-poker-calibration-bootstrap-v1/summary.json"
        )
        assert delivery_sha256(
            delivery,
            "results/rebuttal/20261109-highway-independent-calibration-v1/aggregate/summary.json",
        ) == sha256(
            root
            / "results/rebuttal/20261109-highway-independent-calibration-v1/aggregate/summary.json"
        )
        report["delivery"] = {
            "path": str(delivery),
            "rebuttal_hash_matches": True,
            "v3c_ledger_hash_matches": True,
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
