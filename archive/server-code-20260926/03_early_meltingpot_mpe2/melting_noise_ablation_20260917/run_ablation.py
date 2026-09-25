"""Single-factor DEVELOPMENT replay: change HF likelihood noise, and nothing else.

This reuses old selection observations and old held-out audit labels. It is not
a new independent benchmark trial and must not be reported as confirmation.
The selector remains paper_loop_extension_v1_not_author_e5c verbatim.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import sys
import tarfile
import time

import numpy as np

ROOTS = tuple(range(44000, 44008))
ADAPTATIONS = (4, 32)
METHODS = ("pivot_sequential", "uniform_random_matched", "global_ivr_matched", "calibrated_no_hf")
ARMS = ("original_noise", "loo_measured_noise")
CAP = 192
ARCHIVE_ROOT = "melting_method_confirm_20260917"
EXPECTED_ARCHIVE_SHA = "117da12ba15c54f6455c831a8cda0f1318508025e6f14ac34008668053323990"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def archive_json(archive, root, filename):
    """Read exact allowlisted members without filesystem extraction."""
    if root not in ROOTS or filename not in {
        "posterior.json", "features.json", "decisions_frozen.json", "summary.json", "selection_seal.json"
    }:
        raise ValueError("non-allowlisted input")
    member = archive.extractfile(f"{ARCHIVE_ROOT}/seed_{root}/{filename}")
    if member is None:
        raise ValueError("missing archive input")
    return json.load(member)


def posterior_from_metadata(metadata, posterior_class):
    return posterior_class(
        prior_precision=1.0,
        noise_variance=float(metadata["noise_variance"]),
        mean=np.asarray(metadata["mean"], dtype=float),
        covariance=np.asarray(metadata["covariance"], dtype=float),
        n_observations=int(metadata["observations"]),
    )


def replace_observation_noise(posterior, variance):
    """No fit, new prior, coefficient change, shrinkage, or latent covariance fix."""
    if not math.isfinite(float(variance)) or float(variance) <= 0:
        raise ValueError("observation variance must be finite and positive")
    changed = copy.deepcopy(posterior)
    changed.noise_variance = float(variance)
    assert np.array_equal(changed.mean, posterior.mean)
    assert np.array_equal(changed.covariance, posterior.covariance)
    assert changed.n_observations == posterior.n_observations
    assert changed.prior_precision == posterior.prior_precision
    return changed


def loo_noise(noise_summaries, held_root, adaptation):
    """Equal-weight candidate mean then equal-weight OTHER-world mean."""
    donors = [root for root in ROOTS if root != held_root]
    world_variances = []
    for root in donors:
        rows = [r for r in noise_summaries[root]["records"] if r["adaptation"] == adaptation]
        if len(rows) != 8 or {r["candidate"] for r in rows} != {str(i) for i in range(8)}:
            raise ValueError("noise audit must contain all eight fixed candidates")
        values = [float(r["paired_variance"]) for r in rows]
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("invalid conditional noise variance")
        world_variances.append(float(np.mean(values)))
    return max(1e-4, float(np.mean(world_variances))), donors, world_variances


def selection_inputs(archive, root):
    """No historical audit labels loaded here; expose selection bank only via callback."""
    metadata = archive_json(archive, root, "posterior.json")
    features = archive_json(archive, root, "features.json")
    historical = archive_json(archive, root, "decisions_frozen.json")
    bank = {}
    for adaptation in ADAPTATIONS:
        matches = [r for r in historical if r["method"] == "all_hf_reference" and r["adaptation"] == adaptation]
        if len(matches) != 1:
            raise ValueError("missing unique historical selection-bank reference")
        bank[adaptation] = {str(i): float(matches[0]["observed"][str(i)]) for i in range(8)}
    return metadata, features, bank, historical


def make_query(values, cost):
    def query(row):
        return {"delta": values[row["transition_id"]], "hf_query_cost": cost, "split": "selection"}
    return query


def replay_world(root, archive, noise, run, posterior_class):
    metadata, features, bank, historical = selection_inputs(archive, root)
    decisions, calibration, reproduction = [], [], []
    for adaptation in ADAPTATIONS:
        original = posterior_from_metadata(metadata[str(adaptation)], posterior_class)
        measured, donors, donor_values = loo_noise(noise, root, adaptation)
        changed = replace_observation_noise(original, measured)
        calibration.append({
            "root": root, "adaptation": adaptation, "original_noise": original.noise_variance,
            "loo_noise": measured, "noise_donor_roots": donors,
            "noise_donor_world_means": donor_values,
            "mean": original.mean.tolist(), "covariance": original.covariance.tolist(),
            "initial_mean_covariance_identical": True, "n_calibration_observations": original.n_observations,
            "initial_calibration_metadata": metadata[str(adaptation)],
        })
        cost = adaptation + 32
        limit = min(8, max(0, (CAP - adaptation) // cost))
        rows = [{"transition_id": r["id"], "delta_proxy": r["proxy_delta"],
                 "features": r["features"], "hf_query_cost": cost} for r in features]
        query = make_query(bank[adaptation], cost)
        for arm, posterior in zip(ARMS, (original, changed)):
            for method in METHODS:
                result = run(rows, posterior, query, method=method, budget=limit, seed=root,
                             stop=False, fantasies=64, posterior_samples=256)
                setup = adaptation if result["hf_queries"] else 0
                result.update(root=root, adaptation=adaptation, arm=arm, budget_cap=CAP,
                              query_limit=limit, incumbent_setup_cost=setup,
                              hf_episode_cost=result["charged_cost"] + setup,
                              starting_noise_variance=posterior.noise_variance)
                assert result["hf_episode_cost"] <= CAP
                decisions.append(result)
                if arm == "original_noise":
                    # no-HF was recorded once at the historical protocol's first cap.
                    prior = [d for d in historical if d["method"] == method and d["adaptation"] == adaptation
                             and (method == "calibrated_no_hf" or d.get("budget_cap") == CAP)]
                    if len(prior) != 1:
                        raise ValueError("historical baseline missing/ambiguous")
                    old = prior[0]
                    reproduction.append({
                        "root": root, "adaptation": adaptation, "method": method,
                        "selected_id_equal": old["selected_id"] == result["selected_id"],
                        "queried_ids_equal": old["queried_ids"] == result["queried_ids"],
                        "estimates_max_abs_difference": max(abs(result["estimates"][k] - old["estimates"][k])
                                                           for k in result["estimates"]),
                    })
        orig_no = next(d for d in decisions if d["adaptation"] == adaptation and d["arm"] == ARMS[0]
                       and d["method"] == "calibrated_no_hf")
        new_no = next(d for d in decisions if d["adaptation"] == adaptation and d["arm"] == ARMS[1]
                      and d["method"] == "calibrated_no_hf")
        assert orig_no["estimates"] == new_no["estimates"]
        assert orig_no["selected_id"] == new_no["selected_id"]
    return decisions, calibration, reproduction


def summarize(scored, reproduction):
    groups = []
    for h in ADAPTATIONS:
        for arm in ARMS:
            for method in METHODS:
                rows = [r for r in scored if r["adaptation"] == h and r["arm"] == arm and r["method"] == method]
                groups.append({"adaptation": h, "arm": arm, "method": method, "worlds": len(rows),
                               "mean_audit_gain": float(np.mean([r["audit_gain"] for r in rows])),
                               "mean_noisy_audit_regret": float(np.mean([r["noisy_audit_regret"] for r in rows])),
                               "mean_hf_episode_cost": float(np.mean([r["hf_episode_cost"] for r in rows]))})
    contrasts = []
    for h in ADAPTATIONS:
        for method in METHODS:
            changes = []
            for root in ROOTS:
                rows = {r["arm"]: r for r in scored if r["root"] == root and r["adaptation"] == h and r["method"] == method}
                old, new = rows[ARMS[0]], rows[ARMS[1]]
                changes.append({"root": root, "audit_gain_change": new["audit_gain"] - old["audit_gain"],
                                "selected_id_changed": old["selected_id"] != new["selected_id"],
                                "query_order_changed": old["queried_ids"] != new["queried_ids"]})
            contrasts.append({"adaptation": h, "method": method,
                              "mean_audit_gain_change_new_minus_original": float(np.mean([r["audit_gain_change"] for r in changes])),
                              "selection_changes": sum(r["selected_id_changed"] for r in changes),
                              "query_order_changes": sum(r["query_order_changed"] for r in changes), "by_world": changes})
    reproduced = all(r["selected_id_equal"] and r["queried_ids_equal"]
                     and r["estimates_max_abs_difference"] < 1e-8 for r in reproduction)
    return {"status": "complete" if reproduced else "complete_with_reproduction_mismatch",
            "comparison_validity": "baseline_reproduced" if reproduced else "investigate_runtime_or_source_mismatch_before_interpreting",
            "scope": "Development replay on eight already-seen response worlds, not confirmation",
            "groups": groups, "noise_only_contrasts": contrasts,
            "historical_reproduction": reproduction,
            "historical_selected_ids_reproduced": all(r["selected_id_equal"] for r in reproduction),
            "historical_query_orders_reproduced": all(r["queried_ids_equal"] for r in reproduction),
            "max_estimate_reproduction_error": max(r["estimates_max_abs_difference"] for r in reproduction),
            "new_native_episodes": 0,
            "caveats": [
                "Historical audit labels already existed and were seen before this development task; sealing prevents code leakage, not research-level double dipping.",
                "Only observation variance changes; cross-world model misspecification and the raw-HF overwrite/fantasy decision mismatch remain unchanged.",
                "Scalar conditional noise is pooled across candidates; this is not a full heteroscedastic or covariance-aware model.",
                "Uniform has one prespecified historical random stream per world, paired between noise arms; eight worlds are descriptive only.",
                "Selector is the existing sequential extension with author EVSI, not unmodified author E5C batch PIVOT.",
                "Global IVR matched control is not the author's Global-VOI baseline. No LUCB claim is made in this diagnostic.",
                "Costs are charged historical equivalent native episodes; computational replay adds zero native episodes. Independent audits were paid separately.",
            ]}


def plot(scored, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    labels = ["PIVOT loop", "Uniform", "Global IVR", "No HF"]
    for axis, h in zip(axes, ADAPTATIONS):
        x = np.arange(len(METHODS))
        for shift, arm, color in zip([-.18, .18], ARMS, ["#61758A", "#CF7F36"]):
            means = [np.mean([r["noisy_audit_regret"] for r in scored if r["adaptation"] == h and r["arm"] == arm and r["method"] == m]) for m in METHODS]
            axis.bar(x + shift, means, .36, label=arm.replace("_", " "), color=color)
        axis.set_xticks(x, labels)
        axis.set_title(f"Adaptation {h}; charged cap {CAP} episodes")
        axis.set_ylabel("Mean historical noisy-audit regret")
        axis.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Observation-noise-only ablation: 8 existing worlds, DEVELOPMENT replay")
    fig.tight_layout()
    fig.savefig(output / "noise_only_ablation.png", dpi=180)
    fig.savefig(output / "noise_only_ablation.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-root", required=True, type=Path)
    parser.add_argument("--cloud-root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--noise-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.path[:0] = [str(args.author_root / "src"), str(args.author_root), str(args.cloud_root)]
    from pivot.acquisition.pivot_voi import BayesianLinearDeltaPosterior
    from sequential_pivot_extension import run
    start = time.monotonic()
    if sha(args.archive) != EXPECTED_ARCHIVE_SHA:
        raise ValueError("historical archive differs from frozen noise-audit input")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    hashes = {str(args.archive): sha(args.archive), str(Path(__file__).resolve()): sha(__file__)}
    for module in ["sequential_pivot_extension", "pivot.acquisition.pivot_voi", "pivot.acquisition.common", "pivot.transfer.features"]:
        source = Path(importlib.import_module(module).__file__)
        hashes[str(source)] = sha(source)
    noise = {}
    for root in ROOTS:
        path = args.noise_dir / f"root_{root}" / "summary.json"
        noise[root] = read(path)
        hashes[str(path)] = sha(path)
        assert noise[root]["status"] == "complete" and noise[root]["root"] == root
    write(output / "protocol.json", {
        "status": "FROZEN_BEFORE_REPLAY", "scope": "Development single-factor noise ablation",
        "roots": ROOTS, "adaptations": ADAPTATIONS, "methods": METHODS, "arms": ARMS,
        "cap": CAP, "fantasies": 64, "posterior_samples": 256, "stop": False,
        "seed_rule": "historical root seed, unchanged in both arms",
        "query_rule": "at most one query per candidate, cost h+32, setup h once if queried",
        "query_limits": {"4": 5, "32": 2}, "charged_episode_costs": {"4": 184, "32": 160},
        "changed_field_only": "posterior.noise_variance; initial mean/covariance not refit",
        "noise_estimator": "mean over eight candidates then seven donor worlds, leave held response world out; floor1e-4",
        "cost_reporting": "Equivalent charged historical native episodes; zero new native episodes",
        "audit_access": "historical summary.json first read only after all decisions persisted and sealed",
        "no_formal_inference": "all worlds are already-seen development data; no significance claim or scale-up gate",
        "implementation": "existing sequential_pivot_extension.run, not author E5C batch method",
        "source_input_hashes": hashes, "runtime": {"python": platform.python_version(), "numpy": np.__version__},
    })
    decisions, calibrations, reproductions = [], [], []
    with tarfile.open(args.archive, "r:gz") as archive:
        for root in ROOTS:
            write(output / "status.json", {"status": "running", "root": root, "completed_worlds": (root - ROOTS[0])})
            result, cal, reproduction = replay_world(root, archive, noise, run, BayesianLinearDeltaPosterior)
            write(output / f"root_{root}" / "decisions.json", result)
            write(output / f"root_{root}" / "calibration.json", cal)
            decisions.extend(result)
            calibrations.extend(cal)
            reproductions.extend(reproduction)
            print(json.dumps({"root": root, "decisions": len(result), "elapsed_seconds": time.monotonic() - start}), flush=True)
        write(output / "decisions_sealed.json", decisions)
        seal = sha(output / "decisions_sealed.json")
        write(output / "selection_seal.json", {"decisions_sha256": seal, "audit_summary_loaded": False,
                                               "historical_labels_already_existed": True})
        # Phase boundary: all method decisions above were computed without audit input.
        labels = {root: archive_json(archive, root, "summary.json")["audit_gains"] for root in ROOTS}
    scored = []
    for d in decisions:
        gains = labels[d["root"]][str(d["adaptation"])]
        scored.append(dict(d, audit_gain=gains[d["selected_id"]], noisy_audit_regret=max(gains.values()) - gains[d["selected_id"]]))
    assert sha(output / "decisions_sealed.json") == seal
    write(output / "scored_decisions.json", scored)
    write(output / "calibrations.json", calibrations)
    summary = summarize(scored, reproductions)
    summary["elapsed_seconds"] = time.monotonic() - start
    write(output / "summary.json", summary)
    fields = ["root", "adaptation", "arm", "method", "selected_id", "audit_gain", "noisy_audit_regret",
              "hf_queries", "charged_cost", "incumbent_setup_cost", "hf_episode_cost", "starting_noise_variance", "queried_ids"]
    with (output / "seed_results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scored)
    plot(scored, output)
    write(output / "status.json", {"status": summary["status"], "worlds": len(ROOTS), "decisions": len(decisions),
                                   "elapsed_seconds": time.monotonic() - start, "new_native_episodes": 0})
    print(json.dumps({"status": summary["status"], "output": str(output), "worlds": len(ROOTS)}), flush=True)


if __name__ == "__main__":
    main()
