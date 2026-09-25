"""Plot measured Melting Pot quality diagnostics and, optionally, audited E5 runs.

No data are simulated, imputed as successful interactions, or borrowed from earlier
runs. Diagnostic plots are explicitly labelled as diagnostics. E5 plots require an
explicit independent-audit declaration and episode-level decisions already scored
by the experiment runner; this module does not select policies or re-fit methods.

Usage::

    python plot_melting_recovery.py --diagnostic RUN_DIR --output SMALL_PACKAGE

The canonical diagnostic payload is ``{"episodes": [...]}``. Each episode has
``task, train_seed, policy, episode_seed, horizon, episode_length, focal_return``;
event metrics are ``interaction_count, first_interaction_step`` (null if absent),
and ``resource_collect_count``. Nested ``rows: {policy: [episodes]}`` is accepted.
An optional E5 JSON has ``metadata`` with ``audit_independent: true`` and
``selection_frozen_before_audit: true``, and ``rows`` with ``task, seed, method,
budget, actual_queries, audit_gain``. These declarations are recorded as reported
provenance, not a proof that the upstream experiment protocol was correctly run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from collections import defaultdict
from datetime import datetime, timezone

_cache = Path(tempfile.gettempdir()) / "colin-melting-recovery-plot-cache"
_cache.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache / "fonts"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASK_LABELS = {
    "melting_stag": "Repeated Stag Hunt",
    "melting_pd": "Repeated Prisoner's Dilemma",
    "stag_hunt_in_the_matrix__repeated": "Repeated Stag Hunt",
    "prisoners_dilemma_in_the_matrix__repeated": "Repeated Prisoner's Dilemma",
}
POLICY_LABELS = {"untrained": "Untrained", "trained": "Trained"}
COLORS = ["#78909c", "#137c8b", "#ad7043", "#755ba1", "#548854", "#b95767"]
METHOD_LABELS = {"pivot_voi": "PIVOT", "random_hf": "Random-HF", "global_voi": "Global-VOI",
                 "paired_lucb": "Paired LUCB (author)", "proxy_only": "Proxy only"}
METHOD_COLORS = {"pivot_voi": "#137c8b", "random_hf": "#78909c", "global_voi": "#755ba1",
                 "paired_lucb": "#ad7043", "proxy_only": "#b95767"}


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, label):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def first(d, names, default=None):
    return next((d[key] for key in names if key in d), default)


def canonical_episode(raw, context, path):
    row = dict(context)
    row.update(raw)
    required = {
        "task": first(row, ["task", "substrate"]),
        "train_seed": first(row, ["train_seed", "training_seed", "seed"]),
        "policy": first(row, ["policy", "policy_label", "condition", "label"]),
        "episode_seed": first(row, ["episode_seed", "eval_seed", "environment_seed", "evaluation_seed"]),
        "horizon": first(row, ["horizon", "max_episode_steps", "frame_cap"]),
        "focal_return": first(row, ["focal_return", "native_return"]),
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(f"{path}: episode missing {missing}")
    result = {
        "task": str(required["task"]),
        "train_seed": str(required["train_seed"]),
        "policy": str(required["policy"]),
        "episode_seed": str(required["episode_seed"]),
        "horizon": int(required["horizon"]),
        "focal_return": finite(required["focal_return"], "focal_return"),
        "eval_mode": str(first(row, ["eval_mode", "action_mode"], "as_run")),
        "opponent": str(first(row, ["opponent_label", "opponent"], "as_run")),
        "source": str(Path(path).resolve()),
    }
    if "horizon_mode" in row:
        result["eval_mode"] += "; horizon=" + str(row["horizon_mode"])
    length = first(row, ["episode_length", "steps", "length", "env_steps"], result["horizon"])
    result["episode_length"] = int(length)
    if result["horizon"] <= 0 or not 0 < result["episode_length"] <= result["horizon"]:
        raise ValueError(f"{path}: invalid episode horizon or length")
    count = first(row, ["interaction_count", "num_interactions", "interactions"])
    if count is not None:
        count = finite(count, "interaction_count")
        if count < 0 or count != int(count):
            raise ValueError(f"{path}: interaction_count must be a nonnegative integer")
        result["interaction_count"] = int(count)
        result["interaction_occurred"] = int(count > 0)
    step = first(row, ["first_interaction_step", "first_interaction_time", "first_interaction_frame"])
    if step is not None:
        step = finite(step, "first_interaction_step")
        if not 0 <= step <= result["episode_length"]:
            raise ValueError(f"{path}: first interaction outside episode")
        if count == 0:
            raise ValueError(f"{path}: non-null first interaction but zero interactions")
        result["first_interaction_step"] = step
    elif count == 0:
        result["first_interaction_step"] = None
    resource = first(row, ["resource_collect_count", "resources_collected", "collection_count"])
    if resource is None and isinstance(row.get("resource_collection_totals"), dict):
        resource = row["resource_collection_totals"].get("agent_0")
    if resource is not None:
        result["resource_collect_count"] = finite(resource, "resource_collect_count")
        if result["resource_collect_count"] < 0:
            raise ValueError(f"{path}: resource counts cannot be negative")
    # A capped waiting-time diagnostic, not an uncensored average event time.
    if "first_interaction_step" in result:
        result["restricted_first_interaction_step"] = (
            result["episode_length"] if result["first_interaction_step"] is None
            else result["first_interaction_step"]
        )
    return result


def diagnostic_episodes(payload, path):
    """Accept explicit episode arrays; never turn summary means into episodes."""
    if not isinstance(payload, dict):
        return []
    context = {key: payload[key] for key in (
        "task", "substrate", "seed", "train_seed", "training_seed", "horizon",
        "max_episode_steps", "eval_mode", "action_mode", "opponent_label",
    ) if key in payload}
    episodes = payload.get("episodes")
    if isinstance(episodes, list) and episodes and isinstance(episodes[0], dict):
        return [canonical_episode(row, context, path) for row in episodes]
    rows = payload.get("rows")
    if isinstance(rows, dict):
        output = []
        for policy, items in rows.items():
            if isinstance(items, list):
                output.extend(canonical_episode(row, {**context, "policy": policy}, path)
                              for row in items)
        return output
    return []


def load_behavior_run(path):
    """Read melting_behavior_diagnostic.py's exact schema and retain exclusions."""
    protocol_path = path.parent / "protocol.json"
    if not protocol_path.exists():
        raise ValueError(f"{path}: episode array requires its protocol.json")
    protocol = read_json(protocol_path)
    manifest = protocol.get("checkpoint_manifest") or {}
    seed = first(manifest, ["seed", "training_seed", "train_seed"])
    if seed is None:
        checkpoint_directory = Path(protocol.get("run_dir", "")).name
        if checkpoint_directory.startswith("seed_"):
            seed = checkpoint_directory.removeprefix("seed_")
    if seed is None:
        raise ValueError(f"{protocol_path}: cannot identify training seed")
    context = {"task": protocol["task"], "train_seed": seed,
               "eval_mode": "stochastic", "opponent_label": "fixed trained checkpoint"}
    rows, exclusions = [], []
    for row in read_json(path):
        reason = None
        if not row.get("complete", False):
            reason = "incomplete_episode"
        elif row.get("event_parse_errors"):
            reason = "event_parse_errors"
        if reason:
            exclusions.append({"source": str(path.resolve()), "train_seed": str(seed),
                               "task": protocol["task"], "policy": row.get("policy"),
                               "evaluation_seed": row.get("evaluation_seed"), "reason": reason})
            continue
        rows.append(canonical_episode(row, context, path))
    return rows, exclusions, {"path": str(protocol_path.resolve()), "sha256": sha256(protocol_path),
                              "kind": "diagnostic_protocol", "native_duration_note": protocol.get("native_duration_note")}


def load_diagnostics(paths):
    episodes, sources, exclusions = [], [], []
    for path in paths:
        candidates = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for candidate in candidates:
            payload = read_json(candidate)
            if candidate.name == "episodes.json" and isinstance(payload, list):
                extracted, rejected, protocol_source = load_behavior_run(candidate)
                exclusions.extend(rejected)
                sources.append(protocol_source)
            else:
                extracted = diagnostic_episodes(payload, candidate)
            if extracted:
                episodes.extend(extracted)
                sources.append({"path": str(candidate.resolve()), "sha256": sha256(candidate),
                                "episode_count": len(extracted)})
    if not episodes:
        raise ValueError("No episode-level diagnostic data found. Summary-only or old return-only files are insufficient.")
    seen = set()
    for row in episodes:
        key = tuple(row[k] for k in ("task", "train_seed", "horizon", "eval_mode", "opponent", "policy", "episode_seed"))
        if key in seen:
            raise ValueError(f"Duplicate episode key {key}; do not mix merged and individual files.")
        seen.add(key)
    return episodes, sources, exclusions


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#aab2bb", "axes.labelcolor": "#293744",
        "text.color": "#293744", "xtick.color": "#566473", "ytick.color": "#566473",
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


GROUP_KEYS = ("task", "horizon", "eval_mode", "opponent")
METRICS = ("focal_return", "interaction_occurred", "interaction_count",
           "restricted_first_interaction_step", "resource_collect_count")


def summarize_diagnostics(episodes):
    groups = defaultdict(list)
    for row in episodes:
        key = tuple(row[k] for k in (*GROUP_KEYS, "train_seed", "policy"))
        groups[key].append(row)
    summaries = []
    for key, rows in sorted(groups.items()):
        entry = dict(zip((*GROUP_KEYS, "train_seed", "policy"), key))
        entry["episodes"] = len(rows)
        for metric in METRICS:
            values = [row[metric] for row in rows if row.get(metric) is not None]
            entry[f"{metric}_n"] = len(values)
            # Missing instrumentation must not be mistaken for a zero count.
            entry[f"{metric}_mean"] = float(np.mean(values)) if len(values) == len(rows) else None
        summaries.append(entry)
    return summaries


def panel_title(key):
    task, horizon, mode, opponent = key
    duration = "Native-duration extrapolation" if "horizon=native" in mode else f"{horizon:,}-frame evaluation"
    return f"{TASK_LABELS.get(task, task)} · {duration}\nActions: {mode} · Opponent: {opponent}"


def policy_order(rows):
    labels = {row["policy"] for row in rows}
    return sorted(labels, key=lambda label: (0 if label == "untrained" else 1 if label == "trained" else 2, label))


def paired_seed_points(ax, rows, metric):
    policies = policy_order(rows)
    seeds = sorted({row["train_seed"] for row in rows})
    lookup = {(row["train_seed"], row["policy"]): row.get(metric + "_mean") for row in rows}
    has_data = False
    for seed_index, seed in enumerate(seeds):
        offset = 0 if len(seeds) == 1 else (seed_index / (len(seeds) - 1) - .5) * .12
        available = [(i, lookup.get((seed, policy))) for i, policy in enumerate(policies)]
        available = [(i, value) for i, value in available if value is not None]
        if not available:
            continue
        has_data = True
        xs, ys = zip(*available)
        ax.plot(np.asarray(xs) + offset, ys, color="#c5cdd2", lw=.9, alpha=.85, zorder=1)
        for i, value in available:
            ax.scatter(i + offset, value, color=COLORS[i % len(COLORS)], s=31, alpha=.72, zorder=2)
    if not has_data:
        ax.text(.5, .5, "Metric not instrumented\nNo values inferred from reward", transform=ax.transAxes,
                ha="center", va="center", color="#697780")
    for i, policy in enumerate(policies):
        values = [lookup[(seed, policy)] for seed in seeds if lookup.get((seed, policy)) is not None]
        if values:
            ax.scatter(i, float(np.mean(values)), marker="D", s=66,
                       color=COLORS[i % len(COLORS)], edgecolor="white", linewidth=.8, zorder=3)
    ax.set_xticks(range(len(policies)), [POLICY_LABELS.get(p, p) for p in policies], rotation=15)
    ax.set_xlim(-.45, max(len(policies) - .55, .55))
    ax.grid(axis="y", alpha=.13)


def diagnostic_figures(summaries, output):
    grouped = defaultdict(list)
    for row in summaries:
        grouped[tuple(row[k] for k in GROUP_KEYS)].append(row)
    for index, (key, rows) in enumerate(sorted(grouped.items()), 1):
        fig, axes = plt.subplots(1, 4, figsize=(15, 4.7), layout="constrained")
        for ax, metric, label in zip(axes, (
            "focal_return", "interaction_occurred", "restricted_first_interaction_step", "resource_collect_count",
        ), ("Native return per episode", "Episodes with an interaction",
            "First interaction / episode end (frames)", "Resources collected per episode")):
            paired_seed_points(ax, rows, metric)
            ax.set_ylabel(label)
        axes[0].axhline(0, color="#b9c0c5", lw=.8, ls="--")
        axes[1].set_ylim(-.04, 1.04)
        axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        axes[2].set_ylim(bottom=0)
        axes[3].set_ylim(bottom=0)
        fig.suptitle("Melting Pot quality diagnostic\n" + panel_title(key), fontsize=14, fontweight="bold")
        fig.supxlabel(
            "Small points: training-seed means; diamonds: equally weighted mean over seeds. Lines connect the same seed.\n"
            "Waiting time is capped at each episode's end when no interaction occurs. This diagnostic does not compare PIVOT with baselines.",
            fontsize=8, color="#64727c")
        save_figure(fig, output, f"quality_diagnostic_{index:02d}")


def paired_diagnostic_differences(episodes):
    groups = defaultdict(dict)
    for row in episodes:
        key = tuple(row[k] for k in (*GROUP_KEYS, "train_seed", "episode_seed"))
        groups[key][row["policy"]] = row
    diffs = defaultdict(list)
    for key, conditions in groups.items():
        if "trained" in conditions and "untrained" in conditions:
            diffs[key[:-1]].append(conditions["trained"]["focal_return"] - conditions["untrained"]["focal_return"])
    return [{**dict(zip((*GROUP_KEYS, "train_seed"), key)), "paired_episodes": len(values),
             "trained_minus_untrained_native_return": float(np.mean(values))}
            for key, values in sorted(diffs.items())]


def bootstrap_mean(values):
    """Exploratory interval over independent training seeds, not over episodes."""
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    if len(x) < 2:
        return mean, None, None
    rng = np.random.default_rng(20260915)
    estimates = x[rng.integers(0, len(x), size=(10000, len(x)))].mean(axis=1)
    low, high = np.quantile(estimates, [.025, .975])
    return mean, float(low), float(high)


def validate_e5(payload):
    metadata = payload.get("metadata", {})
    for key in ("audit_independent", "selection_frozen_before_audit"):
        if metadata.get(key) is not True:
            raise ValueError(f"E5 plot requires metadata.{key}=true; no method-performance plot produced.")
    if "selection_episode_seeds" in metadata and "audit_episode_seeds" in metadata:
        overlap = set(metadata["selection_episode_seeds"]) & set(metadata["audit_episode_seeds"])
        if overlap:
            raise ValueError("E5 metadata reports overlapping selection and audit episode seeds")
    rows = payload.get("rows")
    if not rows:
        raise ValueError("E5 file has no independently scored decision rows")
    normalized, seen = [], set()
    for row in rows:
        row = dict(row)
        if "actual_queries" not in row and "query_packages_used" in row:
            row["actual_queries"] = row["query_packages_used"]
        for key in ("task", "seed", "method", "budget", "actual_queries", "audit_gain"):
            if key not in row:
                raise ValueError(f"E5 row missing {key}")
        entry = {key: row[key] for key in ("task", "seed", "method", "budget", "actual_queries", "audit_gain")}
        entry["task"], entry["seed"], entry["method"] = map(str, (entry["task"], entry["seed"], entry["method"]))
        for key in ("budget", "actual_queries", "audit_gain"):
            entry[key] = finite(entry[key], key)
        if not 0 <= entry["actual_queries"] <= entry["budget"]:
            raise ValueError("E5 actual query count outside stated budget")
        key = tuple(entry[k] for k in ("task", "seed", "method", "budget"))
        if key in seen:
            raise ValueError(f"Duplicate scored E5 decision {key}; aggregate audit replicates before plotting")
        seen.add(key)
        normalized.append(entry)
    return normalized, metadata


def load_e5(path):
    return validate_e5(read_json(path))


def load_hierarchical_panels(paths):
    """Verify real per-panel selection seals and independent audit artifacts."""
    decisions, candidates, sources, excluded = [], [], [], []
    contract, test_seeds, calibration_seeds = None, set(), set()
    for path in paths:
        protocol = read_json(path / "protocol.json")
        config = protocol["config"]
        if config.get("diagnostic_only", False):
            excluded.append({"path": str(path), "reason": "diagnostic_only_K2_smoke"})
            continue
        summary = read_json(path / "summary.json")
        if config["phase"] != "test" or summary.get("status") != "complete":
            raise ValueError(f"{path}: only completed test panels can enter E5 plots")
        current_contract = protocol["calibration_contract"]
        if contract is not None and current_contract != contract:
            raise ValueError("Do not pool hierarchical panels with different experimental contracts")
        contract = current_contract
        seed = config["seed"]
        if seed in test_seeds:
            raise ValueError(f"Duplicate E5 test seed {seed}")
        test_seeds.add(seed)
        calibration_seeds.update(protocol["calibration_seeds"])
        seal = read_json(path / "selection_seal.json")
        if seal.get("audit_opened") is not False or seal.get("audit_generated") is not False:
            raise ValueError(f"{path}: selection seal does not declare audit unopened")
        for field, name in (("decisions_sha256", "decisions_frozen.json"),
                            ("candidate_sha256", "candidates.json"),
                            ("features_sha256", "features.json"), ("posterior_sha256", "posterior.json")):
            if sha256(path / name) != seal[field]:
                raise ValueError(f"{path}: selection-sealed {name} hash mismatch")
        isolation = read_json(path / "isolation.json")
        if isolation.get("overlap_count") != 0:
            raise ValueError(f"{path}: selection and audit overlap")
        if sha256(path / "seeds.json") != isolation["seed_registry_sha256"]:
            raise ValueError(f"{path}: seed registry hash mismatch")
        registry = read_json(path / "seeds.json")
        if not isinstance(registry, dict):
            raise ValueError(f"{path}: invalid seed registry")
        selection_stream, audit_stream = set(), set()
        for key, value in registry.items():
            components = json.loads(key)
            if components[0] == "selection":
                selection_stream.add(value)
            elif components[0] == "audit":
                audit_stream.add(value)
        if selection_stream & audit_stream:
            raise ValueError(f"{path}: independent recheck found overlapping random streams")
        frozen = read_json(path / "decisions_frozen.json")
        lookup = {(row["method"], row["budget"]): row for row in frozen}
        scored = read_json(path / "scored_decisions.json")
        for row in scored:
            frozen_row = lookup[(row["method"], row["budget"])]
            if any(row.get(key) != value for key, value in frozen_row.items()):
                raise ValueError(f"{path}: scored decision differs from its frozen original")
            provenance = row["audit_provenance"]
            if not provenance.get("response_training_independent") or not provenance.get("evaluation_independent_of_selection"):
                raise ValueError(f"{path}: audit independence missing")
            audit = path / "audit" / f"candidate_{row['selected_candidate_id']}" / "paired.json"
            if sha256(audit) != provenance["sha256"]:
                raise ValueError(f"{path}: selected candidate audit hash mismatch")
            if row["audit_gain"] != read_json(audit)["delta"]:
                raise ValueError(f"{path}: plotted gain differs from independently audited gain")
            decisions.append(row)
        features = read_json(path / "features.json")
        candidate_data = read_json(path / "candidates.json")
        for j in range(config["k"]):
            audit_path = path / "audit" / f"candidate_{j}" / "paired.json"
            audit = read_json(audit_path)
            candidate = {"task": config["task"], "seed": seed, "candidate": j,
                         "proxy_gain": features[j][0], "audit_gain": audit["delta"],
                         "candidate_stag_probability": candidate_data["candidate_probabilities"][j]}
            if audit.get("evaluation_scheme") == "four_strata_real_rollout_weighted_expectation":
                block_path = audit_path.with_name("evaluation_block.json")
                if sha256(block_path) != audit["evaluation_block"]["sha256"]:
                    raise ValueError(f"{path}: independent stratified evaluation block hash mismatch")
                block = read_json(block_path)
                if not block.get("complete") or block.get("training_reads_this_block") is not False:
                    raise ValueError(f"{path}: incomplete stratified block or evaluation used by training")
                if block["conditional_return_means"] != audit["conditional_return_means"]:
                    raise ValueError(f"{path}: copied conditional means disagree with block")
                for estimate_name in ("old_estimate", "new_estimate"):
                    estimate = audit[estimate_name]
                    weights = estimate["stratum_weights"]
                    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-10):
                        raise ValueError(f"{path}: mixture weights do not sum to one")
                    recomputed = sum(weight * block["conditional_return_means"][stratum]["focal_return"]
                                     for stratum, weight in weights.items())
                    if not math.isclose(recomputed, estimate["focal_return"], rel_tol=1e-10, abs_tol=1e-10):
                        raise ValueError(f"{path}: weighted deployment estimate is inconsistent")
                candidate["audit_estimator"] = "stratified_weighted_expected_return"
                candidate["response_training_replicates_per_branch"] = audit["independent_response_training_replicates_per_branch"]
                candidate["evaluation_episodes_per_stratum"] = audit["evaluation_episodes_per_stratum"]
                sources.append({"path": str(block_path.resolve()), "sha256": sha256(block_path), "kind": "independent_stratified_evaluation_block"})
            candidate.update({key: audit[key] for key in (
                "frozen_delta", "response_effect", "old_response_own_gain", "new_response_own_gain", "candidate_specific_effect") if key in audit})
            candidates.append(candidate)
            sources.append({"path": str(audit_path.resolve()), "sha256": sha256(audit_path), "kind": "independent_candidate_audit"})
        for name in ("protocol.json", "summary.json", "selection_seal.json", "decisions_frozen.json",
                     "scored_decisions.json", "isolation.json", "seeds.json", "features.json",
                     "candidates.json", "posterior.json"):
            source = path / name
            sources.append({"path": str(source.resolve()), "sha256": sha256(source), "kind": "hierarchical_E5_panel"})
    if not decisions:
        raise ValueError("No completed non-diagnostic hierarchical test panels; no E5 figure generated")
    if test_seeds & calibration_seeds:
        raise ValueError("Test and calibration training seeds overlap")
    metadata = {
        "audit_independent": True, "selection_frozen_before_audit": True,
        "freeze_scope": "each test panel independently; not a global freeze of all panels",
        "query_unit": "validation-query packages", "pivot_method": "pivot_voi", "random_method": "random_hf",
        "policy_scope": "Episode-level specialist-mixture REINFORCE; frozen low-level network",
        "evidence_scope": "restricted hierarchical-policy single-round E5C benchmark slice",
        "test_training_seeds": sorted(test_seeds), "calibration_training_seeds": sorted(calibration_seeds),
        "contract": contract,
        "evaluation_scheme": contract.get("evaluation_scheme", "sampled_episode_evaluation"),
    }
    return validate_e5({"metadata": metadata, "rows": decisions}), candidates, sources, excluded


def hierarchical_candidate_figure(rows, metadata, output):
    write_csv(output / "hierarchical_candidates.csv", rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["task"]].append(row)
    for task, task_rows in groups.items():
        seeds = sorted({row["seed"] for row in task_rows})
        fig, axes = plt.subplots(1, 2, figsize=(10.9, 4.9), layout="constrained")
        for i, seed in enumerate(seeds):
            items = [row for row in task_rows if row["seed"] == seed]
            axes[0].scatter([row["proxy_gain"] for row in items], [row["audit_gain"] for row in items],
                            s=37, color=COLORS[i % len(COLORS)], alpha=.7, edgecolors="white", lw=.4,
                            label=f"Seed {seed}")
        values = [row[key] for row in task_rows for key in ("proxy_gain", "audit_gain")]
        low, high = min(values), max(values)
        padding = max((high-low) * .12, .05)
        axes[0].plot([low-padding, high+padding], [low-padding, high+padding], "--", color="#aab3ba", lw=1)
        axes[0].axhline(0, color="#c3c9ce", lw=.8)
        axes[0].axvline(0, color="#c3c9ce", lw=.8)
        stratified = metadata.get("evaluation_scheme") == "four_strata_real_rollout_weighted_expectation"
        axes[0].set(xlabel="Stratified fixed-opponent backtest gain" if stratified else "Fixed-opponent backtest update gain",
                    ylabel="Stratified estimate of\nexpected deployment gain" if stratified else "Independent adaptive-deployment audit gain",
                    xlim=(low-padding, high+padding), ylim=(low-padding, high+padding))
        if len(seeds) <= 8:
            axes[0].legend(frameon=False, fontsize=7, ncols=2)
        mechanism_keys = [("response_effect", "Response effect on update gain"),
                          ("old_response_own_gain", "Old-policy responder reward change"),
                          ("new_response_own_gain", "New-policy responder reward change")]
        for index, (key, label) in enumerate(mechanism_keys):
            available = [row for row in task_rows if key in row]
            if len(available) != len(task_rows):
                continue
            means = [np.mean([row[key] for row in available if row["seed"] == seed]) for seed in seeds]
            center, lo, hi = bootstrap_mean(means)
            axes[1].scatter(means, np.full(len(means), index), s=24, color=COLORS[index+1], alpha=.4)
            if lo is not None:
                axes[1].errorbar(center, index, xerr=[[max(0,center-lo)], [max(0,hi-center)]],
                                 fmt="D", color=COLORS[index+1], capsize=3, ms=5)
            else:
                axes[1].scatter(center, index, marker="D", color=COLORS[index+1])
        axes[1].set_yticks(range(len(mechanism_keys)), [label for _, label in mechanism_keys], fontsize=8)
        axes[1].set_ylim(2.6, -.6)
        axes[1].set_xlabel("Native return difference")
        axes[1].axvline(0, color="#aab3ba", ls="--", lw=.8)
        axes[1].grid(axis="x", alpha=.12)
        fig.suptitle(f"Backtest versus adaptive deployment · {TASK_LABELS.get(task, task)}\n{metadata['policy_scope']}", fontsize=13, fontweight="bold")
        stratum_note = " One trained response per branch; stratum episodes are not response-training replicates." if stratified else ""
        fig.supxlabel(f"{len(task_rows)} candidate updates from {len(seeds)} test training seeds. Gains are noisy estimates, not true values.\n"
                      "Mechanisms are correlated contrasts from the same audit block; points average candidates within a seed, intervals use seed bootstrap."
                      + stratum_note,
                      fontsize=8, color="#64727c")
        save_figure(fig, output, "hierarchical_candidate_audit_" + task)


def e5_figures(rows, metadata, output):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["method"], row["budget"])].append(row)
    summary = []
    for (task, method, budget), items in sorted(grouped.items()):
        mean, low, high = bootstrap_mean([row["audit_gain"] for row in items])
        summary.append({"task": task, "method": method, "budget": budget,
                        "mean_actual_queries": float(np.mean([row["actual_queries"] for row in items])),
                        "training_seeds": len(items), "audit_gain_mean": mean,
                        "exploratory_ci95_low": low, "exploratory_ci95_high": high})
    write_csv(output / "e5_by_seed.csv", rows)
    write_csv(output / "e5_summary.csv", summary)
    differences = []
    pivot_method = metadata.get("pivot_method", "pivot_voi")
    random_method = metadata.get("random_method", "random_hf")
    for task in sorted({row["task"] for row in rows}):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), layout="constrained")
        task_rows = [row for row in summary if row["task"] == task]
        methods = sorted({row["method"] for row in task_rows})
        for color_index, method in enumerate(methods):
            items = sorted((row for row in task_rows if row["method"] == method), key=lambda r: r["budget"])
            xs = [row["mean_actual_queries"] for row in items]
            ys = [row["audit_gain_mean"] for row in items]
            color = METHOD_COLORS.get(method, COLORS[color_index % len(COLORS)])
            axes[0].plot(xs, ys, marker="D" if method == "pivot_voi" else "o", linestyle="-",
                         markerfacecolor="none" if method == "pivot_voi" else color,
                         label=METHOD_LABELS.get(method, method), color=color, ms=5 if method == "pivot_voi" else 4)
            if all(row["exploratory_ci95_low"] is not None for row in items):
                axes[0].fill_between(xs, [row["exploratory_ci95_low"] for row in items],
                                     [row["exploratory_ci95_high"] for row in items], color=color, alpha=.10)
        cost_unit = metadata.get("query_unit", "validation-query packages")
        stratified = metadata.get("evaluation_scheme") == "four_strata_real_rollout_weighted_expectation"
        axes[0].set(xlabel=f"Mean {cost_unit} used", ylabel="Stratified estimate of\nexpected deployment gain" if stratified else "Independent audit: selected update gain")
        axes[0].legend(frameon=False, fontsize=8)
        axes[0].set_xticks(sorted({row["mean_actual_queries"] for row in task_rows}))
        for budget in sorted({row["budget"] for row in task_rows}):
            method_rows = {method: {row["seed"]: row for row in rows
                                   if row["task"] == task and row["method"] == method and row["budget"] == budget}
                           for method in (pivot_method, random_method)}
            left, right = method_rows[pivot_method], method_rows[random_method]
            if not left or not right:
                continue
            if set(left) != set(right):
                raise ValueError(f"Unmatched training seeds for equal-budget comparison: {task}, {budget}")
            values = []
            for seed in sorted(left):
                difference = left[seed]["audit_gain"] - right[seed]["audit_gain"]
                values.append(difference)
                differences.append({"task": task, "budget": budget, "seed": seed,
                                    "comparison": f"{pivot_method} minus {random_method}", "audit_gain_difference": difference})
            mean, low, high = bootstrap_mean(values)
            axes[1].scatter(np.full(len(values), budget), values, s=19, color="#137c8b", alpha=.30)
            if low is not None:
                axes[1].errorbar(budget, mean, yerr=[[max(0, mean-low)], [max(0, high-mean)]],
                                 fmt="D", color="#137c8b", capsize=3, ms=5)
            else:
                axes[1].scatter(budget, mean, marker="D", color="#137c8b")
        if not any(row["task"] == task for row in differences):
            axes[1].text(.5, .5, "No matched PIVOT / random rows\nEqual-budget difference not estimated",
                         ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set(xlabel="Shared validation-query budget",
                    ylabel=f"Estimated deployment-gain difference\n{METHOD_LABELS.get(pivot_method, pivot_method)} minus {METHOD_LABELS.get(random_method, random_method)}")
        axes[1].set_xticks(sorted({row["budget"] for row in task_rows}))
        for ax in axes:
            ax.axhline(0, color="#aab3ba", ls="--", lw=.8)
            ax.grid(alpha=.13)
        scope_note = "\n" + metadata["policy_scope"] if metadata.get("policy_scope") else ""
        fig.suptitle(f"Melting Pot E5-style adaptive extension · {TASK_LABELS.get(task, task)}" + scope_note, fontsize=13, fontweight="bold")
        fig.supxlabel("Intervals: exploratory bootstrap over test training seeds. Each point uses an independently scored frozen decision.\n"
                      "Query cost excludes candidate training, calibration and independent audit. "
                      + ("Four skill-pair strata estimate mixture returns; stratum episodes are not independent training seeds." if stratified
                         else "Single-round evidence; not a full official-suite evaluation."),
                      fontsize=8, color="#64727c")
        save_figure(fig, output, "e5_" + task)
    write_csv(output / "e5_equal_budget_differences.csv", differences)
    return summary, differences


def reference_figures(paths, output):
    """Official specialist controls have no new training seeds or training gain."""
    references, sources, excluded = [], [], []
    for path in paths:
        file = path / "episodes.json" if path.is_dir() else path
        protocol_file = file.parent / "protocol.json"
        protocol = read_json(protocol_file)
        if protocol.get("eval_mode") != "official_specialist_reference":
            raise ValueError(f"{file}: --reference is only for the official specialist control schema")
        for raw in read_json(file):
            if not raw.get("complete", False) or raw.get("event_parse_errors"):
                excluded.append({"source": str(file), "episode_seed": raw.get("evaluation_seed"),
                                 "policy": raw.get("policy"), "reason": "incomplete_or_event_parse_error"})
                continue
            # Canonical event validation is reused; the placeholder is removed
            # immediately and is never used for seed statistics or inference.
            copy = dict(raw)
            copy["train_seed"] = "validation_placeholder"
            row = canonical_episode(copy, {"task": protocol["task"]}, file)
            row.pop("train_seed")
            row["matchup"] = raw["matchup"]
            row["analysis_unit"] = "reference_evaluation_episode"
            references.append(row)
        for source in (file, protocol_file):
            sources.append({"path": str(source.resolve()), "sha256": sha256(source), "kind": "official_reference_control"})
    if not references:
        raise ValueError("No complete valid official reference episodes")
    write_csv(output / "reference_episodes.csv", references)
    write_csv(output / "reference_exclusions.csv", excluded)
    groups = defaultdict(list)
    for row in references:
        groups[(row["task"], row["horizon"])].append(row)
    summary = []
    for index, ((task, horizon), rows) in enumerate(sorted(groups.items()), 1):
        matchups = sorted({row["matchup"] for row in rows})
        fig, axes = plt.subplots(1, 4, figsize=(14.2, 4.9), layout="constrained")
        for i, matchup in enumerate(matchups):
            items = [row for row in rows if row["matchup"] == matchup]
            entry = {"task": task, "horizon": horizon, "matchup": matchup, "episodes": len(items)}
            for ax, metric in zip(axes, ("focal_return", "interaction_occurred", "interaction_count", "resource_collect_count")):
                values = [row[metric] for row in items if row.get(metric) is not None]
                if len(values) != len(items):
                    raise ValueError(f"Reference {metric} is incompletely instrumented")
                entry[metric + "_mean"] = float(np.mean(values))
                offset = np.linspace(-.075, .075, len(values)) if len(values) > 1 else np.zeros(1)
                ax.scatter(i + offset, values, color=COLORS[i % len(COLORS)], alpha=.45, s=26)
                ax.scatter(i, np.mean(values), marker="D", s=56, color=COLORS[i % len(COLORS)], edgecolor="white", lw=.7)
            summary.append(entry)
        for ax, label in zip(axes, ("Native focal return", "Episodes with an interaction", "Interactions per episode", "Focal resources collected")):
            ax.set_xticks(range(len(matchups)), matchups, rotation=15)
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=.13)
        axes[1].set_ylim(-.04, 1.04)
        axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        fig.suptitle(f"Official specialist behavior controls · {TASK_LABELS.get(task, task)}\nEpisode cap: {horizon:,} frames · no new training", fontsize=14, fontweight="bold")
        fig.supxlabel("Small points: evaluation episodes; diamonds: within-control episode mean. These are reference policies, not additional training seeds.\n"
                      "Controls check real environment interaction. They do not establish PPO learning, PIVOT superiority, or a matched-observation comparison.",
                      fontsize=8, color="#64727c")
        save_figure(fig, output, f"official_reference_control_{index:02d}")
    write_csv(output / "reference_summary.csv", summary)
    return references, summary, sources, excluded


def registered_interval(metric):
    """Read already registered analysis intervals without another bootstrap."""
    center = finite(metric["mean"], "registered mean")
    bounds = metric["ci95"]
    if len(bounds) != 2:
        raise ValueError("Registered interval must have exactly two bounds")
    low, high = [finite(value, "registered interval bound") for value in bounds]
    if low > high:
        raise ValueError("Registered interval bounds are reversed")
    return center, low, high


def load_registered_analysis(directory):
    summary_path, manifest_path = directory / "summary.json", directory / "manifest.json"
    report, manifest = read_json(summary_path), read_json(manifest_path)
    if report.get("status") != "complete" or report.get("formal_effects_available") is not True:
        raise ValueError("Formal figures require a fully validated complete analysis; partial-panel effects are not plotted")
    if report.get("all_planned_test_seeds_used") is not True or report.get("validation_passed_panels") != 28:
        raise ValueError("Formal figures require all 8 calibration and 20 test panels")
    if manifest.get("status") != "complete":
        raise ValueError("Analysis hash manifest is not complete")
    for name, expected in manifest["output_sha256"].items():
        candidate = (directory / name).resolve()
        if not candidate.is_relative_to(directory.resolve()) or sha256(candidate) != expected:
            raise ValueError(f"Analysis output changed or path is unsafe: {name}")
    if "summary.json" not in manifest["output_sha256"]:
        raise ValueError("Analysis summary is absent from output hash manifest")
    protocol = report["registered_protocol"]
    test_seeds = protocol["test_seeds"]
    if len(test_seeds) != 20 or len(set(test_seeds)) != 20 or len(protocol["calibration_seeds"]) != 8:
        raise ValueError("Unexpected registered training-seed allocation")
    if protocol["candidate_count"] != 4 or protocol["budgets"] != [0, 1, 2, 4]:
        raise ValueError("Unexpected registered candidate count or budget grid")
    expected_methods = {(method, budget) for method in METHOD_LABELS
                        for budget in ([0] if method == "proxy_only" else protocol["budgets"])}
    if {(row["method"], row["budget"]) for row in report["methods"]} != expected_methods:
        raise ValueError("Analysis method grid is incomplete")
    for method in report["methods"]:
        values = [row for row in report["method_seed_results"]
                  if row["method"] == method["method"] and row["budget"] == method["budget"]]
        if len(values) != 20 or {row["seed"] for row in values} != set(test_seeds) or method["n_seeds"] != 20:
            raise ValueError("A plotted method is missing or duplicating training seeds")
        center, _, _ = registered_interval(method["audit_gain"])
        if not math.isclose(center, float(np.mean([row["audit_gain"] for row in values])), rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("Plotted method mean differs from supplied seed-level scores")
    expected_contrasts = {"pivot_voi_minus_proxy_only", "pivot_voi_minus_random_hf"}
    if {row["contrast"] for row in report["primary_contrasts"]} != expected_contrasts:
        raise ValueError("The two registered primary contrasts are missing")
    for contrast in report["primary_contrasts"]:
        center, _, _ = registered_interval(contrast["difference"])
        values = contrast["paired_seed_differences"]
        if len(values) != 20 or not math.isclose(center, float(np.mean(values)), rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("Plotted primary contrast differs from paired seed scores")
    expected_checkpoints = set(range(protocol["response_train_episodes"] + 1))
    if {row["response_training_episodes"] for row in report["response_learning_progress"]} != expected_checkpoints:
        raise ValueError("Response checkpoint curve is incomplete")
    sources = [{"path": str(path.resolve()), "sha256": sha256(path), "kind": "validated_analysis"}
               for path in (summary_path, manifest_path)]
    return report, sources


def draw_registered_band(ax, rows, xkey, metric, color, label, marker="o"):
    xs = [row[xkey] for row in rows]
    points = [registered_interval(row[metric]) for row in rows]
    ax.plot(xs, [point[0] for point in points], marker=marker, color=color, lw=1.7,
            markersize=4.5, label=label, markerfacecolor="none" if marker == "D" else color)
    ax.fill_between(xs, [point[1] for point in points], [point[2] for point in points], color=color, alpha=.10)


def registered_budget_figure(report, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8), layout="constrained")
    methods = ["pivot_voi", "random_hf", "global_voi", "paired_lucb", "proxy_only"]
    for method in methods:
        rows = sorted([row for row in report["methods"] if row["method"] == method], key=lambda r: r["budget"])
        rows = [{**row, "query_episodes": row["logical_query_episodes"]["mean"]} for row in rows]
        draw_registered_band(axes[0], rows, "query_episodes", "audit_gain", METHOD_COLORS[method],
                             METHOD_LABELS[method], marker="D" if method == "pivot_voi" else "o")
        if len(rows) == 1:
            center, low, high = registered_interval(rows[0]["audit_gain"])
            axes[0].vlines(rows[0]["query_episodes"], low, high, color=METHOD_COLORS[method], lw=1.6)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    axes[0].set(xlabel="Validation cost charged per method (real episodes)",
                ylabel="Stratified estimate of\nexpected deployment gain", title="A  Validation budget frontier")
    episode_ticks = sorted({row["logical_query_episodes"]["mean"] for row in report["methods"]})
    axes[0].set_xticks(episode_ticks)
    per_query = 2 * report["registered_protocol"]["response_train_episodes"] + 4 * report["registered_protocol"]["stratified_replicates_per_skill_pair"]
    top = axes[0].secondary_xaxis("top", functions=(lambda x: x / per_query, lambda x: x * per_query))
    top.set_xticks([0, 1, 2, 4])
    top.set_xlabel("Query packages (four candidate updates)", fontsize=9)
    by_name = {row["contrast"]: row for row in report["primary_contrasts"]}
    contrast_names = ["pivot_voi_minus_random_hf", "pivot_voi_minus_proxy_only"]
    contrast_labels = ["PIVOT (2) − Random-HF (2)\nSame query budget", "PIVOT (2) − Proxy only (0)\nAdditional validation cost"]
    for index, name in enumerate(contrast_names):
        row = by_name[name]
        values = row["paired_seed_differences"]
        center, low, high = registered_interval(row["difference"])
        color = "#137c8b" if index == 0 else "#ad7043"
        # Deterministic offsets display the 20 independent seed-level pairs.
        axes[1].scatter(index + np.linspace(-.10, .10, len(values)), values, color=color, s=23, alpha=.36)
        axes[1].vlines(index, low, high, color=color, lw=2.1)
        axes[1].hlines([low, high], index-.045, index+.045, color=color, lw=1.2)
        axes[1].scatter(index, center, marker="D", s=64, color=color, edgecolor="white", lw=.6, zorder=3)
        contrast_labels[index] += f"\nHolm p = {row['test']['holm_p_two_tests']:.3g}"
    axes[1].set_xticks([0, 1], contrast_labels, fontsize=9)
    axes[1].set(xlim=(-.42, 1.42), ylabel="Paired difference in estimated deployment gain",
                title="B  Two predeclared primary comparisons",)
    for ax in axes:
        ax.axhline(0, color="#adb7bf", lw=.9, ls="--")
        ax.grid(axis="y", alpha=.13)
    fig.suptitle("Single-round Melting Pot E5 benchmark · Repeated Stag Hunt\nEpisode-level specialist mixture; frozen low-level network", fontsize=14, fontweight="bold")
    both_cross_zero = all(registered_interval(row["difference"])[1] <= 0 <= registered_interval(row["difference"])[2]
                          for row in report["primary_contrasts"])
    primary_note = " Both primary intervals include zero." if both_cross_zero else ""
    fig.supxlabel("20 test training seeds; bands/bars are the registered 95% seed-bootstrap intervals." + primary_note + "\n"
                  "Primary p-values: exact paired sign flips, Holm over these two contrasts. Query costs exclude candidate training, calibration, proxy and audit.",
                  fontsize=8, color="#64727c")
    save_figure(fig, output, "melting_e5_budget_frontier")


def registered_candidate_figure(report, output):
    rows = report["candidate_seed_results"]
    seeds = report["registered_protocol"]["test_seeds"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), layout="constrained")
    for index, seed in enumerate(seeds):
        candidates = [row for row in rows if row["seed"] == seed]
        axes[0].scatter([row["proxy_delta"] for row in candidates], [row["audit_gain"] for row in candidates],
                        s=34, color=COLORS[index % len(COLORS)], alpha=.66, edgecolors="white", lw=.35)
    values = [row[key] for row in rows for key in ("proxy_delta", "audit_gain")]
    lo, hi = min(values), max(values)
    pad = max((hi-lo)*.1, .03)
    axes[0].plot([lo-pad, hi+pad], [lo-pad, hi+pad], "--", color="#aab3ba", lw=1)
    axes[0].axhline(0, color="#bdc5cc", lw=.7)
    axes[0].axvline(0, color="#bdc5cc", lw=.7)
    axes[0].set(xlabel="Stratified fixed-opponent backtest gain", ylabel="Stratified estimate of\nexpected deployment gain",
                xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad), title="A  Backtest versus adaptive deployment")
    effects = [("response_effect", "Response effect on update gain"),
               ("candidate_specific_effect", "Candidate-specific response effect"),
               ("old_response_own_gain", "Responder reward change: old target"),
               ("new_response_own_gain", "Responder reward change: new target")]
    for index, (key, _) in enumerate(effects):
        per_seed = [float(np.mean([row[key] for row in rows if row["seed"] == seed])) for seed in seeds]
        center, low, high = registered_interval(report["candidate_mechanism_means"][key])
        color = COLORS[index + 1]
        axes[1].scatter(per_seed, np.full(len(seeds), index), s=20, color=color, alpha=.32)
        axes[1].hlines(index, low, high, color=color, lw=2)
        axes[1].vlines([low, high], index-.055, index+.055, color=color, lw=1)
        axes[1].scatter(center, index, marker="D", s=50, color=color, edgecolor="white", lw=.6, zorder=3)
    axes[1].set_yticks(range(len(effects)), [label for _, label in effects], fontsize=8)
    axes[1].set(xlabel="Native return difference", ylim=(3.6, -.6), title="B  Correlated audit-block contrasts")
    axes[1].axvline(0, color="#adb7bf", lw=.9, ls="--")
    axes[1].grid(axis="x", alpha=.13)
    fig.suptitle("What changes after the opponent adapts?\nOne-dimensional specialist mixtures on the real Melting Pot substrate", fontsize=14, fontweight="bold")
    fig.supxlabel("80 candidate updates from 20 test training seeds. Candidate dots are noisy stratified estimates, not independent training seeds or known true values.\n"
                  "Mechanism points average four candidates within each seed; bars use registered intervals. All contrasts reweight the same audit block and cost no new episodes.",
                  fontsize=8, color="#64727c")
    save_figure(fig, output, "melting_e5_candidate_audit")


def registered_response_figure(report, output):
    rows = sorted(report["response_learning_progress"], key=lambda row: row["response_training_episodes"])
    key = "response_training_episodes"
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.1), layout="constrained")
    draw_registered_band(axes[0], rows, key, "audit_gain_at_saved_checkpoint", "#137c8b", "Expected update gain")
    draw_registered_band(axes[1], rows, key, "gain_change_from_t0", "#755ba1", "Change from initial response")
    draw_registered_band(axes[2], rows, key, "q_old", "#78909c", "Opponent trained on old policy")
    draw_registered_band(axes[2], rows, key, "q_new", "#ad7043", "Opponent trained on candidate")
    axes[0].set(ylabel="Stratified estimate of\nexpected deployment gain", title="A  Saved-checkpoint deployment gain")
    axes[1].set(ylabel="Expected update-gain change from t = 0", title="B  Change induced by response learning")
    axes[2].set(ylabel="Opponent probability of selecting stag", ylim=(0, 1), title="C  Learned response probabilities")
    axes[2].legend(frameon=False, fontsize=8, loc="best")
    for ax in axes:
        ax.set_xlabel("Response training episodes completed")
        ax.set_xticks([row[key] for row in rows])
        ax.grid(axis="y", alpha=.13)
    for ax in axes[:2]:
        ax.axhline(0, color="#adb7bf", lw=.9, ls="--")
    fig.suptitle("Response-learning checkpoints t = 0…4\nRetrospective reweighting of the same independent audit block", fontsize=14, fontweight="bold")
    fig.supxlabel("Real sampled episodes train the high-level response mixture; the low-level specialist network remains frozen.\n"
                  "Each curve averages four candidates within each of 20 test seeds. Registered seed-bootstrap bands; checkpoint evaluations are correlated and add no new environment episodes.",
                  fontsize=8, color="#64727c")
    save_figure(fig, output, "melting_e5_response_checkpoints")


def plot_registered_analysis(directory, output):
    report, sources = load_registered_analysis(directory)
    output.mkdir(parents=True, exist_ok=False)
    setup_style()
    registered_budget_figure(report, output)
    registered_candidate_figure(report, output)
    registered_response_figure(report, output)
    payload_keys = ("analysis_version", "registered_protocol", "statistics", "methods", "primary_contrasts",
                    "candidate_metrics", "candidate_mechanism_means", "candidate_seed_results", "response_learning_progress",
                    "accounting", "limitations", "status", "formal_effects_available", "all_planned_test_seeds_used")
    payload = {key: report[key] for key in payload_keys}
    (output / "plot_data_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    notes = ["# Melting Pot 主实验图", "",
             "三组图直接读取已完整验收的分析文件和其中的注册置信区间；绘图过程没有重新 bootstrap，也没有追加显著性检验。",
             "", "1. **预算前沿与两项主比较**：左图展示每种方法的部署增益与验证成本；右图分别区分同预算比较和相对零查询的额外验证收益。",
             "2. **候选回测与部署、响应机制**：80 个候选来自 20 个独立测试训练种子，候选点不是额外的独立训练样本。机制差值来自同一 audit block 的相关重加权。",
             "3. **t0–t4 响应学习检查点**：使用真实训练保存的上层概率，在同一独立 audit block 上回顾评估各个检查点；不是另外启动五套独立实验。",
             "", "策略空间只有每回合选择 stag/hare 技能的一维混合概率，低层官方神经网络冻结；上层使用真实抽样环境回报做 REINFORCE。",
             "部署收益来自四种技能配对的真实分层评估，每种两局，按已冻结概率加权估计。每个候选、每个 old/new 目标分支只有一次独立响应训练；评估回合不能冒充响应训练重复。",
             "", "图中区间均以 20 个测试训练种子为统计单位。两项主对比采用精确双侧配对符号翻转检验，并仅对这两项做 Holm 校正。",
             "原始对局和权重留在云端；plot_data_summary.json 含重画这些图所需的小数据与原始分析定义。", ""]
    for row in report["primary_contrasts"]:
        center, low, high = registered_interval(row["difference"])
        label = "PIVOT2 − Random-HF2（同查询成本）" if "random_hf" in row["contrast"] else "PIVOT2 − Proxy0（增加验证成本）"
        notes.append(f"- {label}：均值 {center:+.6f}，95% 区间 [{low:+.6f}, {high:+.6f}]，Holm p = {row['test']['holm_p_two_tests']:.6g}。")
    (output / "读图说明.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    manifest = {"status": "complete_registered_analysis_figures", "plot_script_sha256": sha256(__file__),
                "created_utc": datetime.now(timezone.utc).isoformat(), "input_files": sources,
                "statistical_intervals": "Copied directly from validated analysis; no re-bootstrap or new significance tests",
                "files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                          for path in sorted(output.iterdir()) if path.is_file()]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "status": manifest["status"],
                      "figure_sets": 3, "training_seeds": 20, "files": len(manifest["files"])+1}, ensure_ascii=False))


def write_readme(output, summaries, differences, e5_summary, exclusions, reference_summary):
    lines = ["# Melting Pot 本轮实测结果", "", "本包仅根据输入的原始实验文件生成；未插入模拟值，未把缺失事件记录记成零。", "",
             "## 质量诊断", "", "图中小点是每个训练种子的均值，菱形是对训练种子等权平均。连接线连接同一训练种子。",
             "回报使用环境的原生分数；交互率来自明确的交互事件计数，不能用非零回报代替。",
             "首交互等待时间在无交互时记到该回合结束，是受回合长度限制的诊断量，不是无条件的真实首次事件时间。",
             "native 时长评估保留原生终止规则，但原策略的剩余时间特征超过训练时长后钳为零，属于时长外推诊断；不能和 1000 帧结果混合，也不是重新训练过的原生时长策略。",
             f"本包排除了 {len(exclusions)} 个未完成或事件解析失败的回合，详情见 diagnostic_exclusions.csv（存在时）。", ""]
    grouped = defaultdict(list)
    for row in differences:
        grouped[tuple(row[k] for k in GROUP_KEYS)].append(row)
    if grouped:
        for key, rows in sorted(grouped.items()):
            values = [row["trained_minus_untrained_native_return"] for row in rows]
            positive = sum(value > 0 for value in values)
            lines.append(f"- {key[0]}，{key[1]} 帧，动作模式 {key[2]}，对手 {key[3]}："
                         f"{len(rows)} 个训练种子的配对原生回报差均值 {np.mean(values):+.4f}；"
                         f"{positive}/{len(rows)} 个种子的差值为正。此项是学习质量诊断，不能据此宣称 PIVOT 有优势。")
    else:
        lines.append("未找到 trained/untrained 两条件共享评估种子的完整配对，未报告配对训练收益。")
    if reference_summary:
        lines.extend(["", "## 官方参考行为对照", "", "reference_* 文件单独展示官方 specialist 参考策略。小点是评估回合，不能把这些回合当成独立训练种子。",
                      "这些对照用于检查环境能否出现有效交互，不证明旧 PPO 学会了任务，也不证明 PIVOT 优于基线。参考策略的输入和控制方式须以其 protocol.json 为准，不能默认与 PPO 具有相同观测条件。"])
        for row in reference_summary:
            lines.append(f"- {row['matchup']}，{row['episodes']} 回合：原生回报均值 {row['focal_return_mean']:.4f}，"
                         f"有交互回合占比 {row['interaction_occurred_mean']:.1%}，每回合交互 {row['interaction_count_mean']:.3f}。")
    lines.extend(["", "## PIVOT 与基线", ""])
    if e5_summary:
        lines.extend(["已提供冻结选择及独立 audit 的实测决策文件，生成 E5 风格单轮比较图。",
                      "独立性声明来自上游实验的元数据，本绘图脚本还检查已提供的选择与测试种子集合是否重叠。",
                      "各方法使用相同预算的比较与不同验证成本的比较须分别解释。置信区间为探索性的训练种子 bootstrap；不以是否显著决定保留任务。",
                      "这里只报告所选任务的 adaptive extension，不代表完整官方 Melting Pot 场景套件，也不是多轮闭环结果。"])
    else:
        lines.append("本包未输入完成独立 audit 的 E5 决策数据，因此没有生成 PIVOT 与基线的性能图。当前质量诊断不能证明或否定 PIVOT 的方法优势。")
    lines.extend(["", "## 文件", "", "- diagnostic_episodes.csv：逐回合的原生分数与行为事件。",
                  "- diagnostic_by_seed.csv：逐训练种子的统计。",
                  "- diagnostic_paired_gain.csv：共享评估种子的 trained / untrained 配对差（存在时）。",
                  "- quality_diagnostic_*.png / .pdf：质量诊断图。",
                  "- manifest.json：原始输入的 SHA-256、生成文件及数据范围。", ""])
    (output / "读图说明.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diagnostic", type=Path, nargs="+", default=[])
    parser.add_argument("--reference", type=Path, nargs="+", default=[], help="Separate official specialist controls; never treated as training seeds")
    parser.add_argument("--analysis", type=Path, help="Formal v2 figures from a complete validated analysis; use its registered intervals directly")
    parser.add_argument("--e5", type=Path, help="Optional independently audited E5 scored decisions")
    parser.add_argument("--e5-panel", type=Path, nargs="+", default=[], help="Complete hierarchical E5 test panel directories; seals and audits verified")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.analysis:
        if args.diagnostic or args.reference or args.e5 or args.e5_panel:
            parser.error("--analysis produces a separate formal-result package; do not mix diagnostic or raw-panel inputs")
        plot_registered_analysis(args.analysis, args.output)
        return
    if not args.diagnostic and not args.reference and not args.e5 and not args.e5_panel:
        parser.error("At least one diagnostic, reference, or independently audited E5 input is required")
    if args.e5 and args.e5_panel:
        parser.error("Use either --e5 or --e5-panel, not both")
    episodes, sources, exclusions = load_diagnostics(args.diagnostic) if args.diagnostic else ([], [], [])
    summaries = summarize_diagnostics(episodes)
    differences = paired_diagnostic_differences(episodes)
    # Validate E5 before producing any output, so an invalid audit cannot leave a
    # half-written package that looks like a completed result.
    e5_input = load_e5(args.e5) if args.e5 else None
    hierarchical_candidates, e5_exclusions = [], []
    if args.e5_panel:
        e5_input, hierarchical_candidates, e5_sources, e5_exclusions = load_hierarchical_panels(args.e5_panel)
        if e5_input[1].get("evaluation_scheme") == "four_strata_real_rollout_weighted_expectation":
            raise ValueError("Formal stratified-v2 figures require --analysis so tables and figures share the exact registered intervals")
        sources.extend(e5_sources)
    args.output.mkdir(parents=True, exist_ok=False)
    setup_style()
    write_csv(args.output / "diagnostic_episodes.csv", episodes)
    write_csv(args.output / "diagnostic_by_seed.csv", summaries)
    write_csv(args.output / "diagnostic_paired_gain.csv", differences)
    write_csv(args.output / "diagnostic_exclusions.csv", exclusions)
    diagnostic_figures(summaries, args.output)
    reference_summary, reference_episode_count = [], 0
    if args.reference:
        reference_rows, reference_summary, reference_sources, _ = reference_figures(args.reference, args.output)
        reference_episode_count = len(reference_rows)
        sources.extend(reference_sources)
    e5_summary = []
    if e5_input:
        e5_summary, _ = e5_figures(*e5_input, args.output)
        if args.e5:
            sources.append({"path": str(args.e5.resolve()), "sha256": sha256(args.e5), "kind": "scored_e5_decisions"})
        if hierarchical_candidates:
            hierarchical_candidate_figure(hierarchical_candidates, e5_input[1], args.output)
            write_csv(args.output / "e5_excluded_panels.csv", e5_exclusions)
            (args.output / "hierarchical_e5_protocol.json").write_text(json.dumps(e5_input[1], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_readme(args.output, summaries, differences, e5_summary, exclusions, reference_summary)
    if args.e5_panel:
        config = e5_input[1]["contract"]
        if config.get("evaluation_scheme") == "four_strata_real_rollout_weighted_expectation":
            evaluation_description = (
                f"候选训练 {config['candidate_train_episodes']} 个真实抽样回合；每支响应训练 {config['response_train_episodes']} 个真实抽样回合。"
                f"每个响应评估 block 包含四种 specialist 配对，每种 {config['stratum_replicates']} 个新真实回合，再按固定混合概率加权估计部署期望；"
                f"proxy 使用另一个独立 block，四种配对各 {config['proxy_stratum_replicates']} 回合，在该面板候选间共享。"
                "每个 candidate 的 old/new 目标各只有一次响应训练，而不是把每个评估回合当独立响应训练重复。"
                "机制差值是同一独立 audit block 的相关重加权，不是额外的独立实验。"
            )
        else:
            evaluation_description = (
                f"候选训练 {config['candidate_train_episodes']} 回合；每支响应训练 {config['response_train_episodes']} 回合；"
                f"每支响应评估 {config['eval_episodes']} 回合；proxy 使用 {config['proxy_episodes']} 回合。"
            )
        context = ["", "## 本轮层级策略实验的范围", "",
                   "只学习每个回合选择 stag/hare specialist 的 Bernoulli 混合概率，使用 REINFORCE 更新；官方底层神经网络保持冻结。",
                   "这项实验是受限层级策略上的单轮 E5 风格 adaptive extension，不代表旧 PPO 已被修好，也不是完整官方 Melting Pot 场景套件。",
                   f"测试训练种子：{e5_input[1]['test_training_seeds']}；校准训练种子：{e5_input[1]['calibration_training_seeds']}。",
                   f"每面板候选数：{config['k']}；每回合最长 {config['horizon']} 帧。",
                   evaluation_description,
                   "每个测试面板先封存其全部方法决策，再生成该面板的独立 audit；这不是所有面板一起全局封存。",
                   "作图前重新验证了决策、候选、特征、后验和种子清单的封存哈希，检查选择/测试随机流无重叠，"
                   "并确认绘图收益与选中候选的 audit 文件一致。", ""]
        with (args.output / "读图说明.md").open("a", encoding="utf-8") as stream:
            stream.write("\n".join(context))
    status = "quality_diagnostic_only"
    if e5_input:
        status = "diagnostic_and_e5_audit" if episodes else "hierarchical_e5_audit" if args.e5_panel else "e5_audit"
    elif reference_summary and not episodes:
        status = "official_reference_control_only"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "input_files": sources, "plot_script_sha256": sha256(__file__),
        "diagnostic_episodes": len(episodes), "diagnostic_seed_policy_groups": len(summaries),
        "excluded_episode_count": len(exclusions),
        "official_reference_control_groups": len(reference_summary),
        "official_reference_episodes": reference_episode_count,
        "e5_excluded_diagnostic_panels": e5_exclusions,
        "e5_audit_present": bool(e5_input),
        "limitations": ["Selected-task adaptive extension, not the full official suite",
                        "Quality diagnostics do not establish PIVOT superiority",
                        "Event instrumentation is never replaced with nonzero-reward counts"],
        "files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                  for path in sorted(args.output.iterdir()) if path.is_file()],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "status": manifest["status"],
                      "diagnostic_episodes": len(episodes), "reference_episodes": reference_episode_count,
                      "files": len(manifest["files"]) + 1}, ensure_ascii=False))


if __name__ == "__main__":
    main()
