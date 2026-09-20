#!/usr/bin/env python3
"""Generate paper-facing evidence from the sealed 2026-09-18 revision archive.

The script validates input hashes, the copied-file manifest, and three decision
seals before emitting any manuscript number. It never executes an experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tarfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARCHIVE_SHA256 = "9bef1bdc271b8953603b14da83442cc9b219b51f78212e8974fe94a3203dbccf"
REVIEW_SHA256 = "80b261ea7853704bf1ac0d5ce7dd07500cd669cefbf43763cbe0aa5598a9630b"
EXPECTED_MISSING = {
    "06_私密云账号原始资料_勿外发/TENCENT_CLOUD_API_LOGIN_AND_SA5_ASSESSMENT(2).md",
    "02_已保存结果/MeltingPot_正式30种子及机制实验/connection.json",
}
STYLE_FILES = (
    "fancyhdr.sty",
    "iclr2027_conference.bst",
    "iclr2027_conference.sty",
    "math_commands.tex",
    "natbib.sty",
)
MELTING_RAW_SHA256 = "d58b6e1e14e888251f19263ce658812d7de561447f34ecd1e0573187a11c8018"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float, digits: int = 3) -> str:
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return "--"
    return f"{number:.{digits}f}"


def interval(item: dict[str, Any], digits: int = 3) -> str:
    return f"[{fmt(item['lo'], digits)}, {fmt(item['hi'], digits)}]"


def ci95(values: list[float], digits: int = 3) -> str:
    return f"[{fmt(values[0], digits)}, {fmt(values[1], digits)}]"


def validate_manifest(evidence: Path) -> dict[str, Any]:
    rows = load(evidence / "文件清单.json")
    missing: list[str] = []
    mismatches: list[str] = []
    verified = 0
    for row in rows:
        path = evidence / row["file"]
        if not path.is_file():
            missing.append(row["file"])
        elif path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            mismatches.append(row["file"])
        else:
            verified += 1
    unexpected = sorted(set(missing) - EXPECTED_MISSING)
    if unexpected or mismatches:
        raise RuntimeError(
            f"manifest verification failed: unexpected_missing={unexpected}, mismatches={mismatches}"
        )
    return {
        "entries": len(rows),
        "verified": verified,
        "missing": sorted(missing),
        "expected_missing": sorted(EXPECTED_MISSING),
    }


def validate_seal(directory: Path) -> dict[str, Any]:
    decisions = directory / "decisions_sealed.json"
    seal = load(directory / "selection_seal.json")
    scored = load(directory / "scored_decisions.json")
    digest = sha256(decisions)
    if digest != seal["decisions_sha256"]:
        raise RuntimeError(f"decision seal mismatch: {directory}")
    if len(scored) != int(seal["n_decisions"]):
        raise RuntimeError(f"decision count mismatch: {directory}")
    roots = sorted({int(row["root"]) for row in scored})
    if roots != sorted(int(root) for root in seal["test_roots"]):
        raise RuntimeError(f"root set mismatch: {directory}")
    return {
        "directory": directory.name,
        "decisions_sha256": digest,
        "n_decisions": len(scored),
        "roots": roots,
    }


def method(
    summary: dict[str, Any], adaptation: int, name: str, cap: int | None = None
) -> dict[str, Any]:
    target_cap = int(cap if cap is not None else summary["primary_cap"])
    for row in summary["method_tables_by_cap"][str(target_cap)]:
        if int(row["adaptation"]) == adaptation and row["method"] == name:
            return row
    raise KeyError((adaptation, name, target_cap))


def primary(summary: dict[str, Any]) -> dict[str, Any]:
    return summary["hypotheses"]["H2_long_primary_gain_minus_comparator_cap192"]


def _bootstrap(values: list[float], draws: int = 10000, seed: int = 20260917) -> dict[str, float | int]:
    """Reproduce the registered root bootstrap from the sealed scored rows."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sample = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "lo": float(np.percentile(sample, 2.5)),
        "hi": float(np.percentile(sample, 97.5)),
        "n": int(len(x)),
    }


def _scored_rows(directory: Path) -> list[dict[str, Any]]:
    """Verify that the public scored rows preserve every sealed decision field."""
    decisions = load(directory / "decisions_sealed.json")
    scored = load(directory / "scored_decisions.json")
    if len(decisions) != len(scored):
        raise RuntimeError(f"sealed/scored length mismatch: {directory}")
    for index, (decision, row) in enumerate(zip(decisions, scored)):
        for key, value in decision.items():
            if row.get(key) != value:
                raise RuntimeError(f"sealed/scored field mismatch at {directory}:{index}:{key}")
    return scored


def _recomputed_summary(directory: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Recompute headline rows from scored decisions, independently of summary.json."""
    rows = _scored_rows(directory)
    long_h, short_h = max(int(r["adaptation"]) for r in rows), min(int(r["adaptation"]) for r in rows)
    primary_cap = int(summary["primary_cap"])

    def take(method_name: str, adaptation: int, cap: int | None = None) -> dict[int, float]:
        return {
            int(r["root"]): float(r["gain"])
            for r in rows
            if r["method"] == method_name
            and int(r["adaptation"]) == adaptation
            and (r["cap"] == cap or r["cap"] is None)
        }

    def contrast(a: str, b: str, adaptation: int, cap: int | None = None) -> dict[str, float | int]:
        xa, xb = take(a, adaptation, cap), take(b, adaptation, cap)
        roots = sorted(set(xa) & set(xb))
        return _bootstrap([xa[root] - xb[root] for root in roots])

    def table(method_name: str, adaptation: int, cap: int | None = None) -> dict[str, float | int]:
        selected = [
            r for r in rows
            if r["method"] == method_name
            and int(r["adaptation"]) == adaptation
            and (r["cap"] == cap or r["cap"] is None)
        ]
        return {
            "mean_isr": float(np.mean([r["isr"] for r in selected])),
            "mean_gain": float(np.mean([r["gain"] for r in selected])),
            "mean_hf_episode_cost": float(np.mean([r["hf_episode_cost"] for r in selected])),
            "n": len(selected),
        }

    long_a = take("pivot_kg", long_h, primary_cap)
    long_b = take("uniform_v2", long_h, primary_cap)
    short_a = take("pivot_kg", short_h, primary_cap)
    short_b = take("uniform_v2", short_h, primary_cap)
    roots = sorted(set(long_a) & set(long_b) & set(short_a) & set(short_b))
    interaction = _bootstrap([
        (long_a[r] - long_b[r]) - (short_a[r] - short_b[r]) for r in roots
    ])
    caps = sorted(int(cap) for cap in summary["caps"])
    return {
        "primary": contrast("pivot_kg", "uniform_v2", long_h, primary_cap),
        "interaction": interaction,
        "one_query": contrast("pivot_kg", "uniform_v2", long_h, caps[0]),
        "short": contrast("pivot_kg", "uniform_v2", short_h, primary_cap),
        "rows": {
            "pivot_kg": table("pivot_kg", long_h, primary_cap),
            "uniform_v2": table("uniform_v2", long_h, primary_cap),
            "all_hf_reference": table("all_hf_reference", long_h, primary_cap),
            "no_hf_v2": table("no_hf_v2", long_h, primary_cap),
            "proxy_only": table("proxy_only", long_h, primary_cap),
            "one_query_pivot_kg": table("pivot_kg", long_h, caps[0]),
            "short_pivot_kg": table("pivot_kg", short_h, primary_cap),
        },
    }


def _close(a: dict[str, Any], b: dict[str, Any], tolerance: float = 1e-9) -> bool:
    return all(abs(float(a[key]) - float(b[key])) <= tolerance for key in ("mean", "lo", "hi")) and int(a["n"]) == int(b["n"])


def _raw_confirmation_worlds(responsive: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, str]]:
    """Load candidate-level proxy/deployment arrays for the 90 held-out roots."""

    server = responsive / "server_results"
    worlds: dict[tuple[str, int], dict[str, Any]] = {}
    roots = {
        "Leduc": server / "openspiel_v2_20260918/leduc_v2b_confirm",
        "Kuhn": server / "openspiel_v2_20260918/kuhn_v2b_confirm",
    }
    for cohort, directory in roots.items():
        for path in sorted(directory.glob("seed_*/summary.json")):
            payload = load(path)
            worlds[(cohort, int(payload["seed"]))] = payload

    raw_archive = server / "melting_v4_hetero_20260918.tar.gz"
    if sha256(raw_archive) != MELTING_RAW_SHA256:
        raise RuntimeError("Melting Pot confirmation archive hash mismatch")
    member_hashes: dict[str, str] = {}
    with tarfile.open(raw_archive, "r:gz") as archive:
        for member in archive.getmembers():
            if (
                "melting_v4_confirm_20260918/seed_" not in member.name
                or not member.name.endswith("/summary.json")
            ):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"missing tar member payload: {member.name}")
            data = handle.read()
            payload = json.loads(data)
            worlds[("Melting Pot", int(payload["seed"]))] = payload
            member_hashes[member.name] = hashlib.sha256(data).hexdigest()

    counts = {cohort: sum(name == cohort for name, _ in worlds) for cohort in ("Leduc", "Kuhn", "Melting Pot")}
    if counts != {"Leduc": 30, "Kuhn": 30, "Melting Pot": 30}:
        raise RuntimeError(f"unexpected decision-relevance root counts: {counts}")
    return worlds, member_hashes


def _decision_relevance(
    responsive: Path,
    directories: dict[str, Path],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Derive a post-hoc decision-relevance bridge from sealed confirmation roots.

    The correction-to-margin ratio is exact for the observed top-1 ordering:
    for the most proxy-favoured truly optimal candidate b and each competitor
    j, it is ((D_b-P_b)-(D_j-P_j))/(D_b-D_j).  Its maximum reaches one exactly
    when a proxy competitor overtakes every deployment-optimal candidate.
    """

    raw_worlds, member_hashes = _raw_confirmation_worlds(responsive)
    cohort_specs = {
        "Leduc": ("leduc", directories["leduc"]),
        "Kuhn": ("kuhn", directories["kuhn"]),
        "Melting Pot": ("melting_v4", directories["melting_v4"]),
    }
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    noise_scales: dict[str, Any] = {}
    for cohort, (summary_name, directory) in cohort_specs.items():
        summary = summaries[summary_name]
        scored = _scored_rows(directory)
        posterior_spec = load(directory / "posterior_v2_spec.json")
        long_h = max(int(row["adaptation"]) for row in scored)
        primary_cap = int(summary["primary_cap"])
        noise_scale = np.asarray(posterior_spec[str(long_h)]["R"], dtype=float)
        noise_scales[cohort] = {"adaptation": long_h, "R": noise_scale.tolist()}
        selected = {
            (int(row["root"]), str(row["method"])): row
            for row in scored
            if int(row["adaptation"]) == long_h
            and row["cap"] == primary_cap
            and row["method"] in {"pivot_kg", "uniform_v2"}
        }
        roots = sorted({root for root, _ in selected})
        if len(roots) != 30:
            raise RuntimeError(f"{cohort}: expected 30 primary roots, found {len(roots)}")
        for root in roots:
            world = raw_worlds[(cohort, root)]
            candidate_ids = sorted((int(key) for key in world["proxy_deltas"]), key=int)
            proxy = np.asarray([float(world["proxy_deltas"][str(i)]) for i in candidate_ids] + [0.0])
            deployment = np.asarray(
                [float(world["audit_gains"][str(long_h)][str(i)]) for i in candidate_ids] + [0.0]
            )
            labels = [str(i) for i in candidate_ids] + ["incumbent"]
            candidate_proxy = proxy[:-1]
            candidate_deployment = deployment[:-1]
            proxy_order = np.argsort(-candidate_proxy, kind="stable")
            deployment_order = np.argsort(-candidate_deployment, kind="stable")
            proxy_top2 = {candidate_ids[int(index)] for index in proxy_order[:2]}
            deployment_top2 = {candidate_ids[int(index)] for index in deployment_order[:2]}
            decision_critical = {
                candidate_ids[int(proxy_order[0])],
                candidate_ids[int(deployment_order[0])],
            }
            proxy_margin = float(candidate_proxy[proxy_order[0]] - candidate_proxy[proxy_order[1]])
            deployment_candidate_margin = float(
                candidate_deployment[deployment_order[0]] - candidate_deployment[deployment_order[1]]
            )
            best = float(deployment.max())
            true_best = np.isclose(deployment, best, rtol=0.0, atol=1e-12)
            lower = deployment[~true_best]
            margin = float(best - lower.max()) if lower.size else 0.0
            if margin <= 0.0:
                raise RuntimeError(f"{cohort}/{root}: non-positive deployment margin")
            best_index = max(np.flatnonzero(true_best), key=lambda idx: float(proxy[idx]))
            correction = deployment - proxy
            ratios = [
                float((correction[best_index] - correction[j]) / (deployment[best_index] - deployment[j]))
                for j in np.flatnonzero(~true_best)
            ]
            ratio = max(ratios)
            proxy_best = np.isclose(proxy, proxy.max(), rtol=0.0, atol=1e-12)
            top1_flip = not bool(np.any(true_best & proxy_best))
            if top1_flip != (ratio >= 1.0 - 1e-12):
                raise RuntimeError(f"{cohort}/{root}: flip/ratio identity failed")
            pivot = selected[(root, "pivot_kg")]
            uniform = selected[(root, "uniform_v2")]
            deployment_range = float(deployment.max() - deployment.min())
            proxy_range = float(candidate_proxy.max() - candidate_proxy.min())
            short_h = min(int(row["adaptation"]) for row in scored)
            short_deployment = np.asarray(
                [float(world["audit_gains"][str(short_h)][str(i)]) for i in candidate_ids] + [0.0]
            )
            root_scale = max(
                proxy_range,
                deployment_range,
                float(short_deployment.max() - short_deployment.min()),
                1e-12,
            )
            mechanism = world.get("mechanism_audit", [])
            gaps: dict[tuple[int, str, str], float] = {
                (int(item["adaptation"]), str(item["candidate"]), str(item["block"])): float(item["gap"])
                for item in mechanism
            }
            def squared_gap(adaptation: int) -> float:
                return float(
                    np.mean(
                        [
                            gaps[(adaptation, str(candidate), block_a)]
                            * gaps[(adaptation, str(candidate), block_b)]
                            for candidate in candidate_ids
                            for block_a, block_b in (("A", "B"),)
                        ]
                    )
                )
            native_response_gap = squared_gap(long_h) - squared_gap(short_h)
            normalized_response_gap = abs(native_response_gap) / (root_scale**2)
            allocation_gain = float(uniform["isr"] - pivot["isr"])
            normalized_gain = allocation_gain / deployment_range if deployment_range > 0 else 0.0
            queried = [int(candidate) for candidate in (pivot.get("queried") or [])]
            query_noise = float(np.mean([np.sqrt(noise_scale[candidate]) for candidate in queried])) if queried else float("nan")
            query_noise_to_margin = query_noise / max(proxy_margin, 1e-12) if queried else float("nan")
            estimates = np.asarray(pivot["estimates"], dtype=float)
            estimate_order = np.argsort(-estimates, kind="stable")
            post_query_margin = float(estimates[estimate_order[0]] - estimates[estimate_order[1]])
            post_query_separation = post_query_margin / root_scale
            first_query = queried[0] if queried else None
            second_query = queried[1] if len(queried) > 1 else None
            query_hits_critical = bool(set(queried) & decision_critical)
            query_hits_deployment_top1 = candidate_ids[int(deployment_order[0])] in set(queried)
            query_hits_deployment_top2 = bool(set(queried) & deployment_top2)
            first_query_hits_critical = bool(first_query is not None and first_query in decision_critical)
            second_query_hits_critical = bool(second_query is not None and second_query in decision_critical)
            rows.append(
                {
                    "cohort": cohort,
                    "root": root,
                    "responder_type": world["responder_type"],
                    "deployment_margin": margin,
                    "max_abs_correction": float(np.max(np.abs(correction))),
                    "pairwise_correction_range": float(correction.max() - correction.min()),
                    "correction_to_margin_ratio": ratio,
                    "top1_ranking_flip": top1_flip,
                    "proxy_top2_margin": proxy_margin,
                    "deployment_candidate_margin": deployment_candidate_margin,
                    "native_response_gap_squared": native_response_gap,
                    "normalized_response_gap": normalized_response_gap,
                    "root_response_scale": root_scale,
                    "queried": queried,
                    "first_query": first_query,
                    "second_query": second_query,
                    "query_hits_decision_pair": query_hits_critical,
                    "first_query_hits_decision_pair": first_query_hits_critical,
                    "second_query_hits_decision_pair": second_query_hits_critical,
                    "query_hits_deployment_top1": query_hits_deployment_top1,
                    "query_hits_deployment_top2": query_hits_deployment_top2,
                    "query_noise_sd": query_noise,
                    "query_noise_to_proxy_margin": query_noise_to_margin,
                    "post_query_margin": post_query_margin,
                    "post_query_separation": post_query_separation,
                    "pivot_minus_uniform_isr_gain": allocation_gain,
                    "deployment_range": deployment_range,
                    "normalized_allocation_gain": normalized_gain,
                    "pivot_isr": float(pivot["isr"]),
                    "uniform_isr": float(uniform["isr"]),
                }
            )
            inputs.append(
                {
                    "cohort": cohort,
                    "root": root,
                    "adaptation": long_h,
                    "primary_cap": primary_cap,
                    "candidate_ids": labels,
                    "proxy_deltas": proxy.tolist(),
                    "deployment_deltas": deployment.tolist(),
                }
            )

    def summarize(values: list[float], *, seed: int) -> dict[str, float | int]:
        return _bootstrap(values, seed=seed)

    def summarize_optional(values: list[float], *, seed: int) -> dict[str, float | int] | None:
        return summarize(values, seed=seed) if values else None

    cohort_summary: dict[str, Any] = {}
    for index, cohort in enumerate(("Leduc", "Kuhn", "Melting Pot")):
        cohort_rows = [row for row in rows if row["cohort"] == cohort]
        flipped = [row for row in cohort_rows if row["top1_ranking_flip"]]
        stable = [row for row in cohort_rows if not row["top1_ranking_flip"]]
        cohort_summary[cohort] = {
            "n_roots": len(cohort_rows),
            "n_top1_flips": len(flipped),
            "top1_flip_rate": summarize(
                [float(row["top1_ranking_flip"]) for row in cohort_rows], seed=20260920 + index
            ),
            "deployment_margin": summarize(
                [float(row["deployment_margin"]) for row in cohort_rows], seed=20260930 + index
            ),
            "normalized_gain_flipped": summarize(
                [float(row["normalized_allocation_gain"]) for row in flipped], seed=20260940 + index
            ),
            "normalized_gain_stable": summarize(
                [float(row["normalized_allocation_gain"]) for row in stable], seed=20260950 + index
            ),
            "normalized_response_gap": summarize(
                [float(row["normalized_response_gap"]) for row in cohort_rows], seed=20260980 + index
            ),
            "proxy_top2_margin": summarize(
                [float(row["proxy_top2_margin"]) for row in cohort_rows], seed=20260990 + index
            ),
            "query_hits_decision_pair": summarize(
                [float(row["query_hits_decision_pair"]) for row in cohort_rows], seed=20261000 + index
            ),
            "first_query_hits_decision_pair": summarize(
                [float(row["first_query_hits_decision_pair"]) for row in cohort_rows], seed=20261010 + index
            ),
            "second_query_hits_decision_pair": summarize(
                [float(row["second_query_hits_decision_pair"]) for row in cohort_rows], seed=20261011 + index
            ),
            "query_hits_deployment_top1": summarize(
                [float(row["query_hits_deployment_top1"]) for row in cohort_rows], seed=20261020 + index
            ),
            "query_hits_deployment_top2": summarize(
                [float(row["query_hits_deployment_top2"]) for row in cohort_rows], seed=20261030 + index
            ),
            "query_noise_to_proxy_margin": summarize(
                [float(row["query_noise_to_proxy_margin"]) for row in cohort_rows], seed=20261040 + index
            ),
            "post_query_separation": summarize(
                [float(row["post_query_separation"]) for row in cohort_rows], seed=20261050 + index
            ),
            "normalized_gain_query_hit": summarize_optional(
                [float(row["normalized_allocation_gain"]) for row in cohort_rows if row["query_hits_decision_pair"]],
                seed=20261060 + index,
            ),
            "normalized_gain_query_miss": summarize_optional(
                [float(row["normalized_allocation_gain"]) for row in cohort_rows if not row["query_hits_decision_pair"]],
                seed=20261070 + index,
            ),
        }

    pooled_flipped = [row for row in rows if row["top1_ranking_flip"]]
    pooled_stable = [row for row in rows if not row["top1_ranking_flip"]]
    query_hits = [row for row in rows if row["query_hits_decision_pair"]]
    query_misses = [row for row in rows if not row["query_hits_decision_pair"]]
    return {
        "schema_version": "pivot-decision-relevance-2",
        "analysis_status": "post_hoc_explanatory_with_observable_resolvability",
        "definition": {
            "deployment_margin": "top deployment delta minus the best strictly lower deployment delta",
            "correction": "deployment delta minus proxy delta",
            "correction_to_margin_ratio": "largest pairwise differential correction divided by the corresponding deployment margin",
            "decision_fragile": "the ratio is at least one, equivalently the proxy top-1 set excludes the deployment top-1 set",
            "allocation_gain": "Uniform ISR minus PIVOT-KG ISR at each cohort's primary long-response budget",
            "normalized_allocation_gain": "allocation gain divided by the root's deployment-delta range",
            "native_response_gap_squared": "long-minus-short difference in the sealed noise-corrected squared gap, recomputed from mechanism_audit",
            "normalized_response_gap": "absolute native response gap divided by the square of the largest root-level candidate range across proxy and long/short deployment deltas",
            "decision_critical_pair": "the union of proxy and deployment top-1 candidates at the long response",
            "query_hits_decision_pair": "the sealed PIVOT query set intersects that decision-critical pair",
            "query_noise_to_proxy_margin": "mean square-root observation noise R for queried candidates divided by the proxy top-two margin",
            "post_query_separation": "final sealed estimate top-two separation divided by the root response scale; this is an observable separation diagnostic, not a posterior probability",
        },
        "source": {
            "experiment_archive_sha256": ARCHIVE_SHA256,
            "melting_raw_archive_sha256": MELTING_RAW_SHA256,
            "melting_member_sha256": member_hashes,
        },
        "noise_scales": noise_scales,
        "n_roots": len(rows),
        "n_top1_flips": len(pooled_flipped),
        "cohorts": cohort_summary,
        "pooled_descriptive": {
            "normalized_gain_flipped": summarize(
                [float(row["normalized_allocation_gain"]) for row in pooled_flipped], seed=20260960
            ),
            "normalized_gain_stable": summarize(
                [float(row["normalized_allocation_gain"]) for row in pooled_stable], seed=20260961
            ),
            "normalized_response_gap": summarize(
                [float(row["normalized_response_gap"]) for row in rows], seed=20260962
            ),
            "query_hits_decision_pair": summarize(
                [float(row["query_hits_decision_pair"]) for row in rows], seed=20260963
            ),
            "query_hits_deployment_top1": summarize(
                [float(row["query_hits_deployment_top1"]) for row in rows], seed=20260964
            ),
            "query_hits_deployment_top2": summarize(
                [float(row["query_hits_deployment_top2"]) for row in rows], seed=20260965
            ),
            "query_noise_to_proxy_margin": summarize(
                [float(row["query_noise_to_proxy_margin"]) for row in rows], seed=20260966
            ),
            "post_query_separation": summarize(
                [float(row["post_query_separation"]) for row in rows], seed=20260967
            ),
            "normalized_gain_query_hit": summarize_optional(
                [float(row["normalized_allocation_gain"]) for row in query_hits], seed=20260968
            ),
            "normalized_gain_query_miss": summarize_optional(
                [float(row["normalized_allocation_gain"]) for row in query_misses], seed=20260969
            ),
        },
        "inputs": inputs,
        "rows": rows,
    }


def public_audit(report: dict[str, Any]) -> dict[str, Any]:
    """Return the reviewer-safe subset of the local revision audit.

    The local audit retains the exact names of deliberately omitted connection
    and account files so the source handoff can be reconciled.  Those names are
    operational metadata, not scientific evidence, and must not enter the
    anonymous release.
    """

    manifest = report.get("manifest", {})
    omissions = manifest.get("expected_missing", manifest.get("missing", []))
    seal_aliases = {
        "melting_v4": "melting_v4",
        "leduc_v2b": "leduc",
        "kuhn_v2b": "kuhn",
    }
    source_seals = report.get("seals", {})
    public_seals = {
        public_name: source_seals[source_name]
        for public_name, source_name in seal_aliases.items()
        if source_name in source_seals
    }
    return {
        "schema_version": "pivot-revision-evidence-public-1",
        "archive_sha256": report.get("archive_sha256"),
        "review_sha256": report.get("review_sha256"),
        "manifest": {
            "entries": int(manifest.get("entries", 0)),
            "verified": int(manifest.get("verified", 0)),
            "private_omissions": len(omissions),
        },
        "seals": public_seals,
        "recomputed_from_scored_rows": report.get("recomputed_from_scored_rows", {}),
        "recomputation_checks": report.get("recomputation_checks", {}),
        "summaries": report.get("summaries", {}),
        "decision_relevance": report.get("decision_relevance", {}),
        "generated": report.get("generated", []),
        "status_boundary": report.get("status_boundary", {}),
    }


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def build(root: Path, evidence: Path) -> dict[str, Any]:
    root = root.resolve()
    evidence = evidence.resolve()
    archive = root / "PIVOT_ICLR_实验全包_20260918.zip"
    review = root / "ICRL论文修改意见.pdf"
    if sha256(archive) != ARCHIVE_SHA256:
        raise RuntimeError("experiment ZIP does not match the supplied archive")
    if sha256(review) != REVIEW_SHA256:
        raise RuntimeError("review PDF does not match the supplied document")

    manifest = validate_manifest(evidence)
    responsive = evidence / "07_下一步_PIVOTv2_异质响应"
    dirs = {
        "melting_v4": responsive / "server_results/melting_v4_confirm_analysis",
        "leduc": responsive / "server_results/openspiel_v2_20260918/reanalysis_v12/leduc_v2b_confirm",
        "kuhn": responsive / "server_results/openspiel_v2_20260918/reanalysis_v12/kuhn_v2b_confirm",
    }
    seals = {name: validate_seal(path) for name, path in dirs.items()}
    summaries = {name: load(path / "summary.json") for name, path in dirs.items()}
    decision_relevance = _decision_relevance(responsive, dirs, summaries)
    v4 = summaries["melting_v4"]
    leduc = summaries["leduc"]
    kuhn = summaries["kuhn"]
    base = evidence / "02_已保存结果/MeltingPot_正式30种子及机制实验"
    homogeneous = load(base / "method_analysis/summary.json")
    mpe2 = load(base / "mpe2_closure/fresh_two_task_summary.json")

    if not leduc["hypotheses"]["prespecified_success"]:
        raise RuntimeError("delivered Leduc primary success flag is false")
    if v4["hypotheses"]["prespecified_success"]:
        raise RuntimeError("delivered Melting Pot v4 primary success flag unexpectedly true")

    leduc_h2 = primary(leduc)
    kuhn_h2 = primary(kuhn)
    v4_h2 = primary(v4)
    v4_h3 = v4["hypotheses"]["H3_interaction_long_minus_short"]
    v4_one = v4["hypotheses"]["secondary"][
        "long_smallest_cap_single_query_kg_minus_uniform"
    ]
    v4_short = v4["hypotheses"]["secondary"]["short_kg_minus_uniform"]

    # Recompute every headline contrast from the sealed scored rows.  The
    # summary JSON is treated as a cross-check, never as the sole source of a
    # manuscript number.
    recomputed = {
        "melting_v4": _recomputed_summary(dirs["melting_v4"], v4),
        "leduc": _recomputed_summary(dirs["leduc"], leduc),
        "kuhn": _recomputed_summary(dirs["kuhn"], kuhn),
    }
    checks = {
        "melting_v4_primary": _close(recomputed["melting_v4"]["primary"], v4_h2),
        "melting_v4_interaction": _close(recomputed["melting_v4"]["interaction"], v4_h3),
        "melting_v4_one_query": _close(recomputed["melting_v4"]["one_query"], v4_one),
        "melting_v4_short": _close(recomputed["melting_v4"]["short"], v4_short),
        "leduc_primary": _close(recomputed["leduc"]["primary"], leduc_h2),
        "kuhn_primary": _close(recomputed["kuhn"]["primary"], kuhn_h2),
    }
    if not all(checks.values()):
        raise RuntimeError(f"sealed-row recomputation mismatch: {checks}")

    rows = {
        "leduc_pivot": method(leduc, 8, "pivot_kg"),
        "leduc_uniform": method(leduc, 8, "uniform_v2"),
        "leduc_all": method(leduc, 8, "all_hf_reference"),
        "leduc_nohf": method(leduc, 8, "no_hf_v2"),
        "kuhn_pivot": method(kuhn, 8, "pivot_kg"),
        "kuhn_uniform": method(kuhn, 8, "uniform_v2"),
        "v4_pivot": method(v4, 32, "pivot_kg"),
        "v4_uniform": method(v4, 32, "uniform_v2"),
        "v4_all": method(v4, 32, "all_hf_reference"),
        "v4_nohf": method(v4, 32, "no_hf_v2"),
        "v4_one_pivot": method(v4, 32, "pivot_kg", 96),
        "v4_short_pivot": method(v4, 4, "pivot_kg"),
    }

    paper = root / "paper"
    figures = paper / "figures/revision"
    tables = paper / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    audit_dir = root / "artifacts/revision/20260920-manuscript-update"
    audit_dir.mkdir(parents=True, exist_ok=True)

    hgap = homogeneous["confirmation_mechanism"]["gap_squared_long_minus_short"]
    heffect = homogeneous["primary_effects"]["long_regret_reduction"]
    push_response = mpe2["tables"]["push"]["mechanism"]["response_effect"]
    adv_response = mpe2["tables"]["adversary"]["mechanism"]["response_effect"]
    macros = [
        "% Generated by scripts/build_revision_evidence.py; do not edit.",
        macro("RevisionLeducHtwo", fmt(leduc_h2["mean"])),
        macro("RevisionLeducHtwoCI", interval(leduc_h2)),
        macro("RevisionLeducHthree", fmt(leduc["hypotheses"]["H3_interaction_long_minus_short"]["mean"])),
        macro("RevisionLeducPivotISR", fmt(rows["leduc_pivot"]["mean_isr"])),
        macro("RevisionLeducUniformISR", fmt(rows["leduc_uniform"]["mean_isr"])),
        macro("RevisionLeducAllHFISR", fmt(rows["leduc_all"]["mean_isr"])),
        macro("RevisionLeducNoHFISR", fmt(rows["leduc_nohf"]["mean_isr"])),
        macro("RevisionLeducCostShare", fmt(rows["leduc_pivot"]["mean_hf_episode_cost"] / rows["leduc_all"]["mean_hf_episode_cost"], 2)),
        macro("RevisionVFourHtwo", fmt(v4_h2["mean"])),
        macro("RevisionVFourHtwoCI", interval(v4_h2)),
        macro("RevisionVFourHthree", fmt(v4_h3["mean"])),
        macro("RevisionVFourHthreeCI", interval(v4_h3)),
        macro("RevisionVFourOneQuery", fmt(v4_one["mean"])),
        macro("RevisionVFourOneQueryCI", interval(v4_one)),
        macro("RevisionVFourShort", fmt(v4_short["mean"])),
        macro("RevisionVFourShortCI", interval(v4_short)),
        macro("RevisionVFourPivotISR", fmt(rows["v4_pivot"]["mean_isr"])),
        macro("RevisionVFourUniformISR", fmt(rows["v4_uniform"]["mean_isr"])),
        macro("RevisionVFourAllHFISR", fmt(rows["v4_all"]["mean_isr"])),
        macro("RevisionVFourNoHFISR", fmt(rows["v4_nohf"]["mean_isr"])),
        macro("RevisionVFourCostShare", fmt(rows["v4_pivot"]["mean_hf_episode_cost"] / rows["v4_all"]["mean_hf_episode_cost"], 2)),
        macro("RevisionKuhnHtwo", fmt(kuhn_h2["mean"])),
        macro("RevisionKuhnHtwoCI", interval(kuhn_h2)),
        macro("RevisionKuhnPivotISR", fmt(rows["kuhn_pivot"]["mean_isr"])),
        macro("RevisionKuhnUniformISR", fmt(rows["kuhn_uniform"]["mean_isr"])),
        macro("RevisionLeducResponseGap", fmt(leduc["mechanism"]["squared_gap_long_minus_short"]["mean"])),
        macro("RevisionLeducResponseGapCI", interval(leduc["mechanism"]["squared_gap_long_minus_short"])),
        macro(
            "RevisionLeducNormalizedResponseGap",
            fmt(decision_relevance["cohorts"]["Leduc"]["normalized_response_gap"]["mean"], 2),
        ),
        macro(
            "RevisionLeducNormalizedResponseGapCI",
            interval(decision_relevance["cohorts"]["Leduc"]["normalized_response_gap"], 2),
        ),
        macro("RevisionLeducIDE", fmt(leduc["mechanism"]["8"]["IDE_mean_abs_delta_error"]["mean"])),
        macro("RevisionLeducIRR", fmt(leduc["mechanism"]["8"]["IRR_reversal_rate_given_proxy_positive"]["mean"])),
        macro("RevisionLeducISC", fmt(leduc["mechanism"]["8"]["ISC_sign_consistency"]["mean"])),
        macro("RevisionKuhnResponseGap", fmt(kuhn["mechanism"]["squared_gap_long_minus_short"]["mean"])),
        macro("RevisionKuhnResponseGapCI", interval(kuhn["mechanism"]["squared_gap_long_minus_short"])),
        macro(
            "RevisionKuhnNormalizedResponseGap",
            fmt(decision_relevance["cohorts"]["Kuhn"]["normalized_response_gap"]["mean"], 2),
        ),
        macro(
            "RevisionKuhnNormalizedResponseGapCI",
            interval(decision_relevance["cohorts"]["Kuhn"]["normalized_response_gap"], 2),
        ),
        macro("RevisionKuhnIDE", fmt(kuhn["mechanism"]["8"]["IDE_mean_abs_delta_error"]["mean"])),
        macro("RevisionKuhnIRR", fmt(kuhn["mechanism"]["8"]["IRR_reversal_rate_given_proxy_positive"]["mean"])),
        macro("RevisionKuhnISC", fmt(kuhn["mechanism"]["8"]["ISC_sign_consistency"]["mean"])),
        macro("RevisionVFourResponseGap", fmt(v4["mechanism"]["squared_gap_long_minus_short"]["mean"])),
        macro("RevisionVFourResponseGapCI", interval(v4["mechanism"]["squared_gap_long_minus_short"])),
        macro(
            "RevisionVFourNormalizedResponseGap",
            fmt(decision_relevance["cohorts"]["Melting Pot"]["normalized_response_gap"]["mean"], 2),
        ),
        macro(
            "RevisionVFourNormalizedResponseGapCI",
            interval(decision_relevance["cohorts"]["Melting Pot"]["normalized_response_gap"], 2),
        ),
        macro("RevisionVFourIDE", fmt(v4["mechanism"]["32"]["IDE_mean_abs_delta_error"]["mean"])),
        macro("RevisionVFourIRR", fmt(v4["mechanism"]["32"]["IRR_reversal_rate_given_proxy_positive"]["mean"])),
        macro("RevisionVFourISC", fmt(v4["mechanism"]["32"]["ISC_sign_consistency"]["mean"])),
        macro("RevisionHomogeneousGap", fmt(hgap["mean"], 1)),
        macro("RevisionHomogeneousGapCI", ci95(hgap["ci95"], 1)),
        macro("RevisionHomogeneousEffect", fmt(heffect["mean"])),
        macro("RevisionHomogeneousEffectCI", ci95(heffect["ci95"])),
        macro("RevisionMPEPushResponse", fmt(push_response["mean"])),
        macro("RevisionMPEPushResponseCI", ci95(push_response["ci95"])),
        macro("RevisionMPEAdversaryResponse", fmt(adv_response["mean"])),
        macro("RevisionMPEAdversaryResponseCI", ci95(adv_response["ci95"])),
        macro("RevisionDecisionRoots", str(decision_relevance["n_roots"])),
        macro("RevisionDecisionFlips", str(decision_relevance["n_top1_flips"])),
        macro(
            "RevisionDecisionStable",
            str(decision_relevance["n_roots"] - decision_relevance["n_top1_flips"]),
        ),
        macro(
            "RevisionFlippedNormalizedGain",
            fmt(decision_relevance["pooled_descriptive"]["normalized_gain_flipped"]["mean"]),
        ),
        macro(
            "RevisionFlippedNormalizedGainCI",
            interval(decision_relevance["pooled_descriptive"]["normalized_gain_flipped"]),
        ),
        macro(
            "RevisionStableNormalizedGain",
            fmt(decision_relevance["pooled_descriptive"]["normalized_gain_stable"]["mean"]),
        ),
        macro(
            "RevisionStableNormalizedGainCI",
            interval(decision_relevance["pooled_descriptive"]["normalized_gain_stable"]),
        ),
        macro(
            "RevisionCriticalQueryRate",
            fmt(decision_relevance["pooled_descriptive"]["query_hits_decision_pair"]["mean"], 2),
        ),
        macro(
            "RevisionCriticalQueryRateCI",
            interval(decision_relevance["pooled_descriptive"]["query_hits_decision_pair"], 2),
        ),
        macro(
            "RevisionCriticalQueryHitGain",
            fmt(decision_relevance["pooled_descriptive"]["normalized_gain_query_hit"]["mean"], 3),
        ),
        macro(
            "RevisionCriticalQueryHitGainCI",
            interval(decision_relevance["pooled_descriptive"]["normalized_gain_query_hit"]),
        ),
        macro(
            "RevisionCriticalQueryMissGain",
            fmt(decision_relevance["pooled_descriptive"]["normalized_gain_query_miss"]["mean"], 3),
        ),
        macro(
            "RevisionCriticalQueryMissGainCI",
            interval(decision_relevance["pooled_descriptive"]["normalized_gain_query_miss"]),
        ),
        macro(
            "RevisionCriticalNoiseMargin",
            fmt(decision_relevance["pooled_descriptive"]["query_noise_to_proxy_margin"]["mean"], 2),
        ),
        macro(
            "RevisionCriticalNoiseMarginCI",
            interval(decision_relevance["pooled_descriptive"]["query_noise_to_proxy_margin"], 2),
        ),
        macro(
            "RevisionPostQuerySeparation",
            fmt(decision_relevance["pooled_descriptive"]["post_query_separation"]["mean"], 2),
        ),
        macro(
            "RevisionPostQuerySeparationCI",
            interval(decision_relevance["pooled_descriptive"]["post_query_separation"], 2),
        ),
        macro("RevisionLeducFlips", str(decision_relevance["cohorts"]["Leduc"]["n_top1_flips"])),
        macro("RevisionKuhnFlips", str(decision_relevance["cohorts"]["Kuhn"]["n_top1_flips"])),
        macro(
            "RevisionMeltingFlips",
            str(decision_relevance["cohorts"]["Melting Pot"]["n_top1_flips"]),
        ),
        macro(
            "RevisionLeducCriticalQueryRate",
            fmt(decision_relevance["cohorts"]["Leduc"]["query_hits_decision_pair"]["mean"], 2),
        ),
        macro(
            "RevisionKuhnCriticalQueryRate",
            fmt(decision_relevance["cohorts"]["Kuhn"]["query_hits_decision_pair"]["mean"], 2),
        ),
        macro(
            "RevisionMeltingCriticalQueryRate",
            fmt(decision_relevance["cohorts"]["Melting Pot"]["query_hits_decision_pair"]["mean"], 2),
        ),
    ]
    (paper / "revision_results.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    (audit_dir / "decision_relevance.json").write_text(
        json.dumps(decision_relevance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (audit_dir / "decision_relevance_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = list(decision_relevance["rows"][0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(decision_relevance["rows"])

    main_table = "\n".join(
        [
            r"\begin{tabular}{@{}lrrrrl@{}}",
            r"\toprule",
            r"World / condition & $n$ & PIVOT--Uniform & 95\% CI & PIVOT ISR & Status \\",
            r"\midrule",
            f"Leduc, long / 2 queries & 30 & {fmt(leduc_h2['mean'])} & {interval(leduc_h2)} & {fmt(rows['leduc_pivot']['mean_isr'])} & primary positive " + r"\\",
            f"Kuhn, long / 2 queries & 30 & {fmt(kuhn_h2['mean'])} & {interval(kuhn_h2)} & {fmt(rows['kuhn_pivot']['mean_isr'])} & near-null " + r"\\",
            f"Melting Pot, long / 2 queries & 30 & {fmt(v4_h2['mean'])} & {interval(v4_h2)} & {fmt(rows['v4_pivot']['mean_isr'])} & primary null " + r"\\",
            f"Melting Pot, long / 1 query & 30 & {fmt(v4_one['mean'])} & {interval(v4_one)} & {fmt(rows['v4_one_pivot']['mean_isr'])} & secondary positive " + r"\\",
            f"Melting Pot, short / 5 queries & 30 & {fmt(v4_short['mean'])} & {interval(v4_short)} & {fmt(rows['v4_short_pivot']['mean_isr'])} & negative " + r"\\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (tables / "revision_main_results.tex").write_text(main_table + "\n", encoding="utf-8")

    method_table = "\n".join(
        [
            r"\begin{tabular}{@{}llrr@{}}",
            r"\toprule",
            r"Cohort & Method & ISR & HF cost \\",
            r"\midrule",
            f"Leduc long & PIVOT-KG & {fmt(rows['leduc_pivot']['mean_isr'])} & {fmt(rows['leduc_pivot']['mean_hf_episode_cost'], 0)} " + r"\\",
            f"Leduc long & Uniform & {fmt(rows['leduc_uniform']['mean_isr'])} & {fmt(rows['leduc_uniform']['mean_hf_episode_cost'], 0)} " + r"\\",
            f"Leduc long & all-HF reference & {fmt(rows['leduc_all']['mean_isr'])} & {fmt(rows['leduc_all']['mean_hf_episode_cost'], 0)} " + r"\\",
            f"Melting Pot long & PIVOT-KG & {fmt(rows['v4_pivot']['mean_isr'])} & {fmt(rows['v4_pivot']['mean_hf_episode_cost'], 0)} " + r"\\",
            f"Melting Pot long & Uniform & {fmt(rows['v4_uniform']['mean_isr'])} & {fmt(rows['v4_uniform']['mean_hf_episode_cost'], 0)} " + r"\\",
            f"Melting Pot long & all-HF reference & {fmt(rows['v4_all']['mean_isr'])} & {fmt(rows['v4_all']['mean_hf_episode_cost'], 0)} " + r"\\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (tables / "revision_method_results.tex").write_text(method_table + "\n", encoding="utf-8")

    cohorts = ("Leduc", "Kuhn", "Melting Pot")
    palette = {"Leduc": "#0B5D66", "Kuhn": "#5B6BA8", "Melting Pot": "#B65C3A"}
    fig = plt.figure(figsize=(7.0, 5.35))
    grid = fig.add_gridspec(2, 6, height_ratios=[0.85, 1.75], hspace=0.44, wspace=0.48)

    ax_gap = fig.add_subplot(grid[0, 0:2])
    ax_gap.axis("off")
    ax_gap.set_title("A  Response magnitude", loc="left", fontsize=9, fontweight="bold")
    ax_gap.text(0.0, 0.82, "within-root normalized gap [95% CI]", fontsize=6.5, color="#555555")
    for y_value, cohort in zip((0.61, 0.36, 0.11), cohorts):
        item = decision_relevance["cohorts"][cohort]["normalized_response_gap"]
        ax_gap.text(0.0, y_value, cohort, color=palette[cohort], fontsize=7.5, fontweight="bold")
        ax_gap.text(
            0.98,
            y_value,
            f"{item['mean']:.2f} [{item['lo']:.2f}, {item['hi']:.2f}]",
            ha="right",
            fontsize=7.2,
        )

    y = np.arange(len(cohorts))
    ax_flip = fig.add_subplot(grid[0, 2:4])
    ax_flip.set_title("B  Top-1 ranking flips", loc="left", fontsize=9, fontweight="bold")
    rates = [decision_relevance["cohorts"][cohort]["top1_flip_rate"] for cohort in cohorts]
    means = np.asarray([item["mean"] for item in rates])
    lows = np.asarray([item["lo"] for item in rates])
    highs = np.asarray([item["hi"] for item in rates])
    ax_flip.errorbar(
        means,
        y,
        xerr=[means - lows, highs - means],
        fmt="none",
        ecolor="#333333",
        capsize=2,
        linewidth=0.8,
    )
    ax_flip.scatter(means, y, c=[palette[name] for name in cohorts], s=24, zorder=3)
    ax_flip.set_yticks(y, cohorts, fontsize=7)
    ax_flip.set_xlim(0, 1)
    ax_flip.set_xlabel("fraction of roots", fontsize=7)
    ax_flip.tick_params(axis="x", labelsize=6.5)
    ax_flip.grid(axis="x", alpha=0.2)
    ax_flip.invert_yaxis()

    ax_value = fig.add_subplot(grid[0, 4:6])
    ax_value.set_title("C  Allocation value", loc="left", fontsize=9, fontweight="bold")
    all_gain = []
    for index, cohort in enumerate(cohorts):
        values = [
            float(row["normalized_allocation_gain"])
            for row in decision_relevance["rows"]
            if row["cohort"] == cohort
        ]
        all_gain.append(_bootstrap(values, seed=20260970 + index))
    means = np.asarray([item["mean"] for item in all_gain])
    lows = np.asarray([item["lo"] for item in all_gain])
    highs = np.asarray([item["hi"] for item in all_gain])
    ax_value.axvline(0, color="#333333", linewidth=0.7)
    ax_value.errorbar(
        means,
        y,
        xerr=[means - lows, highs - means],
        fmt="none",
        ecolor="#333333",
        capsize=2,
        linewidth=0.8,
    )
    ax_value.scatter(means, y, c=[palette[name] for name in cohorts], s=24, zorder=3)
    ax_value.set_yticks(y, cohorts, fontsize=7)
    ax_value.set_xlabel("normalized ISR reduction", fontsize=7)
    ax_value.tick_params(axis="x", labelsize=6.5)
    ax_value.grid(axis="x", alpha=0.2)
    ax_value.invert_yaxis()

    ax_scatter = fig.add_subplot(grid[1, 0:3])
    ax_scatter.set_title("D  Root-level decision relevance", loc="left", fontsize=9, fontweight="bold")
    for cohort in cohorts:
        cohort_rows = [row for row in decision_relevance["rows"] if row["cohort"] == cohort]
        ax_scatter.scatter(
            [row["correction_to_margin_ratio"] for row in cohort_rows],
            [row["normalized_allocation_gain"] for row in cohort_rows],
            label=cohort,
            color=palette[cohort],
            alpha=0.72,
            s=22,
            edgecolor="white",
            linewidth=0.35,
        )
    ax_scatter.axvline(1.0, color="#333333", linestyle="--", linewidth=0.8, label="top-1 flip boundary")
    ax_scatter.axhline(0.0, color="#333333", linewidth=0.7)
    ax_scatter.set_xscale("symlog", linthresh=1.0, linscale=0.6)
    ax_scatter.set_xlabel("pairwise correction / deployment margin", fontsize=8)
    ax_scatter.set_ylabel("(Uniform ISR - PIVOT-KG ISR) / deployment range", fontsize=8)
    ax_scatter.tick_params(labelsize=7)
    ax_scatter.grid(alpha=0.16)
    ax_scatter.legend(ncol=4, fontsize=6.5, loc="upper center", frameon=False)

    ax_resolve = fig.add_subplot(grid[1, 3:6])
    ax_resolve.set_title("E  HF resolvability", loc="left", fontsize=9, fontweight="bold")
    for cohort in cohorts:
        cohort_rows = [row for row in decision_relevance["rows"] if row["cohort"] == cohort]
        for hit, marker, face in ((True, "o", "filled"), (False, "x", "open")):
            points = [row for row in cohort_rows if bool(row["query_hits_decision_pair"]) == hit]
            if not points:
                continue
            kwargs = {
                "marker": marker,
                "color": palette[cohort],
                "alpha": 0.78,
                "s": 24,
                "label": f"{cohort} / {'hit' if hit else 'miss'}",
            }
            if marker == "o":
                kwargs["edgecolor"] = "white"
                kwargs["linewidth"] = 0.35
            ax_resolve.scatter(
                [row["query_noise_to_proxy_margin"] for row in points],
                [row["normalized_allocation_gain"] for row in points],
                **kwargs,
            )
    ax_resolve.axvline(1.0, color="#333333", linestyle="--", linewidth=0.8)
    ax_resolve.axhline(0.0, color="#333333", linewidth=0.7)
    ax_resolve.set_xscale("symlog", linthresh=1.0, linscale=0.6)
    ax_resolve.set_xlabel("queried HF noise / proxy top-2 margin", fontsize=7.5)
    ax_resolve.set_ylabel("normalized ISR reduction", fontsize=7.5)
    ax_resolve.tick_params(labelsize=6.5)
    ax_resolve.grid(alpha=0.16)
    ax_resolve.legend(ncol=2, fontsize=5.6, loc="upper center", frameon=False)
    fig.savefig(figures / "fig1_decision_relevance.pdf", bbox_inches="tight")
    fig.savefig(figures / "fig1_decision_relevance.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    point_rows = [
        ("Leduc", "proxy", method(leduc, 8, "proxy_only")),
        ("Leduc", "no HF", rows["leduc_nohf"]),
        ("Leduc", "Uniform", rows["leduc_uniform"]),
        ("Leduc", "PIVOT", rows["leduc_pivot"]),
        ("Leduc", "all HF", rows["leduc_all"]),
        ("Melting Pot v4", "proxy", method(v4, 32, "proxy_only")),
        ("Melting Pot v4", "no HF", rows["v4_nohf"]),
        ("Melting Pot v4", "Uniform", rows["v4_uniform"]),
        ("Melting Pot v4", "PIVOT", rows["v4_pivot"]),
        ("Melting Pot v4", "all HF", rows["v4_all"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    colors = {"Leduc": "#0B5D66", "Melting Pot v4": "#B65C3A"}
    for ax, cohort in zip(axes, ("Leduc", "Melting Pot v4")):
        for _, label, row in [item for item in point_rows if item[0] == cohort]:
            ax.scatter(row["mean_hf_episode_cost"], row["mean_isr"], color=colors[cohort], s=26)
            ax.annotate(label, (row["mean_hf_episode_cost"], row["mean_isr"]), xytext=(3, 3), textcoords="offset points", fontsize=6)
        ax.set_title(cohort, fontsize=8)
        ax.set_xlabel("HF cost")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Selection regret")
    fig.tight_layout()
    fig.savefig(figures / "fig2_regret_cost.pdf", bbox_inches="tight")
    fig.savefig(figures / "fig2_regret_cost.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    architecture = paper / "snapshot/figures/fig3_pivot_architecture.pdf"
    if architecture.is_file():
        shutil.copy2(architecture, figures / "fig3_pivot_architecture.pdf")
    style_source = root / "archive/submission_v9/source/paper/iclr2027/style"
    style_target = paper / "style"
    style_target.mkdir(exist_ok=True)
    for name in STYLE_FILES:
        shutil.copy2(style_source / name, style_target / name)
    source_style_manifest = style_source.parent / "style_manifest.json"
    style_manifest = load(source_style_manifest)
    for name, expected in style_manifest.get("files", {}).items():
        candidate = style_target / name
        if not candidate.is_file() or sha256(candidate) != expected:
            raise RuntimeError(f"official style hash mismatch: {name}")
    (paper / "style_manifest.json").write_text(
        json.dumps(style_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = {
        "archive_sha256": ARCHIVE_SHA256,
        "review_sha256": REVIEW_SHA256,
        "manifest": manifest,
        "seals": seals,
        "recomputed_from_scored_rows": recomputed,
        "recomputation_checks": checks,
        "decision_relevance": decision_relevance,
        "summaries": {
            "leduc_primary": leduc_h2,
            "kuhn_primary": kuhn_h2,
            "melting_v4_primary": v4_h2,
            "melting_v4_one_query_secondary": v4_one,
            "melting_v4_short_negative": v4_short,
        },
        "generated": [
            "paper/revision_results.tex",
            "paper/tables/revision_main_results.tex",
            "paper/tables/revision_method_results.tex",
            "paper/figures/revision/fig1_cross_benchmark.pdf",
            "artifacts/revision/20260920-manuscript-update/decision_relevance.json",
            "artifacts/revision/20260920-manuscript-update/decision_relevance_rows.csv",
            "paper/figures/revision/fig1_decision_relevance.pdf",
            "paper/figures/revision/fig2_regret_cost.pdf",
        ],
        "status_boundary": {
            "leduc_v3_eta_0.40": "raw confirm roots present; delivered formal analysis is LORO only, so excluded from paper claims",
            "unrun_30_seed_draft": "not experimental evidence",
            "posthoc_noise_sensitivity": "appendix-only sensitivity",
        },
    }
    (audit_dir / "revision_evidence_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (paper / "revision_evidence_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (paper / "revision_evidence_public.json").write_text(
        json.dumps(public_audit(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    evidence = (args.evidence or root / "artifacts/revision/iclr_20260918/source").resolve()
    print(json.dumps(build(root, evidence), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
