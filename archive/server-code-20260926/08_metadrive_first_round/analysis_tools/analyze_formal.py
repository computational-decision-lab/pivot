"""Frozen calibration and independently audited confirmation for MetaDrive.

Scientific choices live in the supplied protocols/formal analysis plan.  LORO
coverage is a measurement-aware diagnostic, never an additional tuning gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

import author_adapter as author
import selector_analysis as selectors


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_bytes(path, value, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"refusing to overwrite changed frozen artifact: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_json(path, value, *, immutable=False):
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(),
                 immutable=immutable)


def _common_protocol(protocol):
    return {k: v for k, v in protocol.items() if k not in ("phase", "root_seeds")}


def _analysis_sources():
    paths = [Path(__file__), Path(selectors.__file__), Path(author.__file__), Path(selectors.pivot_v2.__file__)]
    return {p.name: sha(p) for p in paths}


def load_roots(directory, protocol, phase):
    directory = Path(directory)
    if protocol["phase"] != phase:
        raise ValueError(f"expected {phase} protocol, got {protocol['phase']}")
    expected = [int(v) for v in protocol["root_seeds"]]
    prescribed = protocol["formal_plan_if_go"][f"{phase}_root_seeds"]
    if expected != prescribed or len(expected) != (12 if phase == "calibration" else 30):
        raise ValueError("root list/count differs from the fixed 12-calibration / 30-confirmation plan")
    bank_protocol = json.loads((directory / "protocol.json").read_text())
    if bank_protocol != protocol:
        raise ValueError("bank protocol differs from analysis protocol")
    status = json.loads((directory / "status.json").read_text())
    if status.get("status") != "COMPLETE":
        raise ValueError(f"bank is incomplete: {status}")
    files = sorted((directory / "roots").glob("seed_*/root.json"))
    roots = [json.loads(p.read_text()) for p in files]
    ids = [int(w["seed"]) for w in roots]
    if len(ids) != len(set(ids)) or sorted(ids) != sorted(expected):
        raise ValueError(f"root bank does not exactly match protocol: {ids}")
    for w in roots:
        if w.get("candidate_names") != protocol["candidate_names"]:
            raise ValueError("candidate order differs from protocol")
        if w.get("adaptation_steps") != protocol["response"]["adaptation_steps"]:
            raise ValueError("adaptation order differs from protocol")
        if np.asarray(w["audit_a"]).shape != (2, 6, int(protocol["audit_episodes_per_split"])):
            raise ValueError("audit block size differs from protocol")
        if np.asarray(w["audit_b"]).shape != np.asarray(w["audit_a"]).shape:
            raise ValueError("audit A/B dimensions differ")
    return sorted(roots, key=lambda w: int(w["seed"])), {
        str(p.relative_to(directory)): sha(p) for p in files}


def coverage_diagnostics(roots, adaptations):
    result = {"diagnostic_only": True, "used_for_tuning_or_continuation_gate": False,
              "target": "noisy held-out audit mean, not latent exact deployment truth",
              "measurement_variance_formula": "(sample_var(audit_A)+sample_var(audit_B))/(4*n_per_split)",
              "interval_formula": "prior mean +/-1.96*sqrt(prior marginal variance + audit measurement variance)",
              "by_adaptation": {}}
    for h in adaptations:
        rows = []
        for w in roots:
            train = [r for r in roots if r["seed"] != w["seed"]]
            fit = selectors.fit_model(train, h)
            proxy, _ = selectors._selection_inputs(w, h)
            post = selectors._post(fit, proxy)
            hi = list(w["adaptation_steps"]).index(h)
            a, b = np.asarray(w["audit_a"], dtype=float)[hi], np.asarray(w["audit_b"], dtype=float)[hi]
            n = a.shape[1]
            labels = (a.mean(1) + b.mean(1)) / 2.0
            audit_variance = (a.var(1, ddof=1) + b.var(1, ddof=1)) / (4.0 * n)
            total_variance = np.maximum(np.diag(post.covariance), 0) + audit_variance
            radius = 1.96 * np.sqrt(total_variance)
            hits = np.abs(labels - post.mean) <= radius
            rows.append({"root": int(w["seed"]), "coverage_fraction": float(hits.mean()),
                         "covered_by_candidate": hits.tolist(), "predicted_means": post.mean.tolist(),
                         "prior_marginal_variance": np.diag(post.covariance).tolist(),
                         "audit_measurement_variance": audit_variance.tolist(),
                         "heldout_audit_mean": labels.tolist(), "half_width": radius.tolist(),
                         "training_root_ids": fit["calibration_roots"]})
        result["by_adaptation"][str(h)] = {
            "root_held_out_predictive_coverage_95": float(np.mean([r["coverage_fraction"] for r in rows])),
            "root_bootstrap_interval": selectors.bootstrap_mean([r["coverage_fraction"] for r in rows]),
            "per_root": rows}
    return result


def _verify_frozen(calibration, frozen_path, protocol):
    calibration, frozen_path = Path(calibration), Path(frozen_path)
    expected_sha = (frozen_path.parent / "frozen.sha256").read_text().split()[0]
    if sha(frozen_path) != expected_sha:
        raise ValueError("frozen.json hash mismatch")
    frozen = json.loads(frozen_path.read_text())
    if frozen["analysis_source_sha256"] != _analysis_sources():
        raise ValueError("analysis source differs from calibration freeze")
    if frozen["common_protocol_sha256"] != digest(_common_protocol(protocol)):
        raise ValueError("scientific protocol differs from calibration freeze")
    cal_protocol_path = calibration / "protocol.json"
    if sha(cal_protocol_path) != frozen["calibration_bank_protocol_file_sha256"]:
        raise ValueError("calibration protocol changed after freeze")
    actual_files = {str(p.relative_to(calibration)) for p in (calibration / "roots").glob("seed_*/root.json")}
    if actual_files != set(frozen["calibration_root_file_sha256"]):
        raise ValueError("calibration root file inventory changed after freeze")
    for relative, expected in frozen["calibration_root_file_sha256"].items():
        if sha(calibration / relative) != expected:
            raise ValueError("calibration root changed after freeze: " + relative)
    for fit in frozen["author_fits"].values():
        author._verify_provenance(fit)
    return frozen


def calibrate(calibration, out, protocol, protocol_path):
    calibration, out = Path(calibration), Path(out)
    roots, hashes = load_roots(calibration, protocol, "calibration")
    adaptations = [int(h) for h in protocol["response"]["adaptation_steps"]]
    fits = {str(h): selectors.fit_model(roots, h) for h in adaptations}
    authorfits = {str(h): author.fit_author(roots, h, protocol) for h in adaptations}
    diagnostics = coverage_diagnostics(roots, adaptations)
    analysis = out / "analysis"
    frozen = {
        "schema": "metadrive_frozen_calibration_v1", "calibration_roots": [w["seed"] for w in roots],
        "expected_confirmation_roots": protocol["formal_plan_if_go"]["confirmation_root_seeds"],
        "joint_fits": fits, "author_fits": authorfits,
        "calibration_root_file_sha256": hashes, "analysis_source_sha256": _analysis_sources(),
        "calibration_protocol_sha256": sha(protocol_path),
        "calibration_bank_protocol_file_sha256": sha(calibration / "protocol.json"),
        "common_protocol_sha256": digest(_common_protocol(protocol)),
        "protocol": protocol, "loro_diagnostic_sha256": digest(diagnostics),
        "freeze_rule": "all fits trained only on 12 calibration roots; no refitting on confirmation",
        "formal_analysis_plan_sha256": protocol.get("formal_analysis_plan_sha256"),
    }
    atomic_json(analysis / "loro_coverage.json", diagnostics)
    atomic_json(analysis / "frozen.json", frozen, immutable=True)
    frozen_sha = sha(analysis / "frozen.json")
    atomic_bytes(analysis / "frozen.sha256", (frozen_sha + "  frozen.json\n").encode(), immutable=True)
    summary = {"stage": "calibration_complete", "n_independent_roots": len(roots),
               "frozen_fit_sha256": frozen_sha, "loro_coverage": diagnostics,
               "diagnostics_are_not_gates": True, "fits_frozen_before_confirmation_analysis": True}
    atomic_json(analysis / "summary.json", summary)
    return summary


class NativeCostLedger:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.episodes = {}
        self.references = {}

    def episode(self, key):
        if key not in self.episodes:
            self.episodes[key] = json.loads((self.directory / "episodes" / (key + ".json")).read_text())
        return self.episodes[key]

    def keys_for(self, root, h, queries):
        if not queries:
            return []
        if root not in self.references:
            self.references[root] = json.loads((self.directory / "roots" / f"seed_{root}" / "episode_references.json").read_text())
        ref = self.references[root]
        keys = [key for episodes in ref["training"][0][:h] for key in episodes]
        hi = [4, 12].index(h)
        for q in queries:
            keys.extend(key for episodes in ref["training"][q][:h] for key in episodes)
            keys.extend(key for pair in ref["selection"][hi][q - 1] for key in pair)
        return keys

    def attach(self, scored):
        for row in scored:
            repetitions = row.get("queries_by_draw", [row.get("queries", [])])
            costs = []
            for queries in repetitions:
                keys = self.keys_for(int(row["root"]), int(row["adaptation"]), queries)
                if len(keys) != row["hf_episode_cost"]:
                    raise ValueError("native episode references disagree with prespecified HF charge")
                episodes = [self.episode(key) for key in keys]
                costs.append([sum(e["steps"] for e in episodes),
                              sum(e["actual_agent_steps"] for e in episodes),
                              sum(e["seconds"] for e in episodes)])
            mean = np.asarray(costs, dtype=float).mean(axis=0)
            row["hf_native_environment_steps"] = float(mean[0])
            row["hf_native_agent_steps"] = float(mean[1])
            row["hf_native_episode_compute_seconds"] = float(mean[2])
            row["native_cost_note"] = "post-decision accounting; allocation-average for Uniform_expected; cached physical generation is reported separately"

    def experiment_total(self):
        count = steps = agent_steps = 0
        seconds = 0.0
        for path in (self.directory / "episodes").glob("*.json"):
            data = self.episode(path.stem)
            count += 1
            steps += int(data["steps"])
            agent_steps += int(data["actual_agent_steps"])
            seconds += float(data["seconds"])
        return {"unique_generated_native_episodes": count, "native_environment_steps": steps,
                "native_agent_steps": agent_steps, "sum_episode_compute_seconds": seconds,
                "runner_status": json.loads((self.directory / "status.json").read_text()),
                "note": "includes all response searches, proxy, query bank, validation and audits; not a single selector's online cost"}


def _pairwise(results, method, comparator, h, budget):
    a, b = {}, {}
    for result in results:
        for row in result["scored"]:
            if row["adaptation"] == h and row["budget_queries"] == budget:
                if row["method"] == method:
                    a[row["root"]] = row["selected_audit_gain"]
                if row["method"] == comparator:
                    b[row["root"]] = row["selected_audit_gain"]
    if set(a) != set(b) or not a:
        raise ValueError("paired contrast has missing/mismatched roots")
    return selectors.bootstrap_mean([a[r] - b[r] for r in sorted(a)])


def _csv(results):
    fields = ["root", "adaptation", "budget_queries", "method", "selected_idx", "hf_queries",
              "hf_episode_cost", "hf_native_environment_steps", "hf_native_agent_steps",
              "hf_native_episode_compute_seconds", "selected_audit_gain", "empirical_oracle_regret",
              "queries", "selected_distribution", "allocation_mc_se_gain"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for result in results:
        for row in result["scored"]:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items() if key in fields})
    return stream.getvalue().encode()


def plots(summary, out, primary_h):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    labels = {"pivot_kg": "PIVOT-KG", "uniform_v2": "Uniform (registered draw)",
              "uniform_v2_expected_100": "Uniform (100 allocation mean)", "ivr_v2": "IVR",
              "lucb_v2": "LUCB", "global_voi_heuristic": "Global-VOI heuristic",
              "no_hf_v2": "Calibrated, no HF", "proxy_only": "Proxy only",
              "author_PIVOT_VOI_E5C": "Author PIVOT-VOI (E5C)",
              "author_GlobalVOI_E5C": "Author Global-VOI (E5C)",
              "author_Uniform_E5C": "Author Uniform (E5C)", "author_LUCB_E5C": "Author LUCB (E5C)"}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for group in ("matched_posterior", "author_E5C"):
        for field, title in (("selected_audit_gain", "Selected deployment gain"),
                             ("empirical_oracle_regret", "Empirical audit-oracle regret")):
            fig, ax = plt.subplots(figsize=(6.6, 4.6))
            rows = [r for r in summary["method_table"] if r["adaptation"] == primary_h and
                    r["method"].startswith("author_") == (group == "author_E5C")]
            names = list(dict.fromkeys(r["method"] for r in rows))
            for i, name in enumerate(names):
                values = sorted((r for r in rows if r["method"] == name), key=lambda r:r["budget_queries"])
                x = np.asarray([r["budget_queries"] for r in values])
                mean = np.asarray([r[field]["mean"] for r in values])
                lo = np.asarray([r[field]["lo"] for r in values]); hi = np.asarray([r[field]["hi"] for r in values])
                ax.plot(x, mean, marker=("o" if i%2 == 0 else "s"), linewidth=1.5,
                        linestyle=("--" if name in ("no_hf_v2", "proxy_only") else "-"), label=labels.get(name,name))
                ax.fill_between(x, lo, hi, alpha=.055)
            ax.set(xticks=[1,2,4], xlabel="Maximum HF candidate queries (response + paired evaluation charged)",
                   ylabel=title, title=f"MetaDrive: long response ({primary_h} profiles), 30 held-out roots")
            ax.axhline(0,color=".75",linewidth=.7)
            ax.legend(fontsize=7, loc="best", ncol=2 if len(names)>4 else 1)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                target = out / f"{group}_{field}.{ext}"
                temp = target.with_name(target.stem + f".tmp.{os.getpid()}." + ext)
                fig.savefig(temp, format=ext, dpi=220, bbox_inches="tight")
                temp.replace(target)
                files.append(str(target))
            plt.close(fig)
    return files


def confirm(calibration, out, protocol, protocol_path):
    calibration, out = Path(calibration), Path(out)
    frozen_path = calibration / "analysis" / "frozen.json"
    frozen = _verify_frozen(calibration, frozen_path, protocol)
    if set(protocol["root_seeds"]) & set(frozen["calibration_roots"]):
        raise ValueError("confirmation/calibration root overlap")
    if protocol["root_seeds"] != frozen["expected_confirmation_roots"]:
        raise ValueError("confirmation root list differs from freeze")
    roots, hashes = load_roots(out, protocol, "confirmation")
    analysis = out / "analysis"
    budgets = [int(b) for b in protocol["query_budgets"]]
    adaptations = [int(h) for h in protocol["response"]["adaptation_steps"]]
    all_decisions = []
    # Seal EVERY selector's choices for EVERY root before any confirmation scoring.
    # root.json is loaded as raw data, but decision APIs cannot access audit fields.
    for w in roots:
        for h in adaptations:
            joint_fit, author_fit = frozen["joint_fits"][str(h)], frozen["author_fits"][str(h)]
            joint = selectors.decide_root(w, joint_fit, budgets)
            original = author.decide_author(w, author_fit, budgets)
            record = {"root": w["seed"], "adaptation": h, "joint_decisions": joint,
                      "author_decisions": original, "joint_decision_sha256": digest(joint),
                      "author_decision_sha256": digest(original), "fit_sha256": sha(frozen_path),
                      "protocol_sha256": sha(protocol_path), "root_file_sha256": hashes[f"roots/seed_{w['seed']}/root.json"]}
            atomic_json(analysis / "decisions" / f"seed_{w['seed']}_h{h}.json", record, immutable=True)
            all_decisions.append(record)
    decision_manifest = {f"seed_{r['root']}_h{r['adaptation']}.json": digest(r) for r in all_decisions}
    atomic_json(analysis / "decision_manifest.json", decision_manifest, immutable=True)
    root_map, results = {int(w["seed"]):w for w in roots}, []
    ledger = NativeCostLedger(out)
    for record in all_decisions:
        w, h = root_map[int(record["root"])], record["adaptation"]
        joint = selectors.score_root(w, frozen["joint_fits"][str(h)], record["joint_decisions"], record["joint_decision_sha256"])
        original = author.score_author(w, frozen["author_fits"][str(h)], record["author_decisions"], record["author_decision_sha256"])
        joint["author_decisions"] = original["decisions"]
        joint["author_decision_sha256"] = original["decision_sha256"]
        joint["scored"] += original["scored"]
        joint["author_fit_sha256"] = original["author_fit_sha256"]
        ledger.attach(joint["scored"])
        results.append(joint)
    summary = selectors.summarize(results)
    h, budget = int(protocol["primary_adaptation"]), int(protocol["primary_query_budget"])
    primary = _pairwise(results, "pivot_kg", "uniform_v2_expected_100", h, budget)
    author_contrast = _pairwise(results, "author_PIVOT_VOI_E5C", "author_Uniform_E5C", h, budget)
    primary["evidence_supported"] = bool(primary["lo"] > 0)
    summary.update({
        "stage": "confirmation_complete", "n_independent_roots": len(roots),
        "confirmation_roots": [w["seed"] for w in roots], "primary_adaptation": h,
        "primary_query_budget": budget, "hypotheses": {
            "primary_PIVOT_KG_minus_Uniform_expected_100_selected_gain": primary,
            "secondary_author_PIVOT_VOI_minus_author_Uniform_selected_gain": author_contrast},
        "prespecified_success": bool(primary["lo"] > 0),
        "evidence_rule": "registered primary paired 95% root-bootstrap lower bound >0; null/negative results retained",
        "source_scope": protocol["scientific_scope"], "frozen_fit_sha256": sha(frozen_path),
        "confirmation_protocol_sha256": sha(protocol_path), "confirmation_root_file_sha256": hashes,
        "decision_manifest_sha256": digest(decision_manifest),
        "analysis_source_sha256": _analysis_sources(),
        "native_experiment_cost": ledger.experiment_total(),
        "budget_accounting": protocol["cost_accounting"],
        "author_comparison_scope": "author E5C batch allocation and author outcome rules; separate from matched joint-posterior selectors; all author contrasts secondary",
        "secondary_warning": "other budgets/responses/method contrasts are secondary; no multiplicity-adjusted superiority claim is made",
        "plots_note": "shaded bands are marginal root-bootstrap 95% intervals; significance uses paired primary contrast, not overlap of these bands",
    })
    for entry in summary["method_table"]:
        matched = [r for result in results for r in result["scored"]
                   if r["method"] == entry["method"] and r["adaptation"] == entry["adaptation"] and r["budget_queries"] == entry["budget_queries"]]
        entry["mean_hf_native_environment_steps"] = float(np.mean([r["hf_native_environment_steps"] for r in matched]))
        entry["mean_hf_native_agent_steps"] = float(np.mean([r["hf_native_agent_steps"] for r in matched]))
        entry["mean_hf_native_episode_compute_seconds"] = float(np.mean([r["hf_native_episode_compute_seconds"] for r in matched]))
    atomic_json(analysis / "result.json", {"summary":summary, "root_results":results})
    atomic_json(analysis / "summary.json", summary)
    atomic_bytes(analysis / "seed_results.csv", _csv(results))
    figures = plots(summary, analysis / "figures", h)
    atomic_json(analysis / "artifact_manifest.json", {
        str(p.relative_to(analysis)):sha(p) for p in analysis.rglob("*")
        if p.is_file() and p.name != "artifact_manifest.json" and ".tmp." not in p.name})
    return {"stage":"CONFIRMATION_COMPLETE", "hypotheses":summary["hypotheses"],
            "prespecified_success":summary["prespecified_success"], "figures":figures,
            "summary_path":str(analysis / "summary.json")}


def self_test():
    rng = np.random.default_rng(45)
    roots = [{"seed":i, "adaptation_steps":[4,12], "proxy":rng.normal(size=6).tolist(),
              "selection":rng.normal(size=(2,6,4)).tolist(),
              "audit_a":rng.normal(size=(2,6,8)).tolist(),
              "audit_b":rng.normal(size=(2,6,8)).tolist()} for i in (1,2,3)]
    coverage = coverage_diagnostics(roots, [4,12])
    for i,w in enumerate(roots):
        a, b = np.asarray(w["audit_a"])[0], np.asarray(w["audit_b"])[0]
        expected = (a.var(1,ddof=1)+b.var(1,ddof=1))/32
        row = coverage["by_adaptation"]["4"]["per_root"][i]
        np.testing.assert_allclose(row["audit_measurement_variance"],expected)
        assert w["seed"] not in row["training_root_ids"]
    results = [{"scored":[{"root":i,"adaptation":12,"budget_queries":2,"method":m,
                           "selected_audit_gain":float(i)+(1 if m=="pivot_kg" else 0)}
                          for m in ("pivot_kg","uniform_v2_expected_100")]} for i in range(3)]
    contrast = _pairwise(results,"pivot_kg","uniform_v2_expected_100",12,2)
    assert contrast["mean"]==contrast["lo"]==contrast["hi"]==1 and contrast["n_independent_roots"]==3
    return {"status":"OK","checks":["measurement_variance_of_two_independent_means",
            "LORO_training_root_exclusion","paired_root_contrast_not_pseudoreplicated"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["calibration","confirmation"], required=True)
    parser.add_argument("--calibration",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if args.mode == "calibration":
        if args.out.resolve() != args.calibration.resolve():
            raise ValueError("calibration --out must equal --calibration so frozen fit is discoverable")
        result = calibrate(args.calibration,args.out,protocol,args.protocol)
        print(json.dumps({"stage":"CALIBRATION_FROZEN","frozen_fit_sha256":result["frozen_fit_sha256"],
                          "coverage":{h:v["root_held_out_predictive_coverage_95"] for h,v in result["loro_coverage"]["by_adaptation"].items()},
                          "diagnostic_only":True}))
    else:
        result = confirm(args.calibration,args.out,protocol,args.protocol)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
