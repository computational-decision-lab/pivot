"""Offline analysis for Leduc v4 banks (multi-direction candidates, per-hand sealed banks). Zero rollouts.

Modes
  pilot   : mechanism / design checks on a small cohort (go / no-go gates G1-G3 from the protocol)
  loro    : leave-one-root-out replay inside the calibration cohort (+ gates C1-C3)
  confirm : fit the prior on --calibration, replay every method on --test, seal, then score

Leakage discipline: every decision (including the full decision distribution of randomised
comparators) is written to decisions_sealed.json and hashed BEFORE any exact deployment label is
read. Selectors receive only proxy deltas and a fresh SealedBank of per-hand outcomes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pivot_v3 import METHODS_V3, SealedBank, fit_prior_v3, make_posterior, run_selector  # noqa: E402

BOOT_SEED, BOOT_DRAWS = 20260917, 10000
MC_DRAWS_SMALL = 64
LABEL_SOURCES = ("exact", "noisy_audit")


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def boot(x, draws=BOOT_DRAWS, seed=BOOT_SEED):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return None
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(1)
    return {"mean": float(x.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)), "n": int(len(x))}


def signflip_p(x, draws=50000, seed=1):
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or np.all(x == 0):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(x.mean())
    flips = rng.choice([-1.0, 1.0], size=(draws, len(x)))
    return float(np.mean(np.abs((flips * x).mean(1)) >= obs - 1e-12))


# ----------------------------------------------------------------------------- data
def load_cohort(directory: Path) -> tuple[list[dict], dict]:
    """Returns (worlds without labels, exact labels keyed by root). Labels are kept apart on purpose."""
    worlds, exact = [], {}
    for d in sorted(directory.glob("seed_*")):
        st = d / "status.json"
        if not st.exists() or json.loads(st.read_text()).get("status") != "complete":
            continue
        s = json.loads((d / "summary.json").read_text())
        z = np.load(d / "bank.npz")
        steps = [int(v) for v in z["steps"]]
        K = len(s["candidates"])
        w = {"root": int(s["seed"]), "type": s["responder_type"], "m": float(s.get("responder_mixture_m", float("nan"))),
             "beta": float(s["beta"]), "candidates": s["candidates"], "K": K,
             "proxy": np.array([s["proxy_deltas"][str(i)] for i in range(K)]), "per_h": {}}
        exact[w["root"]] = {}
        for hi, h in enumerate(steps):
            paired = z["paired_diff"][hi].astype(float); new = z["new_returns"][hi].astype(float); old = z["old_indep"][hi].astype(float)
            w["per_h"][h] = {"paired": paired, "new": new, "old": old,
                             "S": paired.mean(1), "n_max": paired.shape[1],
                             "unit_var_paired": paired.var(1, ddof=1),
                             "unit_var_unpaired": new.var(1, ddof=1) + old.var(1, ddof=1),
                             "auditA": np.array([s["audit_noisy"][str(h)][str(i)]["A"] for i in range(K)]),
                             "auditB": np.array([s["audit_noisy"][str(h)][str(i)]["B"] for i in range(K)])}
            exact[w["root"]][h] = np.array([s["audit_gains"][str(h)][str(i)] for i in range(K)])
        worlds.append(w)
    if not worlds:
        raise SystemExit(f"no complete roots under {directory}")
    return worlds, exact


def fresh_bank(w: dict, h: int) -> SealedBank:
    ph = w["per_h"][h]
    return SealedBank(ph["paired"], ph["new"], ph["old"])


def fit_spec(train: list[dict], exact: dict, h: int, label_source: str):
    if label_source == "exact":
        labels = np.array([exact[w["root"]][h] for w in train]); ab = np.zeros_like(labels)
    elif label_source == "noisy_audit":
        A = np.array([w["per_h"][h]["auditA"] for w in train]); B = np.array([w["per_h"][h]["auditB"] for w in train])
        labels = 0.5 * (A + B); ab = A - B
    else:
        raise ValueError(label_source)
    g = labels - np.array([w["proxy"] for w in train])
    uvp = np.array([w["per_h"][h]["unit_var_paired"] for w in train])
    uvu = np.array([w["per_h"][h]["unit_var_unpaired"] for w in train])
    return fit_prior_v3(g, ab, uvp, uvu, label_source=label_source)


# ----------------------------------------------------------------------------- decisions (no labels here)
def decide(w: dict, spec, h: int, protocol: dict, label_source: str) -> list[dict]:
    """All method decisions for one world / adaptation / calibration source, using ONLY proxy + sealed bank."""
    sizes = [int(s) for s in protocol["query_sizes"]]
    fixed = int(protocol["fixed_query_size"])
    K = w["K"]
    out = []
    base = {"root": w["root"], "type": w["type"], "adaptation": h, "calibration": label_source}
    for cap in protocol["hf_caps"]:
        for m in protocol["methods"]:
            if m in ("no_hf", "proxy_only", "all_hf_fixed") and cap != protocol["hf_caps"][0]:
                continue
            if m == "uniform_fixed":
                # exact decision distribution over all equally likely query subsets (order irrelevant for a Gaussian update)
                b = min(K, int(cap) // (2 * fixed))
                dist = []
                for sub in itertools.combinations(range(K), b):
                    post = make_posterior(spec, w["proxy"]); bank = fresh_bank(w, h)
                    for j in sub:
                        post = post.condition(j, bank.query(j, fixed), fixed)
                    dist.append(int(np.argmax(post.means_with_incumbent())))
                reg = run_selector(make_posterior(spec, w["proxy"]), w["proxy"], fresh_bank(w, h), method=m, cap_hands=int(cap),
                                   sizes=sizes, fixed_size=fixed, seed=w["root"])
                out.append({**base, "cap": int(cap), "method": m, "selected": reg["selected"], "queries": reg["queries"],
                            "n_queries": reg["n_queries"], "hands_used": reg["hands_used"], "stop_reason": reg["stop_reason"],
                            "selected_distribution": dist, "distribution_kind": "exact_subsets"})
                continue
            if m == "uniform_small":
                dist = []
                for dr in range(MC_DRAWS_SMALL):
                    r = run_selector(make_posterior(spec, w["proxy"]), w["proxy"], fresh_bank(w, h), method=m, cap_hands=int(cap),
                                     sizes=sizes, fixed_size=fixed, seed=w["root"] * 1000 + dr)
                    dist.append(r["selected"])
                reg = run_selector(make_posterior(spec, w["proxy"]), w["proxy"], fresh_bank(w, h), method=m, cap_hands=int(cap),
                                   sizes=sizes, fixed_size=fixed, seed=w["root"])
                out.append({**base, "cap": int(cap), "method": m, "selected": reg["selected"], "queries": reg["queries"],
                            "n_queries": reg["n_queries"], "hands_used": reg["hands_used"], "stop_reason": reg["stop_reason"],
                            "selected_distribution": dist, "distribution_kind": "mc_draws"})
                continue
            r = run_selector(make_posterior(spec, w["proxy"]), w["proxy"], fresh_bank(w, h), method=m, cap_hands=int(cap),
                             sizes=sizes, fixed_size=fixed, seed=w["root"],
                             unpaired_unit_variance=np.asarray(spec.unit_variance_unpaired))
            out.append({**base, "cap": None if m in ("no_hf", "proxy_only", "all_hf_fixed") else int(cap), "method": m,
                        "selected": r["selected"], "queries": r["queries"], "n_queries": r["n_queries"],
                        "hands_used": r["hands_used"], "stop_reason": r["stop_reason"], "estimates": r["estimates"]})
    return out


# ----------------------------------------------------------------------------- scoring
def score(decisions: list[dict], exact: dict) -> list[dict]:
    scored = []
    for d in decisions:
        lab = np.concatenate([exact[d["root"]][d["adaptation"]], [0.0]])
        if "selected_distribution" in d:
            gains = np.array([lab[i] for i in d["selected_distribution"]])
            gain = float(gains.mean()); gain_reg = float(lab[d["selected"]])
        else:
            gain = float(lab[d["selected"]]); gain_reg = gain
        scored.append({**d, "gain": gain, "isr": float(lab.max() - gain), "gain_registered_draw": gain_reg,
                       "isr_registered_draw": float(lab.max() - gain_reg)})
    return scored


def _rows(scored, method, h, cap, calibration="exact"):
    return {r["root"]: r for r in scored if r["method"] == method and r["adaptation"] == h and r["calibration"] == calibration
            and (r["cap"] == cap or r["cap"] is None)}


def paired(scored, m1, m2, h, cap, calib1="exact", calib2="exact", field1="gain", field2="gain"):
    a = _rows(scored, m1, h, cap, calib1); b = _rows(scored, m2, h, cap, calib2)
    roots = sorted(set(a) & set(b))
    d = [a[r][field1] - b[r][field2] for r in roots]
    out = boot(d)
    if out:
        out["signflip_p"] = signflip_p(d); out["n_nonzero"] = int(np.sum(np.array(d) != 0))
    return out


def method_table(scored, h, cap, calibration="exact"):
    rows = []
    for m in sorted({r["method"] for r in scored}):
        rs = _rows(scored, m, h, cap, calibration)
        if not rs:
            continue
        isr = [r["isr"] for r in rs.values()]
        b = boot(isr)
        rows.append({"adaptation": h, "cap": cap, "calibration": calibration, "method": m, "mean_isr": b["mean"], "isr_lo": b["lo"], "isr_hi": b["hi"],
                     "mean_gain": float(np.mean([r["gain"] for r in rs.values()])), "mean_hands": float(np.mean([r["hands_used"] for r in rs.values()])),
                     "mean_queries": float(np.mean([r["n_queries"] for r in rs.values()])), "n": len(rs),
                     "by_type": {t: float(np.mean([r["isr"] for r in rs.values() if r["type"] == t])) for t in sorted({r["type"] for r in rs.values()})}})
    return sorted(rows, key=lambda r: r["mean_isr"])


def first_query_disagreement(scored, h, cap, m1="pivot_kg_menu", m2="ivr_menu", calibration="exact"):
    a = _rows(scored, m1, h, cap, calibration); b = _rows(scored, m2, h, cap, calibration)
    roots = sorted(set(a) & set(b))
    first = [(tuple(a[r]["queries"][0][:2]) if a[r]["queries"] else None) != (tuple(b[r]["queries"][0][:2]) if b[r]["queries"] else None) for r in roots]
    seqs = [[q[:2] for q in a[r]["queries"]] != [q[:2] for q in b[r]["queries"]] for r in roots]
    sel = [a[r]["selected"] != b[r]["selected"] for r in roots]
    return {"n_roots": len(roots), "first_query_differs": int(sum(first)), "query_sequence_differs": int(sum(seqs)), "selection_differs": int(sum(sel))}


# ----------------------------------------------------------------------------- mechanism
def mechanism_report(worlds, exact, protocol):
    rep = {}
    steps = sorted(set(int(v) for v in protocol["adaptation_steps"].values()))
    fixed = int(protocol["fixed_query_size"])
    ids = [c["id"] for c in worlds[0]["candidates"]]
    for h in steps:
        P = np.array([w["proxy"] for w in worlds]); L = np.array([exact[w["root"]][h] for w in worlds])
        types = sorted({w["type"] for w in worlds})
        r = {"candidate_ids": ids}
        for t in types:
            idx = [k for k, w in enumerate(worlds) if w["type"] == t]
            Lt = L[idx]
            r[t] = {"roots": [worlds[k]["root"] for k in idx], "mean_deployment_delta": Lt.mean(0).tolist(),
                    "mean_proxy_delta": P[idx].mean(0).tolist(),
                    "best_candidate_hist_incl_incumbent": np.bincount(np.concatenate([Lt, np.zeros((len(idx), 1))], 1).argmax(1), minlength=L.shape[1] + 1).tolist(),
                    "proxy_best_hist": np.bincount(P[idx].argmax(1), minlength=P.shape[1]).tolist()}
        ide = np.mean(np.abs(P - L), 1); pos = P > 0
        irr = np.array([np.mean(L[i][pos[i]] < 0) if pos[i].any() else np.nan for i in range(len(worlds))])
        r["IDE"] = boot(ide); r["IRR"] = boot(irr[~np.isnan(irr)]) if (~np.isnan(irr)).any() else None
        r["ISC"] = boot(np.mean(np.sign(P) == np.sign(L), 1))
        r["proxy_best_is_deployment_best_rate"] = float(np.mean(np.concatenate([L, np.zeros((len(worlds), 1))], 1).argmax(1) == P.argmax(1)))
        # informativeness: type separation vs instrument SE at the fixed query size
        se = np.sqrt(np.mean([w["per_h"][h]["unit_var_paired"] for w in worlds], 0) / fixed)
        se_unp = np.sqrt(np.mean([w["per_h"][h]["unit_var_unpaired"] for w in worlds], 0) / fixed)
        r["se_paired_fixed_size"] = se.tolist(); r["se_unpaired_fixed_size"] = se_unp.tolist()
        r["se_ratio_unpaired_over_paired"] = (se_unp / np.maximum(se, 1e-12)).tolist()
        if len(types) == 2:
            sep = np.abs(np.array(r[types[0]]["mean_deployment_delta"]) - np.array(r[types[1]]["mean_deployment_delta"]))
            ratio = sep / np.maximum(se, 1e-12)
            r["type_separation"] = sep.tolist(); r["separation_over_se"] = ratio.tolist()
            r["informativeness_heterogeneity_ratio"] = float(ratio.max() / max(ratio.min(), 1e-12))
            r["n_candidates_below_1.96se"] = int(np.sum(ratio < 1.96)); r["n_candidates_above_5se"] = int(np.sum(ratio > 5))
            r["best_candidate_differs_by_type"] = bool(int(np.argmax(np.append(r[types[0]]["mean_deployment_delta"], 0.0)))
                                                       != int(np.argmax(np.append(r[types[1]]["mean_deployment_delta"], 0.0))))
        # value of world-specific information ceiling (population prior choice vs per-world oracle)
        mu = L.mean(0); v_pop = float(L[:, mu.argmax()].mean()) if mu.max() > 0 else 0.0
        v_oracle = float(np.concatenate([L, np.zeros((len(worlds), 1))], 1).max(1).mean())
        r["population_prior_choice_value"] = v_pop; r["oracle_choice_value"] = v_oracle; r["vwi_ceiling"] = v_oracle - v_pop
        rep[str(h)] = r
    return rep


# ----------------------------------------------------------------------------- hypotheses
def evaluate_hypotheses(scored, protocol, gates: dict | None = None):
    steps = sorted(set(int(v) for v in protocol["adaptation_steps"].values())); hl, hs = max(steps), min(steps)
    cap = int(protocol["primary_hf_cap"]); m = protocol["primary_method"]; u = protocol["primary_comparator"]; acq = protocol.get("acquisition_comparator", "ivr_menu")
    out = {
        "H_A_allocation_pivot_minus_expected_uniform": paired(scored, m, u, hl, cap),
        "H_A_registered_draw": paired(scored, m, u, hl, cap, field2="gain_registered_draw"),
        "H_B_acquisition_pivot_minus_ivr_menu": paired(scored, m, acq, hl, cap),
        "H_B2_pivot_minus_lucb_fixed": paired(scored, m, "lucb_fixed", hl, cap),
        "H_B3_pivot_minus_top_proxy_fixed": paired(scored, m, "top_proxy_fixed", hl, cap),
        "H_C_pairing_pivot_minus_unpaired": paired(scored, m, "pivot_kg_menu_unpaired", hl, cap),
        "H_D_exact_minus_noisy_calibration": paired(scored, m, m, hl, cap, calib1="exact", calib2="noisy_audit"),
        "H_D_noisy_calibration_pivot_minus_expected_uniform": paired(scored, m, u, hl, cap, calib1="noisy_audit", calib2="noisy_audit"),
        "H_E_stop_gain_minus_menu": paired(scored, "pivot_kg_menu_stop", m, hl, cap),
        "H_E_stop_hands_minus_menu_hands": paired(scored, "pivot_kg_menu_stop", m, hl, cap, field1="hands_used", field2="hands_used"),
        "H_F_menu_minus_fixed_size": paired(scored, m, "pivot_kg_fixed", hl, cap),
        "secondary": {
            "pivot_minus_no_hf (query value)": paired(scored, m, "no_hf", hl, cap),
            "no_hf_minus_proxy_only (calibration value)": paired(scored, "no_hf", "proxy_only", hl, cap),
            "pivot_minus_all_hf_fixed": paired(scored, m, "all_hf_fixed", hl, cap),
            "pivot_minus_uniform_small": paired(scored, m, "uniform_small", hl, cap),
            "ivr_menu_minus_expected_uniform": paired(scored, acq, u, hl, cap),
            "short_pivot_minus_expected_uniform": paired(scored, m, u, hs, cap),
        },
        "by_cap": {str(c): {"pivot_minus_expected_uniform": paired(scored, m, u, hl, int(c)), "pivot_minus_ivr_menu": paired(scored, m, acq, hl, int(c)),
                            "pivot_minus_unpaired": paired(scored, m, "pivot_kg_menu_unpaired", hl, int(c))} for c in protocol["hf_caps"]},
        "acquisition_discriminability": {str(c): first_query_disagreement(scored, hl, int(c)) for c in protocol["hf_caps"]},
    }
    a = out["H_A_allocation_pivot_minus_expected_uniform"]; s = out["secondary"]["short_pivot_minus_expected_uniform"]
    out["H_G_interaction_long_minus_short"] = {"mean": a["mean"] - s["mean"], "note": "short-response contrast is reported separately; see the paired per-root interaction below"} if a and s else None
    a_rows = _rows(scored, m, hl, cap); u_rows = _rows(scored, u, hl, cap); a_s = _rows(scored, m, hs, cap); u_s = _rows(scored, u, hs, cap)
    roots = sorted(set(a_rows) & set(u_rows) & set(a_s) & set(u_s))
    inter = [(a_rows[r]["gain"] - u_rows[r]["gain"]) - (a_s[r]["gain"] - u_s[r]["gain"]) for r in roots]
    out["H_G_interaction_long_minus_short_paired"] = boot(inter)
    out["prespecified_success_H_A"] = bool(a and a["lo"] > 0)
    out["H_B_testable_by_design_gate"] = None if gates is None else bool(gates.get("C3_pass"))
    return out


# ----------------------------------------------------------------------------- IO helpers
def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


def write_csv(path: Path, rows: list[dict], fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "loro", "confirm"], required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--test", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    protocol = json.loads(a.protocol.read_text())
    steps = sorted(set(int(v) for v in protocol["adaptation_steps"].values())); hl = max(steps)
    a.output.mkdir(parents=True, exist_ok=True)
    sources = tuple(protocol.get("calibration_label_sources", ["exact"]))

    if a.mode == "pilot":
        worlds, exact = load_cohort(a.calibration)
        rep = mechanism_report(worlds, exact, protocol)
        r = rep[str(hl)]; crit = protocol.get("pilot_go_criteria", {})
        fixed = int(protocol["fixed_query_size"]); smallest = min(int(s) for s in protocol["query_sizes"])
        se = np.sqrt(np.mean([w["per_h"][hl]["unit_var_paired"] for w in worlds], 0) / fixed)
        eq = [w for w in worlds if w["type"] == "equilibrator"]
        contest, value = [], []
        for w in eq:
            L = exact[w["root"]][hl]; best = float(max(L.max(), 0.0))
            contest.append(int(np.sum(L >= best - 1.96 * se))); value.append(best)
        g2a = float(np.mean(contest)) if contest else float("nan"); g2b = float(np.mean(value)) if value else float("nan")
        checks = {"G1_best_candidate_differs_by_type": bool(r.get("best_candidate_differs_by_type", False)),
                  "G2a_within_type_contest": bool(contest and g2a >= float(crit.get("G2a_within_type_contest_min_candidates", 2.0))),
                  "G2b_value_to_find": bool(value and g2b > float(crit.get("G2b_value_to_find_min_se_multiple", 3.0)) * float(se.mean()))}
        rep["G2_details"] = {"equilibrator_roots": [w["root"] for w in eq], "n_candidates_within_1.96se_of_best": contest, "mean_contest_size": g2a,
                             "best_gain_over_incumbent": value, "mean_best_gain": g2b, "mean_se_fixed_size": float(se.mean())}
        if "type_separation" in r:
            se_small = np.sqrt(np.mean([w["per_h"][hl]["unit_var_paired"] for w in worlds], 0) / smallest)
            ratio_small = np.array(r["type_separation"]) / np.maximum(se_small, 1e-12)
            rep["G3_reported_only"] = {"separation_over_se_fixed": r.get("separation_over_se"), "heterogeneity_ratio_fixed": r.get("informativeness_heterogeneity_ratio"),
                                      "smallest_query_size": smallest, "separation_over_se_smallest": ratio_small.tolist(),
                                      "n_below_1.96se_smallest": int(np.sum(ratio_small < 1.96)), "n_above_5se_smallest": int(np.sum(ratio_small > 5))}
        rep["pilot_checks"] = checks; rep["pilot_go"] = all(checks.values()); rep["n_roots"] = len(worlds)
        write_json(a.output / "pilot_report.json", rep)
        print(json.dumps({"pilot_go": rep["pilot_go"], "checks": checks, "G2_details": rep["G2_details"], "G3": rep.get("G3_reported_only"),
                          "candidate_ids": r["candidate_ids"]}, indent=2))
        return

    coverage = {}
    gates = None
    if a.mode == "loro":
        worlds, exact = load_cohort(a.calibration)
        decisions = []; hits = defaultdict(list)
        for w in worlds:
            train = [x for x in worlds if x["root"] != w["root"]]
            for h in steps:
                for src in sources:
                    spec = fit_spec(train, exact, h, src)
                    decisions += decide(w, spec, h, protocol, src)
                    if src == "exact":
                        post = make_posterior(spec, w["proxy"])
                        sd = np.sqrt(np.diag(post.covariance) + post.unit_variance / w["per_h"][h]["n_max"])
                        hits[h] += list(np.abs(w["per_h"][h]["S"] - post.mean) <= 1.96 * sd)
        coverage = {str(h): float(np.mean(v)) for h, v in hits.items()}
    else:
        cal, exact_cal = load_cohort(a.calibration)
        worlds, exact = load_cohort(a.test)
        assert not ({w["root"] for w in cal} & {w["root"] for w in worlds}), "calibration and test roots overlap"
        specs = {(h, src): fit_spec(cal, exact_cal, h, src) for h in steps for src in sources}
        write_json(a.output / "prior_spec_v3.json", {f"{h}|{src}": specs[(h, src)].__dict__ for (h, src) in specs})
        decisions = [d for w in worlds for h in steps for src in sources for d in decide(w, specs[(h, src)], h, protocol, src)]
    # ---- seal before touching labels
    blob = json.dumps(decisions, sort_keys=True, default=float).encode()
    (a.output / "decisions_sealed.json").write_bytes(blob)
    seal = sha_bytes(blob)
    write_json(a.output / "selection_seal.json", {"decisions_sha256": seal, "mode": a.mode, "n_decisions": len(decisions),
                                                   "test_roots": sorted({d["root"] for d in decisions})})
    # ---- score
    scored = score(decisions, exact)
    assert sha_bytes((a.output / "decisions_sealed.json").read_bytes()) == seal
    write_json(a.output / "scored_decisions.json", scored)
    cap = int(protocol["primary_hf_cap"])
    if a.mode == "loro":
        g = protocol.get("preflight_gate_before_confirmation_freeze", {})
        lo_hi = g.get("C1_root_held_out_predictive_coverage_95", [0.8, 1.0])
        kg_no = paired(scored, protocol["primary_method"], "no_hf", hl, cap)
        disc = first_query_disagreement(scored, hl, cap)
        gates = {"C1_coverage_pass": bool(all(lo_hi[0] <= coverage[str(h)] <= lo_hi[1] for h in steps)), "coverage": coverage,
                 "C2_pivot_not_worse_than_no_hf_pass": bool(kg_no and kg_no["mean"] > -float(g.get("C2_loro_long_pivot_kg_menu_not_worse_than_no_hf_by_more_than", 0.02))),
                 "C2_contrast": kg_no, "C3_pass": bool(disc["first_query_differs"] >= 3), "C3_first_query_disagreement": disc}
        gates["all_pass"] = bool(gates["C1_coverage_pass"] and gates["C2_pivot_not_worse_than_no_hf_pass"])
    hyp = evaluate_hypotheses(scored, protocol, gates)
    result = {"mode": a.mode, "n_roots": len(worlds), "types": {t: sum(w["type"] == t for w in worlds) for t in sorted({w["type"] for w in worlds})},
              "candidate_ids": [c["id"] for c in worlds[0]["candidates"]], "mechanism": mechanism_report(worlds, exact, protocol),
              "method_tables": {f"h{h}|cap{c}|{src}": method_table(scored, h, int(c), src) for h in steps for c in protocol["hf_caps"] for src in sources},
              "hypotheses": hyp, "gates": gates, "root_held_out_predictive_coverage_95": coverage, "decisions_sha256": seal,
              "primary_cap": cap, "caps": protocol["hf_caps"], "query_sizes": protocol["query_sizes"]}
    write_json(a.output / "summary.json", result)
    write_csv(a.output / "seed_results.csv", scored, ["root", "type", "adaptation", "calibration", "cap", "method", "selected", "n_queries", "hands_used",
                                                       "gain", "isr", "gain_registered_draw", "isr_registered_draw", "stop_reason", "queries"])
    print(json.dumps({"hypotheses": {k: v for k, v in hyp.items() if not k.startswith(("secondary", "by_cap"))}, "gates": gates}, indent=2, default=str))
    for r in method_table(scored, hl, cap, "exact"):
        print(f"h={r['adaptation']:>2} cap={r['cap']} {r['method']:24s} ISR={r['mean_isr']:.4f} [{r['isr_lo']:.4f},{r['isr_hi']:.4f}] gain={r['mean_gain']:+.4f} hands={r['mean_hands']:7.0f} q={r['mean_queries']:.1f} by_type={ {k: round(v, 4) for k, v in r['by_type'].items()} }")


if __name__ == "__main__":
    main()
