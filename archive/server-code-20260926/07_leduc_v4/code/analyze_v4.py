"""Offline analysis for v4 cohorts. Zero native episodes.

Modes
  pilot   : mechanism / manipulation check on a small cohort (go / no-go for the design)
  loro    : leave-one-root-out method replay INSIDE one cohort (pre-flight on calibration roots)
  confirm : fit posterior on --calibration cohort, replay all methods on --test cohort, then score

Leakage discipline (confirm/loro): all method decisions are computed and written to
decisions_sealed.json (with SHA256) BEFORE any audit label is read. Selectors receive only
proxy deltas and a callback into the sealed selection bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pivot_v2 import METHODS_V2, fit_posterior_v2, make_posterior, run_selector  # noqa: E402

GRID = np.array([.125, .225, .325, .425, .575, .675, .775, .875])  # Melting Pot default; overridden by protocol
CAPS = (96, 192, 384)      # Melting Pot defaults; overridden by protocol in set_candidate_geometry
PRIMARY_CAP = 192
QUERY_COST = None          # None -> Melting Pot rule h+32 with setup h; else flat per-query cost, no setup
LEDUC_GEOMETRY = False     # Keep Leduc-specific corrections out of other benchmarks.
BOOT_SEED, BOOT_DRAWS = 20260917, 10000


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_cohort(directory: Path, adaptations) -> list[dict]:
    worlds = []
    global K_CAND
    for d in sorted(directory.glob("seed_*")):
        st = d / "status.json"
        if not st.exists() or json.loads(st.read_text()).get("status") != "complete":
            continue
        s = json.loads((d / "summary.json").read_text())
        K_CAND = len(s["proxy_deltas"])
        w = {"root": s["seed"], "type": s["responder_type"], "proxy": np.array([s["proxy_deltas"][str(i)] for i in range(K_CAND)]),
             "q": s.get("response_probabilities"), "per_h": {}, "_audit": {}}
        for h in adaptations:
            hs = str(h)
            S = np.array([s["selection_bank"][hs][str(i)] for i in range(K_CAND)])
            A = np.array([next(m["deployment_delta_audit"] for m in s["mechanism_audit"] if m["candidate"] == str(i) and m["adaptation"] == h and m["block"] == "A") for i in range(K_CAND)])
            B = np.array([next(m["deployment_delta_audit"] for m in s["mechanism_audit"] if m["candidate"] == str(i) and m["adaptation"] == h and m["block"] == "B") for i in range(K_CAND)])
            PA = np.array([next(m["proxy_delta_audit"] for m in s["mechanism_audit"] if m["candidate"] == str(i) and m["adaptation"] == h and m["block"] == "A") for i in range(K_CAND)])
            PB = np.array([next(m["proxy_delta_audit"] for m in s["mechanism_audit"] if m["candidate"] == str(i) and m["adaptation"] == h and m["block"] == "B") for i in range(K_CAND)])
            own = np.array([np.mean([m["responder_own_gain"] for m in s["mechanism_audit"] if m["candidate"] == str(i) and m["adaptation"] == h]) for i in range(K_CAND)])
            w["per_h"][h] = {"S": S, "cost": float(h + 32) if QUERY_COST is None else QUERY_COST, "setup": float(h) if QUERY_COST is None else 0.0}
            w["_audit"][h] = {"A": A, "B": B, "PA": PA, "PB": PB, "label": 0.5 * (A + B), "own_gain": own}
        worlds.append(w)
    if not worlds:
        raise SystemExit(f"no complete roots under {directory}")
    return worlds


K_CAND = 8
DISTANCES = None


def set_candidate_geometry(protocol):
    global GRID, DISTANCES, CAPS, PRIMARY_CAP, QUERY_COST, LEDUC_GEOMETRY
    LEDUC_GEOMETRY = protocol.get("game") == "leduc_poker"
    if "hf_caps" in protocol:
        CAPS = tuple(protocol["hf_caps"]); PRIMARY_CAP = protocol["primary_hf_cap"]; QUERY_COST = float(protocol["query_cost"])
    elif "hf_episode_caps" in protocol:
        CAPS = tuple(protocol["hf_episode_caps"]); PRIMARY_CAP = protocol["primary_hf_episode_cap"]
    if "candidate_probabilities" in protocol:
        GRID = np.array(protocol["candidate_probabilities"]); DISTANCES = np.abs(GRID - .5)
    elif "candidate_alphas" in protocol:
        GRID = np.array(protocol["candidate_alphas"]); DISTANCES = np.arange(len(GRID), dtype=float)  # no pooling


def fit_spec(train: list[dict], h: int):
    g = np.array([w["_audit"][h]["label"] - w["proxy"] for w in train])
    ab = np.array([w["_audit"][h]["A"] - w["_audit"][h]["B"] for w in train])
    sab = np.array([(w["per_h"][h]["S"] - w["_audit"][h]["A"]) * (w["per_h"][h]["S"] - w["_audit"][h]["B"]) for w in train])
    noise_args = {"r_floor": 1e-8} if LEDUC_GEOMETRY else {}
    return fit_posterior_v2(g, ab, sab, distances=DISTANCES if DISTANCES is not None else np.abs(GRID - .5), **noise_args)


AUTHOR = None


def load_author():
    """Author E5C batch allocation + outcome rule, if the frozen author package is importable."""
    global AUTHOR
    try:
        from pivot.acquisition.pivot_voi import BayesianLinearDeltaPosterior
        from experiments.v9 import e5c_efficiency as e5c
        AUTHOR = {"posterior_cls": BayesianLinearDeltaPosterior, "e5c": e5c}
    except Exception as exc:  # noqa: BLE001
        AUTHOR = {"error": f"{type(exc).__name__}: {exc}"}
    return AUTHOR


def author_features(k: int) -> np.ndarray:
    """Domain features vanishing at the incumbent: signed and absolute normalised candidate parameter."""
    incumbent = 0.0 if LEDUC_GEOMETRY else 0.5
    x = (GRID - incumbent) / 0.25   # Leduc's incumbent copy is alpha=0; retain the original feature scale.
    return np.stack([x, np.abs(x)], axis=1)


def fit_author_posterior(train: list[dict], h: int):
    """Author E5C rule: Bayesian linear correction, noise = population variance of calibration corrections."""
    if not AUTHOR or "error" in AUTHOR:
        return None
    X = []; y = []
    F = author_features(len(train[0]["proxy"]))
    for w in train:
        g = w["_audit"][h]["label"] - w["proxy"]
        X.extend(F.tolist()); y.extend(g.tolist())
    return AUTHOR["posterior_cls"](noise_variance=max(float(np.var(y)), 1e-4)).fit(np.array(X), np.array(y))


def author_decisions(w: dict, post, h: int, cap: int) -> list[dict]:
    if post is None:
        return []
    e5c = AUTHOR["e5c"]; ph = w["per_h"][h]; F = author_features(len(w["proxy"]))
    budget = min(len(w["proxy"]), max(0, int((cap - ph["setup"]) // ph["cost"])))
    rows = [{"transition_id": str(i), "delta_proxy": float(w["proxy"][i]), "features": F[i].tolist(), "hf_query_cost": ph["cost"]} for i in range(len(w["proxy"]))]
    out = []
    for m in ("random_hf", "paired_lucb", "global_voi", "pivot_voi"):
        try:
            selected = e5c._select(m, rows, post, budget, w["root"], {"statistics": {"voi_fantasies": 64, "voi_posterior_samples": 256}})
            observed = {j: float(ph["S"][int(j)]) for j in selected}
            incumbent = {"transition_id": "incumbent", "delta_proxy": 0.0, "features": [0.0, 0.0], "hf_query_cost": 1.0}
            outcome = [dict(r, delta_true=observed.get(r["transition_id"], float("nan"))) for r in rows + [incumbent]]
            pick, _ = e5c._select_outcome(outcome, selected, m, post)
            sel = len(w["proxy"]) if pick == "incumbent" else int(pick)
            out.append({"root": w["root"], "type": w["type"], "adaptation": h, "cap": cap, "method": "author_" + m, "selected": sel,
                        "queried": [int(j) for j in selected], "hf_queries": len(selected),
                        "hf_episode_cost": len(selected) * ph["cost"] + (ph["setup"] if selected else 0), "stop_reason": "batch", "estimates": None})
        except Exception as exc:  # noqa: BLE001
            out.append({"root": w["root"], "type": w["type"], "adaptation": h, "cap": cap, "method": "author_" + m, "selected": len(w["proxy"]),
                        "queried": [], "hf_queries": 0, "hf_episode_cost": 0.0, "stop_reason": f"author_error:{type(exc).__name__}", "estimates": None})
    return out


def decide(w: dict, spec, h: int, seed_offset: int = 0, author_post=None) -> list[dict]:
    """All method decisions for one world/adaptation, using ONLY proxy + sealed bank."""
    ph = w["per_h"][h]
    out = []
    for cap in CAPS:
        budget = min(len(w["proxy"]), max(0, int((cap - ph["setup"]) // ph["cost"])))
        for m in METHODS_V2:
            if m in ("no_hf_v2", "proxy_only") and cap != CAPS[0]:
                continue
            post = make_posterior(spec, w["proxy"])
            r = run_selector(post, w["proxy"], lambda j: ph["S"][j], method=m, budget=budget,
                             costs=np.full(len(w["proxy"]), ph["cost"]), seed=w["root"] + seed_offset)
            setup = ph["setup"] if r["hf_queries"] else 0
            out.append({"root": w["root"], "type": w["type"], "adaptation": h, "cap": None if m in ("no_hf_v2", "proxy_only") else cap,
                        "method": m, "selected": r["selected"], "queried": r["queried"], "hf_queries": r["hf_queries"],
                        "hf_episode_cost": r["charged_cost"] + setup, "stop_reason": r["stop_reason"], "estimates": r["estimates"]})
        out += author_decisions(w, author_post, h, cap)
    # all-HF reference: query every candidate in the bank, pick the noisy argmax (incumbent admissible)
    est = np.concatenate([ph["S"], [0.0]])
    out.append({"root": w["root"], "type": w["type"], "adaptation": h, "cap": None, "method": "all_hf_reference",
                "selected": int(np.argmax(est)), "queried": list(range(len(w["proxy"]))), "hf_queries": len(w["proxy"]), "hf_episode_cost": len(w["proxy"]) * ph["cost"] + ph["setup"],
                "stop_reason": "all", "estimates": est.tolist()})
    return out


def score(decisions: list[dict], worlds: list[dict]) -> list[dict]:
    by_root = {w["root"]: w for w in worlds}
    scored = []
    for d in decisions:
        lab = np.concatenate([by_root[d["root"]]["_audit"][d["adaptation"]]["label"], [0.0]])
        scored.append({**d, "gain": float(lab[d["selected"]]), "isr": float(lab.max() - lab[d["selected"]])})
    return scored


def boot(x, draws=BOOT_DRAWS, seed=BOOT_SEED):
    x = np.asarray(x, dtype=float); rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(1)
    return {"mean": float(x.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)), "n": int(len(x))}


def paired(scored, m1, m2, h, cap=None):
    cap = PRIMARY_CAP if cap is None else cap
    a = {r["root"]: r["gain"] for r in scored if r["method"] == m1 and r["adaptation"] == h and (r["cap"] == cap or r["cap"] is None)}
    b = {r["root"]: r["gain"] for r in scored if r["method"] == m2 and r["adaptation"] == h and (r["cap"] == cap or r["cap"] is None)}
    roots = sorted(set(a) & set(b))
    return boot([a[r] - b[r] for r in roots]) if roots else None


def interaction(scored, m1, m2, h_long, h_short, cap=None):
    cap = PRIMARY_CAP if cap is None else cap
    def g(root, m, h):
        return next(r["gain"] for r in scored if r["root"] == root and r["method"] == m and r["adaptation"] == h and (r["cap"] == cap or r["cap"] is None))
    roots = sorted({r["root"] for r in scored})
    return boot([(g(r, m1, h_long) - g(r, m2, h_long)) - (g(r, m1, h_short) - g(r, m2, h_short)) for r in roots])


def mechanism_report(worlds, adaptations):
    rep = {}
    for h in adaptations:
        by_type = defaultdict(list)
        for w in worlds:
            by_type[w["type"]].append(w)
        rep[str(h)] = {}
        for t, ws in by_type.items():
            L = np.array([w["_audit"][h]["label"] for w in ws]); P = np.array([w["proxy"] for w in ws])
            has_q = all(w["q"] for w in ws)
            q7 = [w["q"]["7"][str(h)] for w in ws] if has_q else [float("nan")]; q0 = [w["q"]["0"][str(h)] for w in ws] if has_q else [float("nan")]; qi = [w["q"]["incumbent"][str(h)] for w in ws] if has_q else [float("nan")]
            rep[str(h)][t] = {"roots": [w["root"] for w in ws], "mean_label_by_candidate": L.mean(0).tolist(),
                              "best_candidate_hist": np.bincount(np.concatenate([L, np.zeros((len(ws), 1))], 1).argmax(1), minlength=L.shape[1] + 1).tolist(),
                              "proxy_best_hist": np.bincount(P.argmax(1), minlength=P.shape[1]).tolist(),
                              "q_after_h_vs_candidate7_mean": float(np.mean(q7)), "q_after_h_vs_candidate0_mean": float(np.mean(q0)),
                              "q_after_h_vs_incumbent_mean": float(np.mean(qi)),
                              "candidate_last_deployment_delta_mean": float(L[:, -1].mean()), "candidate0_deployment_delta_mean": float(L[:, 0].mean()),
                              "candidate7_deployment_delta_mean": float(L[:, min(7, L.shape[1]-1)].mean())}
        # cross-block squared gap (noise corrected), all roots
        gaps = [np.mean((w["_audit"][h]["A"] - w["_audit"][h]["PA"]) * (w["_audit"][h]["B"] - w["_audit"][h]["PB"])) for w in worlds]
        rep[str(h)]["noise_corrected_squared_gap_all_roots"] = boot(gaps)
        # paper metrics (Sec. 3) on transition rows, aggregated per root then bootstrapped over roots
        P = np.array([w["proxy"] for w in worlds]); L = np.array([w["_audit"][h]["label"] for w in worlds])
        ide = np.mean(np.abs(P - L), axis=1)
        pos = P > 0
        irr = np.array([np.mean(L[i][pos[i]] < 0) if pos[i].any() else np.nan for i in range(len(worlds))])
        isc = np.array([np.mean(np.sign(P[i]) == np.sign(L[i])) for i in range(len(worlds))])
        rep[str(h)]["IDE_mean_abs_delta_error"] = boot(ide)
        rep[str(h)]["IRR_reversal_rate_given_proxy_positive"] = boot(irr[~np.isnan(irr)]) if (~np.isnan(irr)).any() else None
        rep[str(h)]["ISC_sign_consistency"] = boot(isc)
        rep[str(h)]["proxy_best_is_deployment_best_rate"] = float(np.mean(P.argmax(1) == L.argmax(1)))
    if len(adaptations) == 2:
        hl, hs = max(adaptations), min(adaptations)
        d = [np.mean((w["_audit"][hl]["A"] - w["_audit"][hl]["PA"]) * (w["_audit"][hl]["B"] - w["_audit"][hl]["PB"]))
             - np.mean((w["_audit"][hs]["A"] - w["_audit"][hs]["PA"]) * (w["_audit"][hs]["B"] - w["_audit"][hs]["PB"])) for w in worlds]
        rep["squared_gap_long_minus_short"] = boot(d)
        rep["responder_own_gain_long_minus_short"] = boot([np.mean(w["_audit"][hl]["own_gain"] - w["_audit"][hs]["own_gain"]) for w in worlds])
    # value of world-specific information ceiling (pick on A, score on B)
    for h in adaptations:
        A = np.array([w["_audit"][h]["A"] for w in worlds]); B = np.array([w["_audit"][h]["B"] for w in worlds]); L = (A + B) / 2
        mu = L.mean(0); n = len(worlds)
        v_pop = float(L[:, mu.argmax()].mean()) if mu.max() > 0 else 0.0
        v_info = float((B[np.arange(n), A.argmax(1)].mean() + A[np.arange(n), B.argmax(1)].mean()) / 2)
        rep[str(h)]["population_prior_choice_value"] = v_pop
        rep[str(h)]["perfect_world_specific_info_choice_value"] = v_info
        rep[str(h)]["value_of_world_specific_information_ceiling"] = v_info - v_pop
    return rep


def method_table(scored, cap=None):
    cap = PRIMARY_CAP if cap is None else cap
    by = defaultdict(list)
    for r in scored:
        if r["cap"] == cap or r["cap"] is None:
            by[(r["adaptation"], r["method"])].append(r)
    rows = []
    for (h, m), rs in by.items():
        rows.append({"adaptation": h, "method": m, "mean_isr": float(np.mean([r["isr"] for r in rs])), "mean_gain": float(np.mean([r["gain"] for r in rs])),
                     "mean_hf_episode_cost": float(np.mean([r["hf_episode_cost"] for r in rs])), "n": len(rs),
                     "by_type": {t: float(np.mean([r["isr"] for r in rs if r["type"] == t])) for t in sorted({r["type"] for r in rs})}})
    return sorted(rows, key=lambda r: (r["adaptation"], r["mean_isr"]))


def adaptations_of(protocol):
    return protocol["adaptation_episodes"] if "adaptation_episodes" in protocol else sorted(protocol["adaptation_steps"].values())


def evaluate_hypotheses(scored, protocol):
    hl, hs = max(adaptations_of(protocol)), min(adaptations_of(protocol))
    m, u = protocol["primary_method"], protocol["primary_comparator"]
    out = {"H2_long_primary_gain_minus_comparator_cap192": paired(scored, m, u, hl),
           "H3_interaction_long_minus_short": interaction(scored, m, u, hl, hs),
           "secondary": {
               "long_smallest_cap_single_query_kg_minus_uniform": paired(scored, m, u, hl, cap=CAPS[0]),
               "long_kg_minus_ivr_v2": paired(scored, m, "ivr_v2", hl),
               "long_kg_minus_no_hf_v2 (query value over calibration)": paired(scored, m, "no_hf_v2", hl),
               "long_no_hf_v2_minus_proxy_only (calibration value)": paired(scored, "no_hf_v2", "proxy_only", hl),
               "long_kg_stop_minus_kg": paired(scored, "pivot_kg_stop", m, hl),
               "short_kg_minus_uniform": paired(scored, m, u, hs),
           }}
    p = out["H2_long_primary_gain_minus_comparator_cap192"]; i = out["H3_interaction_long_minus_short"]
    out["prespecified_success"] = bool(p and p["lo"] > 0 and i and i["lo"] > 0)
    return out


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "loro", "confirm"], required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, help="cohort dir used to fit the posterior (confirm) or to replay (loro/pilot)")
    ap.add_argument("--test", type=Path, help="cohort dir replayed and scored (confirm)")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    protocol = json.loads(a.protocol.read_text())
    set_candidate_geometry(protocol)
    load_author()
    adapt = adaptations_of(protocol)
    a.output.mkdir(parents=True, exist_ok=True)

    if a.mode == "pilot":
        worlds = load_cohort(a.calibration, adapt)
        rep = mechanism_report(worlds, adapt)
        checks = protocol.get("pilot_go_criteria", {})
        hl = str(max(adapt))
        got = {}
        types = [t for t in rep[hl] if isinstance(rep[hl][t], dict) and "roots" in rep[hl][t]]
        if len(types) == 2 and not all(w["q"] for w in worlds):   # generic (OpenSpiel) pilot criteria
            ta, tb = types
            got["best_candidate_differs_by_type"] = int(np.argmax(rep[hl][ta]["mean_label_by_candidate"])) != int(np.argmax(rep[hl][tb]["mean_label_by_candidate"]))
            ceiling = rep[hl]["value_of_world_specific_information_ceiling"]; oracle = rep[hl]["perfect_world_specific_info_choice_value"]
            got["world_specific_info_ceiling_at_least_fraction_of_oracle"] = bool(oracle > 0 and ceiling >= float(checks.get("min_info_ceiling_fraction_of_oracle", 0.2)) * oracle)
        if "competitive" in rep[hl] and "cooperative" in rep[hl]:
            got["competitive_q_vs_candidate7_below_incumbent_q"] = rep[hl]["competitive"]["q_after_h_vs_candidate7_mean"] < rep[hl]["competitive"]["q_after_h_vs_incumbent_mean"]
            got["cooperative_q_vs_candidate7_above_incumbent_q"] = rep[hl]["cooperative"]["q_after_h_vs_candidate7_mean"] > rep[hl]["cooperative"]["q_after_h_vs_incumbent_mean"]
            got["candidate7_deployment_sign_differs_by_type"] = (rep[hl]["cooperative"]["candidate7_deployment_delta_mean"] > 0) and (rep[hl]["competitive"]["candidate7_deployment_delta_mean"] < rep[hl]["cooperative"]["candidate7_deployment_delta_mean"] - float(checks.get("min_type_separation_return_units", 5.0)))
            got["best_candidate_differs_by_type"] = int(np.argmax(rep[hl]["cooperative"]["mean_label_by_candidate"])) != int(np.argmax(rep[hl]["competitive"]["mean_label_by_candidate"]))
        rep["pilot_checks"] = got
        rep["pilot_go"] = bool(got) and all(got.values())
        write_json(a.output / "pilot_report.json", rep)
        print(json.dumps({"pilot_go": rep["pilot_go"], "checks": got}, indent=2))
        return

    coverage = {}
    if a.mode == "loro":
        worlds = load_cohort(a.calibration, adapt)
        decisions = []
        hits = defaultdict(list)
        for w in worlds:
            train = [x for x in worlds if x["root"] != w["root"]]
            for h in adapt:
                spec = fit_spec(train, h)
                decisions += decide(w, spec, h, author_post=fit_author_posterior(train, h))
                # root-held-out predictive coverage of the selection observations (uses S only, no labels)
                post = make_posterior(spec, w["proxy"])
                sd = np.sqrt(np.diag(post.covariance) + post.observation_variance)
                hits[h] += list(np.abs(w["per_h"][h]["S"] - post.mean) <= 1.96 * sd)
        coverage = {str(h): float(np.mean(v)) for h, v in hits.items()}
    else:
        cal = load_cohort(a.calibration, adapt)
        worlds = load_cohort(a.test, adapt)
        assert not ({w["root"] for w in cal} & {w["root"] for w in worlds}), "calibration and test roots overlap"
        specs = {h: fit_spec(cal, h) for h in adapt}
        write_json(a.output / "posterior_v2_spec.json", {str(h): specs[h].__dict__ for h in adapt})
        author_posts = {h: fit_author_posterior(cal, h) for h in adapt}
        decisions = [d for w in worlds for h in adapt for d in decide(w, specs[h], h, author_post=author_posts[h])]
    # ---- seal decisions before touching labels
    blob = json.dumps(decisions, sort_keys=True, default=float).encode()
    (a.output / "decisions_sealed.json").write_bytes(blob)
    seal = sha_bytes(blob)
    write_json(a.output / "selection_seal.json", {"decisions_sha256": seal, "mode": a.mode, "n_decisions": len(decisions),
                                                   "test_roots": sorted({d["root"] for d in decisions})})
    # ---- phase 2: score
    scored = score(decisions, worlds)
    assert sha_bytes((a.output / "decisions_sealed.json").read_bytes()) == seal
    write_json(a.output / "scored_decisions.json", scored)
    result = {"mode": a.mode, "n_roots": len(worlds), "types": {t: sum(w["type"] == t for w in worlds) for t in sorted({w["type"] for w in worlds})},
              "mechanism": mechanism_report(worlds, adapt),
              "method_table_cap192": method_table(scored, PRIMARY_CAP), "caps": list(CAPS), "primary_cap": PRIMARY_CAP,
              "method_tables_by_cap": {str(c): method_table(scored, c) for c in CAPS},
              "hypotheses": evaluate_hypotheses(scored, protocol), "decisions_sha256": seal,
              "root_held_out_predictive_coverage_95": coverage,
              "author_baselines": "included" if AUTHOR and "error" not in AUTHOR else f"skipped ({AUTHOR.get('error') if AUTHOR else 'not loaded'})"}
    write_json(a.output / "summary.json", result)
    import csv
    with (a.output / "seed_results.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["root", "type", "adaptation", "cap", "method", "selected", "hf_queries", "hf_episode_cost", "gain", "isr", "stop_reason", "queried"], extrasaction="ignore")
        wr.writeheader(); wr.writerows(scored)
    print(json.dumps({"hypotheses": result["hypotheses"]}, indent=2))
    for r in result["method_table_cap192"]:
        print(f"h={r['adaptation']:>2} {r['method']:18s} ISR={r['mean_isr']:7.3f} gain={r['mean_gain']:7.3f} cost={r['mean_hf_episode_cost']:6.1f} by_type={r['by_type']}")


if __name__ == "__main__":
    main()
