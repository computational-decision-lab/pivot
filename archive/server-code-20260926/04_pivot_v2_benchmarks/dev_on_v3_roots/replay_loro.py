"""Leave-one-root-out development replay of PIVOT v2 on the 30 formal roots.

For held-out root r: fit the v2 posterior hyper-parameters on the other 29 roots'
(proxy, S, A, B) panels, instantiate the prior for r from r's own proxies, let each
selector query r's real selection observations S, then score with r's audit label.
Zero new native episodes. Historical frozen-v1 decisions are read from the archive
for side-by-side comparison (they were produced by the same roots).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

from dev_data import ADAPT, GRID, load_all, check_against_archive
from pivot_v2 import METHODS_V2, fit_posterior_v2, make_posterior, run_selector

CAPS = (96, 192, 384)
HIST = ("proxy_only", "calibrated_no_hf", "pivot_sequential", "pivot_sequential_adaptive_stop",
        "uniform_random_matched", "global_ivr_matched", "posterior_lucb_matched",
        "author_random_hf", "author_paired_lucb", "author_global_voi", "author_pivot_voi", "all_hf_reference")


def fit_on(train, h):
    g = np.array([w["per_h"][h]["label"] - w["proxy"] for w in train])
    ab = np.array([w["per_h"][h]["A"] - w["per_h"][h]["B"] for w in train])
    sab = np.array([(w["per_h"][h]["S"] - w["per_h"][h]["A"]) * (w["per_h"][h]["S"] - w["per_h"][h]["B"]) for w in train])
    return fit_posterior_v2(g, ab, sab, distances=np.abs(GRID - .5))


def replay(data, seed_offset=0, verbose=False):
    rows = []
    for w in data:
        train = [x for x in data if x["root"] != w["root"]]
        for h in ADAPT:
            spec = fit_on(train, h)
            ph = w["per_h"][h]
            label = np.concatenate([ph["label"], [0.0]])
            best = label.max()
            cost = ph["cost"]
            costs = np.full(8, cost)
            for cap in CAPS:
                budget = min(8, max(0, (cap - h) // int(cost)))
                for m in METHODS_V2:
                    if m in ("no_hf_v2", "proxy_only") and cap != CAPS[0]:
                        continue
                    post = make_posterior(spec, w["proxy"])
                    res = run_selector(post, w["proxy"], lambda j: ph["S"][j], method=m, budget=budget,
                                       costs=costs, seed=w["root"] + seed_offset)
                    setup = h if res["hf_queries"] else 0
                    rows.append({"root": w["root"], "adaptation": h, "cap": cap if m not in ("no_hf_v2", "proxy_only") else None,
                                 "method": m, "selected": res["selected"], "queried": res["queried"],
                                 "hf_episode_cost": res["charged_cost"] + setup, "gain": float(label[res["selected"]]),
                                 "isr": float(best - label[res["selected"]]), "stop_reason": res["stop_reason"]})
            # historical frozen-v1 decisions (cap-specific), for reference
            for d in w["historical_scored"]:
                if d["adaptation"] != h:
                    continue
                rows.append({"root": w["root"], "adaptation": h, "cap": d.get("budget_cap"), "method": d["method"],
                             "selected": d["selected_id"], "queried": d.get("queried_ids"), "hf_episode_cost": d["hf_episode_cost"],
                             "gain": d["audit_gain"], "isr": d["noisy_audit_isr"], "stop_reason": d.get("stop_reason")})
    return rows


def bootstrap_ci(x, draws=10000, seed=20260917):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(draws, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def table(rows, cap=192):
    by = defaultdict(list)
    for r in rows:
        if r["cap"] == cap or r["cap"] is None:
            by[(r["adaptation"], r["method"])].append(r)
    out = {}
    for (h, m), rs in sorted(by.items(), key=lambda kv: (kv[0][0], np.mean([r["isr"] for r in kv[1]]))):
        out[(h, m)] = (np.mean([r["isr"] for r in rs]), np.mean([r["gain"] for r in rs]),
                       np.mean([r["hf_episode_cost"] for r in rs]), len(rs))
    return out


def paired(rows, m1, m2, h, cap=192):
    a = {r["root"]: r["gain"] for r in rows if r["method"] == m1 and r["adaptation"] == h and (r["cap"] == cap or r["cap"] is None)}
    b = {r["root"]: r["gain"] for r in rows if r["method"] == m2 and r["adaptation"] == h and (r["cap"] == cap or r["cap"] is None)}
    roots = sorted(set(a) & set(b))
    return bootstrap_ci([a[r] - b[r] for r in roots]), roots


if __name__ == "__main__":
    data = load_all()
    check_against_archive(data)
    rows = replay(data)
    json.dump(rows, open("loro_rows.json", "w"), default=float)
    for cap in CAPS:
        print(f"\n===== cap {cap} =====")
        t = table(rows, cap)
        for (h, m), (isr, gain, cost, n) in t.items():
            print(f"h={h:>2} {m:32s} ISR={isr:7.3f} gain={gain:7.3f} cost={cost:6.1f} n={n}")
    print("\n===== paired contrasts (gain m1 - m2), cap 192, root bootstrap 95% =====")
    for h in ADAPT:
        for m1, m2 in [("pivot_kg", "uniform_v2"), ("pivot_kg", "ivr_v2"), ("pivot_kg", "no_hf_v2"), ("pivot_kg", "pivot_sequential"),
                       ("pivot_kg", "global_ivr_matched"), ("pivot_kg_stop", "pivot_kg"), ("ivr_v2", "uniform_v2"), ("no_hf_v2", "proxy_only"),
                       ("uniform_v2", "uniform_random_matched")]:
            (mean, lo, hi), roots = paired(rows, m1, m2, h)
            print(f"h={h:>2} {m1:>18} - {m2:<26} {mean:+7.3f} [{lo:+7.3f}, {hi:+7.3f}] n={len(roots)}")
    # interaction long-short for pivot_kg vs uniform_v2
    dl, _ = paired(rows, "pivot_kg", "uniform_v2", 32); ds, _ = paired(rows, "pivot_kg", "uniform_v2", 4)
    a = {r["root"]: r["gain"] for r in rows if r["method"] == "pivot_kg" and r["cap"] == 192}
    diff = []
    for root in sorted({r["root"] for r in rows}):
        g = lambda m, h: next(r["gain"] for r in rows if r["root"] == root and r["method"] == m and r["adaptation"] == h and r["cap"] == 192)
        diff.append((g("pivot_kg", 32) - g("uniform_v2", 32)) - (g("pivot_kg", 4) - g("uniform_v2", 4)))
    print("interaction (long-short) pivot_kg - uniform_v2:", bootstrap_ci(diff))
