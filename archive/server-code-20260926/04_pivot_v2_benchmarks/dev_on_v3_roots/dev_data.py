"""Load the 30 formal Melting Pot roots (44000-44029) into a development dataset.

Per root r and adaptation h we expose:
  proxy[c]        proxy delta actually observed by the methods (features.json)
  S[c]            the HF selection observation the methods could query (selection block)
  A[c], B[c]      two independent 64-episode audit deployment deltas (labels; never shown to selectors)
  cost            h + 32 per query; incumbent setup h charged once if any query
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GRID = np.array([.125, .225, .325, .425, .575, .675, .775, .875])
COMB = ((0, 0), (0, 1), (1, 0), (1, 1))
import os
FORMAL = Path(os.environ.get("MELTING_V3_ARCHIVE_DIR", "/tmp/iclr_read/formal/melting_method_confirm_20260917"))
ROOTS = list(range(44000, 44030))
ADAPT = (4, 32)


def weighted(block, p, q):
    w = {f"{a}{b}": (p if a else 1 - p) * (q if b else 1 - q) for a, b in COMB}
    return sum(w[s] * block["conditional_return_means"][s]["focal_return"] for s in w)


def load_root(root: int, base: Path = FORMAL) -> dict:
    d = base / f"seed_{root}"
    feats = json.load(open(d / "features.json"))
    q = json.load(open(d / "training_seal.json"))["response_probabilities"]
    sel = {i: json.load(open(d / f"selection/{i}.json")) for i in range(8)}
    aud = {b: json.load(open(d / f"audit/{b}.json")) for b in "AB"}
    meta = json.load(open(d / "posterior.json"))
    hist = json.load(open(d / "decisions_frozen.json"))
    scored = json.load(open(d / "scored_decisions.json"))
    out = {"root": root, "proxy": np.array([r["proxy_delta"] for r in feats]), "frozen_posterior": meta,
           "historical": hist, "historical_scored": scored, "per_h": {}}
    for h in ADAPT:
        hs = str(h)
        S = np.array([weighted(sel[i], GRID[i], q[str(i)][hs]) - weighted(sel[i], .5, q["incumbent"][hs]) for i in range(8)])
        A = np.array([weighted(aud["A"], GRID[i], q[str(i)][hs]) - weighted(aud["A"], .5, q["incumbent"][hs]) for i in range(8)])
        B = np.array([weighted(aud["B"], GRID[i], q[str(i)][hs]) - weighted(aud["B"], .5, q["incumbent"][hs]) for i in range(8)])
        # frozen-response proxy measured inside audit blocks (for mechanism only)
        PA = np.array([weighted(aud["A"], GRID[i], .5) - weighted(aud["A"], .5, .5) for i in range(8)])
        PB = np.array([weighted(aud["B"], GRID[i], .5) - weighted(aud["B"], .5, .5) for i in range(8)])
        out["per_h"][h] = {"S": S, "A": A, "B": B, "PA": PA, "PB": PB, "label": 0.5 * (A + B), "cost": float(h + 32),
                           "q": {k: v[hs] for k, v in q.items()}}
    return out


def load_all(roots=ROOTS) -> list[dict]:
    return [load_root(r) for r in roots]


def check_against_archive(data: list[dict]) -> None:
    """Sanity: S must equal the historical all_hf_reference bank and label must equal audit_gains."""
    for w in data:
        summ = json.load(open(FORMAL / f"seed_{w['root']}/summary.json"))
        for h in ADAPT:
            ref = [x for x in w["historical"] if x["method"] == "all_hf_reference" and x["adaptation"] == h][0]["observed"]
            assert np.allclose(w["per_h"][h]["S"], [ref[str(i)] for i in range(8)], atol=1e-9)
            gains = summ["audit_gains"][str(h)]
            assert np.allclose(w["per_h"][h]["label"], [gains[str(i)] for i in range(8)], atol=1e-9)
