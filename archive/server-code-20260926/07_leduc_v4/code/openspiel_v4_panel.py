"""OpenSpiel v4 panel generator: multi-direction candidates, per-hand sealed banks, noisy audits.

Extends the frozen v2 panel (latent strategic-response family, exact audit) so that the PIVOT
*components* become testable:

  candidates      : protocol list of {id, direction, alpha}. direction in
                    {"all", "rank<r>", "round1", "round2"}; the candidate mixes the incumbent toward
                    the exact best response to the proxy opponent ONLY on player-0 information sets
                    that belong to the direction. No incumbent copy is included (the incumbent is the
                    selectors' explicit no-update option).
  per-hand bank   : for each (adaptation, candidate) we store per-hand outcomes of n_max paired
                    common-random-number hands  ->  multi-fidelity queries (any n <= n_max) and
                    repeated queries with fresh hands; plus an INDEPENDENT incumbent stream for the
                    paired-vs-unpaired ablation.
  noisy audits    : two independent paired blocks A and B (n_audit hands each) so the posterior can be
                    calibrated from realistic noisy audits; the exact deployment delta is still stored
                    and is the only quantity used for scoring.

Output per root: seed_<root>/summary.json (analyze_v4-compatible fields + v4 fields) and bank.npz.
Selectors never see exact deployment values; analysis seals decisions before reading them.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import best_response, cfr, expected_game_score

TYPES = ("equilibrator", "exploiter")  # index = root % 2
TYPE_RNG_OFFSET = 271828
BANK_VERSION = "openspiel_v4_multidirection_bank"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ latent type (identical to v2 panel)
def latent_type(root: int, protocol: dict):
    rule = protocol.get("responder_type_rule", "exploiter_if_seed_odd_else_equilibrator")
    if rule == "exploiter_if_seed_odd_else_equilibrator":
        kind = TYPES[root % 2]
        return kind, float(kind == "exploiter")
    rng = np.random.default_rng(root + TYPE_RNG_OFFSET)
    if rule == "bernoulli":
        kind = "exploiter" if rng.uniform() < float(protocol["p_exploiter"]) else "equilibrator"
        return kind, float(kind == "exploiter")
    if rule == "continuous_mixture":
        m = float(rng.uniform())
        return ("mix_low" if m < 1 / 3 else "mix_mid" if m < 2 / 3 else "mix_high"), m
    raise ValueError(f"unknown responder_type_rule {rule!r}")


# ------------------------------------------------------------------ policy helpers (mirror frozen line)
def clone(game, src):
    out = policy.TabularPolicy(game)
    out.action_probability_array[:] = src.action_probability_array
    return out


def player_rows(tab):
    return np.asarray([s.current_player() for s in tab.states], dtype=int)


def merge(game, p0, p1, rows):
    j = clone(game, p0)
    j.action_probability_array[rows == 1] = p1.action_probability_array[rows == 1]
    return j


def mix_mask(game, left, right, mask, w):
    """Per-information-set mixture (1-w) left + w right on the boolean state mask."""
    m = clone(game, left)
    m.action_probability_array[mask] = (1 - w) * left.action_probability_array[mask] + w * right.action_probability_array[mask]
    return m


def mix(game, left, right, pid, w, rows):
    return mix_mask(game, left, right, rows == pid, w)


def best_resp(game, joint, pid):
    br = best_response.BestResponsePolicy(game, pid, joint)
    out = clone(game, joint)
    for i, s in enumerate(out.states):
        if s.current_player() != pid:
            continue
        probs = br.action_probabilities(s, pid)
        out.action_probability_array[i, :] = 0.0
        for a, p in probs.items():
            out.action_probability_array[i, int(a)] = float(p)
    return out


def value(game, p0, p1):
    return float(expected_game_score.policy_value(game.new_initial_state(), [p0, p1])[0])


def cfr_incumbent(game, iterations):
    solver = cfr.CFRPlusSolver(game)
    for _ in range(iterations):
        solver.evaluate_and_update_policy()
    return solver.average_policy()


def perturbed_opponent(game, incumbent, rows, rng, beta, concentration):
    opp = clone(game, incumbent)
    for i, s in enumerate(opp.states):
        if s.current_player() != 1:
            continue
        legal = np.flatnonzero(opp.legal_actions_mask[i])
        row = np.zeros(opp.action_probability_array.shape[1]); row[legal] = rng.dirichlet(np.full(len(legal), concentration))
        opp.action_probability_array[i] = (1 - beta) * incumbent.action_probability_array[i] + beta * row
    return opp


# ------------------------------------------------------------------ candidate directions
def state_round(game, state) -> int:
    s = game.new_initial_state(); chance = 0
    for a in state.history():
        if s.is_chance_node():
            chance += 1
        s.apply_action(a)
    return 1 if chance <= 2 else 2   # Leduc: two private deals, then one public card starts round 2


def direction_mask(game, tab, rows, direction: str, num_suits: int) -> np.ndarray:
    """Boolean mask over tabular states: player-0 information sets belonging to `direction`."""
    mask = np.zeros(len(tab.states), dtype=bool)
    for i, s in enumerate(tab.states):
        if rows[i] != 0:
            continue
        hist = s.history()
        if direction == "all":
            mask[i] = True
        elif direction.startswith("rank"):
            mask[i] = (int(hist[0]) // num_suits) == int(direction[4:])   # history[0] = player 0's private card
        elif direction in ("round1", "round2"):
            mask[i] = state_round(game, s) == int(direction[-1])
        else:
            raise ValueError(f"unknown direction {direction!r}")
    return mask


# ------------------------------------------------------------------ paired CRN sampler (game-agnostic)
def _rng(seed, key, hand, source):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=tuple(key) + (hand, source))))


def play_hand(game, p0, p1, seed, key, hand):
    state = game.new_initial_state()
    rngs = {"c": _rng(seed, key, hand, 0), 0: _rng(seed, key, hand, 1), 1: _rng(seed, key, hand, 2)}
    while not state.is_terminal():
        if state.is_chance_node():
            acts, probs = zip(*state.chance_outcomes())
            state.apply_action(int(rngs["c"].choice(acts, p=np.asarray(probs) / np.sum(probs))))
        else:
            pid = state.current_player()
            pol = p0 if pid == 0 else p1
            d = pol.action_probabilities(state)
            acts = list(d.keys()); probs = np.asarray([d[a] for a in acts], dtype=float); probs /= probs.sum()
            state.apply_action(int(rngs[pid].choice(acts, p=probs)))
    return float(state.returns()[0])


def hand_returns(game, p0, p1, seed, key, n) -> np.ndarray:
    return np.fromiter((play_hand(game, p0, p1, seed, key, h) for h in range(n)), dtype=np.float32, count=n)


# ------------------------------------------------------------------ one root
def run_root(root: int, protocol: dict, out_dir: Path) -> dict:
    t0 = time.monotonic()
    game = pyspiel.load_game(protocol["game"])
    num_suits = int(protocol["num_suits"])   # Leduc default deck: 3 ranks x 2 suits; Kuhn: 3 ranks x 1 suit. Not exposed by pyspiel.
    n_cards = len(game.new_initial_state().chance_outcomes())
    assert n_cards % num_suits == 0, (n_cards, num_suits)
    inc = cfr_incumbent(game, protocol["cfr_iterations"])
    rows = player_rows(inc)
    beta = float(np.random.default_rng(root + 314159).uniform(protocol["beta_min"], protocol["beta_max"]))
    opp = perturbed_opponent(game, inc, rows, np.random.default_rng(root), beta, protocol["dirichlet_concentration"])
    kind, m_expl = latent_type(root, protocol)
    joint_inc = merge(game, inc, opp, rows)
    focal_br = best_resp(game, joint_inc, 0)
    cand_specs = protocol["candidates"]
    masks = {c["direction"]: direction_mask(game, inc, rows, c["direction"], num_suits) for c in cand_specs}
    rank_masks = [direction_mask(game, inc, rows, f"rank{r}", num_suits) for r in range(n_cards // num_suits)]
    assert sum(int(m.sum()) for m in rank_masks) == int((rows == 0).sum()) and len({int(m.sum()) for m in rank_masks}) == 1, "rank masks must partition player-0 infosets evenly"
    cands = [mix_mask(game, inc, focal_br, masks[c["direction"]], float(c["alpha"])) for c in cand_specs]
    K = len(cands)
    base = value(game, inc, opp)
    proxy = {str(i): value(game, c, opp) - base for i, c in enumerate(cands)}
    eta = protocol["eta"]
    n_max = int(protocol["bank_hands"]); n_audit = int(protocol["audit_hands"])

    def responder_target(focal):
        if m_expl >= 1.0:
            return best_resp(game, merge(game, focal, opp, rows), 1)
        if m_expl <= 0.0:
            return inc
        return mix(game, inc, best_resp(game, merge(game, focal, opp, rows), 1), 1, m_expl, rows)

    inc_target = responder_target(inc)
    steps_list = [int(v) for v in protocol["adaptation_steps"].values()]
    H = len(steps_list)
    paired = np.zeros((H, K, n_max), dtype=np.float32)
    new_ret = np.zeros((H, K, n_max), dtype=np.float32)
    old_indep = np.zeros((H, K, n_max), dtype=np.float32)
    audit_A = np.zeros((H, K, n_audit), dtype=np.float32)
    audit_B = np.zeros((H, K, n_audit), dtype=np.float32)
    bank, labels, mech, weights, audit_noisy, unit_var = {}, {}, [], {}, {}, {}
    for hi, (hname, steps) in enumerate(protocol["adaptation_steps"].items()):
        steps = int(steps)
        w = 1 - (1 - eta) ** steps; weights[hname] = w
        inc_resp = mix(game, opp, inc_target, 1, w, rows)
        dbase = value(game, inc, inc_resp)
        bank[str(steps)], labels[str(steps)], audit_noisy[str(steps)], unit_var[str(steps)] = {}, {"incumbent": 0.0}, {}, {}
        for i, c in enumerate(cands):
            resp = mix(game, opp, responder_target(c), 1, w, rows)
            exact = value(game, c, resp) - dbase
            labels[str(steps)][str(i)] = exact
            key = (1, steps, i)
            u_new = hand_returns(game, c, resp, root, key, n_max)
            u_old = hand_returns(game, inc, inc_resp, root, key, n_max)
            u_old_i = hand_returns(game, inc, inc_resp, root, (4, steps, i), n_max)
            a_new = hand_returns(game, c, resp, root, (2, steps, i), n_audit); a_old = hand_returns(game, inc, inc_resp, root, (2, steps, i), n_audit)
            b_new = hand_returns(game, c, resp, root, (3, steps, i), n_audit); b_old = hand_returns(game, inc, inc_resp, root, (3, steps, i), n_audit)
            paired[hi, i] = u_new - u_old; new_ret[hi, i] = u_new; old_indep[hi, i] = u_old_i
            audit_A[hi, i] = a_new - a_old; audit_B[hi, i] = b_new - b_old
            bank[str(steps)][str(i)] = float(paired[hi, i].mean())
            audit_noisy[str(steps)][str(i)] = {"A": float(audit_A[hi, i].mean()), "B": float(audit_B[hi, i].mean())}
            unit_var[str(steps)][str(i)] = {"paired": float(paired[hi, i].astype(float).var(ddof=1)),
                                            "unpaired": float(u_new.astype(float).var(ddof=1) + u_old_i.astype(float).var(ddof=1))}
            own_gain = float(-(value(game, c, resp) - value(game, c, opp)))
            for b in ("A", "B"):   # exact audit kept in the v2 schema so analyze_v4 can cross-check
                mech.append({"candidate": str(i), "probability": float(cand_specs[i]["alpha"]), "adaptation": steps, "block": b,
                             "proxy_delta_audit": proxy[str(i)], "deployment_delta_audit": exact, "gap": exact - proxy[str(i)],
                             "responder_own_gain": own_gain})
    summary = {"status": "complete", "version": BANK_VERSION, "seed": root, "game": protocol["game"],
               "responder_type": kind, "responder_mixture_m": m_expl, "beta": beta, "response_weights": weights,
               "candidates": cand_specs, "proxy_deltas": proxy, "selection_bank": bank, "audit_gains": labels,
               "audit_noisy": audit_noisy, "unit_variance": unit_var, "mechanism_audit": mech, "response_probabilities": None,
               "audit_is_exact": True, "bank_hands": n_max, "audit_hands": n_audit, "query_hands": n_max,
               "adaptation_steps": {k: int(v) for k, v in protocol["adaptation_steps"].items()},
               "mask_sizes": {d: int(m.sum()) for d, m in masks.items()}, "duration_seconds": time.monotonic() - t0}
    d = out_dir / f"seed_{root}"; d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "bank.npz", steps=np.asarray(steps_list), paired_diff=paired, new_returns=new_ret,
                        old_indep=old_indep, audit_A=audit_A, audit_B=audit_B)
    summary["bank_sha256"] = sha(d / "bank.npz")
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    (d / "status.json").write_text(json.dumps({"status": "complete"}))
    return {"root": root, "type": kind, "seconds": round(summary["duration_seconds"], 1), "mask_sizes": summary["mask_sizes"]}


SELFTEST_PROTOCOL = {
    "protocol_id": "selftest", "status": "SELFTEST", "seeds": [1, 2], "game": "kuhn_poker", "cfr_iterations": 20,
    "beta_min": 0.35, "beta_max": 0.95, "dirichlet_concentration": 0.35, "eta": 0.15, "num_suits": 1,
    "adaptation_steps": {"short": 1, "long": 8}, "bank_hands": 64, "audit_hands": 32,
    "candidates": [{"id": "all_a1.0", "direction": "all", "alpha": 1.0}, {"id": "all_a0.5", "direction": "all", "alpha": 0.5},
                   {"id": "rank0_a1.0", "direction": "rank0", "alpha": 1.0}, {"id": "rank2_a1.0", "direction": "rank2", "alpha": 1.0}],
    "responder_type_rule": "exploiter_if_seed_odd_else_equilibrator",
}


def selftest(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps(SELFTEST_PROTOCOL, indent=1))
    for r in SELFTEST_PROTOCOL["seeds"]:
        print(json.dumps(run_root(r, SELFTEST_PROTOCOL, out)), flush=True)
    for r in SELFTEST_PROTOCOL["seeds"]:
        s = json.loads((out / f"seed_{r}" / "summary.json").read_text())
        z = np.load(out / f"seed_{r}" / "bank.npz")
        K = len(s["candidates"])
        assert z["paired_diff"].shape == (2, K, 64) and z["audit_A"].shape == (2, K, 32)
        for hi, steps in enumerate(z["steps"]):
            for i in range(K):
                assert abs(z["paired_diff"][hi, i].mean() - s["selection_bank"][str(steps)][str(i)]) < 1e-6
                assert np.isfinite(s["audit_gains"][str(steps)][str(i)])
        assert s["responder_type"] == TYPES[r % 2]
        assert all(np.isfinite(v) for v in s["proxy_deltas"].values())
        # proxy deltas of 'all' candidates are non-negative (mixing toward the exact best response cannot hurt at alpha=1)
        assert s["proxy_deltas"]["0"] >= -1e-9, s["proxy_deltas"]
    print("SELFTEST_OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.output)
        return
    protocol = json.loads(a.protocol.read_text())
    assert protocol["status"] in ("FROZEN", "PILOT_FROZEN"), protocol["status"]
    a.output.mkdir(parents=True, exist_ok=a.resume)
    (a.output / "protocol.json").write_text(json.dumps({**protocol, "protocol_sha256": sha(a.protocol),
                                                        "script_sha256": sha(Path(__file__)),
                                                        "open_spiel": getattr(pyspiel, "__version__", "?")}, indent=1))
    todo = [r for r in protocol["seeds"] if not (a.resume and (a.output / f"seed_{r}" / "status.json").exists())]
    if a.workers <= 1:
        for r in todo:
            print(json.dumps(run_root(r, protocol, a.output)), flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
            for res in pool.map(run_root, todo, [protocol] * len(todo), [a.output] * len(todo)):
                print(json.dumps(res), flush=True)
    (a.output / "status.json").write_text(json.dumps({"status": "complete", "roots": len(protocol["seeds"])}))


if __name__ == "__main__":
    main()
