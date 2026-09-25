"""OpenSpiel v2 panel generator: latent strategic-response family, exact audit.

Per root (one world):
  proxy opponent (player 1)  : Dirichlet-perturbed CFR+ incumbent, beta ~ U(beta_min, beta_max) from root
  candidates (player 0)      : alpha-mixtures incumbent -> exact BR(proxy opponent), alpha grid from protocol
  latent type (root parity)  : exploiter    -> deployment opponent moves w toward exact BR(candidate)
                               equilibrator -> deployment opponent moves w toward CFR+ equilibrium
  adaptation (known cond.)   : short/long = response weight w = 1-(1-eta)^T for T in protocol
  proxy delta                : EXACT value(candidate vs proxy opp) - value(incumbent vs proxy opp)
  HF selection observation S : paired common-random-number rollouts, N hands, of
                               [candidate vs its responder] - [incumbent vs incumbent's responder]
  audit label                : EXACT deployment delta (so A == B == exact; ISR is noise-free)

Output per root: seed_<root>/summary.json in the SAME schema analyze_v4.py consumes
(proxy_deltas, selection_bank, audit_gains, mechanism_audit, responder_type, response_probabilities=None).
Selectors never see the exact deployment values; analyze_v4 seals decisions before reading them.
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
TYPE_RNG_OFFSET = 271828  # latent-type draw stream; disjoint from beta (root+314159) and Dirichlet (root)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latent_type(root: int, protocol: dict):
    """Return (kind_label, mixture_m). mixture_m is the weight on the exploiter target (1 = pure exploiter).

    responder_type_rule (protocol field; legacy default = parity):
      "exploiter_if_seed_odd_else_equilibrator" : kind = TYPES[root % 2], m in {0, 1}
      "bernoulli"                               : exploiter with prob p_exploiter (root-seeded), m in {0, 1}
      "continuous_mixture"                      : m ~ U(0,1) root-seeded; target = (1-m) equilibrium + m BR;
                                                  kind label binned mix_low/mix_mid/mix_high for reporting only
    Selectors never see kind or m.
    """
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


def mix(game, left, right, pid, w, rows):
    m = clone(game, left)
    mask = rows == pid
    m.action_probability_array[mask] = (1 - w) * left.action_probability_array[mask] + w * right.action_probability_array[mask]
    return m


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


# ------------------------------------------------------------------ paired CRN sampler (game-agnostic)
def _rng(seed, key, hand, source):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=tuple(key) + (hand, source))))


def play_hand(game, p0, p1, seed, key, hand):
    """One hand; chance and each player's action draws come from separate streams shared across pairings."""
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


def paired_mean(game, new_pair, old_pair, seed, key, n):
    diffs = [play_hand(game, *new_pair, seed, key, h) - play_hand(game, *old_pair, seed, key, h) for h in range(n)]
    return float(np.mean(diffs)), float(np.var(diffs, ddof=1) / n)


# ------------------------------------------------------------------ one root
def run_root(root: int, protocol: dict, out_dir: Path) -> dict:
    t0 = time.monotonic()
    game = pyspiel.load_game(protocol["game"])
    inc = cfr_incumbent(game, protocol["cfr_iterations"])
    rows = player_rows(inc)
    beta = float(np.random.default_rng(root + 314159).uniform(protocol["beta_min"], protocol["beta_max"]))
    opp = perturbed_opponent(game, inc, rows, np.random.default_rng(root), beta, protocol["dirichlet_concentration"])
    kind, m_expl = latent_type(root, protocol)
    alphas = protocol["candidate_alphas"]
    joint_inc = merge(game, inc, opp, rows)
    focal_br = best_resp(game, joint_inc, 0)
    cands = [mix(game, inc, focal_br, 0, a, rows) for a in alphas]
    base = value(game, inc, opp)
    proxy = {str(i): value(game, c, opp) - base for i, c in enumerate(cands)}
    eta = protocol["eta"]
    bank, labels, mech, weights = {}, {}, [], {}

    def responder_target(focal):
        """Deployment opponent's target: exact BR(focal) for exploiter, equilibrium for equilibrator,
        and their m-mixture for continuous types. m in {0,1} reproduces the legacy two-type code path exactly."""
        if m_expl >= 1.0:
            return best_resp(game, merge(game, focal, opp, rows), 1)
        if m_expl <= 0.0:
            return inc
        return mix(game, inc, best_resp(game, merge(game, focal, opp, rows), 1), 1, m_expl, rows)

    inc_target = responder_target(inc)
    for hname, steps in protocol["adaptation_steps"].items():
        w = 1 - (1 - eta) ** steps; weights[hname] = w
        inc_resp = mix(game, opp, inc_target, 1, w, rows)
        dbase = value(game, inc, inc_resp)
        bank[hname], labels[hname] = {}, {"incumbent": 0.0}
        for i, c in enumerate(cands):
            target = responder_target(c)
            resp = mix(game, opp, target, 1, w, rows)
            exact = value(game, c, resp) - dbase
            labels[hname][str(i)] = exact
            s, s_var = paired_mean(game, (c, resp), (inc, inc_resp), root, (1, int(steps), i), protocol["query_hands"])
            bank[hname][str(i)] = s
            for b in ("A", "B"):
                mech.append({"candidate": str(i), "probability": float(alphas[i]), "adaptation": int(steps), "block": b,
                             "proxy_delta_audit": proxy[str(i)], "deployment_delta_audit": exact, "gap": exact - proxy[str(i)],
                             "responder_own_gain": float(-(value(game, c, resp) - value(game, c, opp)))})
    # analyze_v4 keys adaptations by integer steps; rename bank/labels accordingly
    bank_i = {str(protocol["adaptation_steps"][h]): v for h, v in bank.items()}
    labels_i = {str(protocol["adaptation_steps"][h]): v for h, v in labels.items()}
    summary = {"status": "complete", "version": "openspiel_v2_latent_family_panel", "seed": root, "game": protocol["game"],
               "responder_type": kind, "responder_mixture_m": m_expl, "beta": beta, "response_weights": weights, "proxy_deltas": proxy,
               "selection_bank": bank_i, "audit_gains": labels_i, "mechanism_audit": mech, "response_probabilities": None,
               "audit_is_exact": True, "query_hands": protocol["query_hands"], "duration_seconds": time.monotonic() - t0}
    d = out_dir / f"seed_{root}"; d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    (d / "status.json").write_text(json.dumps({"status": "complete"}))
    return {"root": root, "type": kind, "seconds": summary["duration_seconds"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    protocol = json.loads(a.protocol.read_text())
    assert protocol["status"] in ("FROZEN", "PILOT_FROZEN", "DISCOVERY"), protocol["status"]
    a.output.mkdir(parents=True, exist_ok=a.resume)
    (a.output / "protocol.json").write_text(json.dumps({**protocol, "protocol_sha256": sha(a.protocol),
                                                        "script_sha256": sha(Path(__file__)), "open_spiel": getattr(pyspiel, "__version__", "?")}, indent=1))
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
