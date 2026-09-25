"""Discovery: does a LATENT strategic-response family make the best update world-specific?

World r: proxy opponent = Dirichlet-perturbed CFR+ incumbent (player 1), as in the frozen Kuhn line.
Candidates: alpha-mixtures of incumbent toward exact BR to the proxy opponent (player 0).
Deployment response (player 1) moves with weight w (short/long known) toward a TARGET that depends
on a latent type:
   exploiter    : target = exact BR to the deployed candidate           (frozen line's mechanism)
   equilibrator : target = CFR+ equilibrium strategy (self-correcting, candidate-independent)
Exact game values everywhere (no rollout noise) -> pure geometry check.
"""
import sys, json
import numpy as np
import pyspiel
import os
sys.path.insert(0, os.environ.get('OPENSPIEL_FROZEN_SOURCE', '/root/autodl-tmp/openspiel_frozen_source'))
from openspiel_kuhn_discovery import (_cfr_incumbent, _player_rows, _perturbed_opponent, _merge_players,
                                      _best_response_policy, _mix_player, _value)

GAME = sys.argv[1] if len(sys.argv) > 1 else "kuhn_poker"
ROOTS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
ALPHAS = np.linspace(0, 1, 9)
ETA = float(os.environ.get("OS_ETA", "0.25")); SHORT, LONG = 1, 8   # sweep ETA in discovery only
game = pyspiel.load_game(GAME)
inc = _cfr_incumbent(game, 2000 if GAME != "kuhn_poker" else 5000)
rows = _player_rows(inc)

def world(root, beta):
    rng = np.random.default_rng(root)
    opp = _perturbed_opponent(game, inc, rows, rng, beta=beta, concentration=0.35)
    joint_inc = _merge_players(game, inc, opp, rows)
    focal_br = _best_response_policy(game, joint_inc, 0)
    cands = [_mix_player(game, inc, focal_br, 0, a, rows) for a in ALPHAS]
    base = _value(game, inc, opp)
    proxy = np.array([_value(game, c, opp) - base for c in cands])
    out = {"proxy": proxy}
    for hname, steps in (("short", SHORT), ("long", LONG)):
        w = 1 - (1 - ETA) ** steps
        for t in ("exploiter", "equilibrator"):
            if t == "exploiter":
                inc_target = _best_response_policy(game, joint_inc, 1)
            else:
                inc_target = inc  # equilibrium strategy for player 1 rows
            inc_resp = _mix_player(game, opp, inc_target, 1, w, rows)
            dbase = _value(game, inc, inc_resp)
            dep = []
            for c in cands:
                if t == "exploiter":
                    target = _best_response_policy(game, _merge_players(game, c, opp, rows), 1)
                else:
                    target = inc
                resp = _mix_player(game, opp, target, 1, w, rows)
                dep.append(_value(game, c, resp) - dbase)
            out[(hname, t)] = np.array(dep)
    return out

res = []
for i in range(ROOTS):
    root = 90000 + i
    beta = float(np.random.default_rng(root + 314159).uniform(0.35, 0.95))
    res.append(world(root, beta))
    print(f"root {root} done", file=sys.stderr, flush=True)
np.set_printoptions(precision=4, suppress=True, linewidth=160)
for hname in ("short", "long"):
    print(f"\n===== {GAME} {hname} =====")
    for t in ("exploiter", "equilibrator"):
        D = np.array([r[(hname, t)] for r in res])
        best = D.argmax(1)
        print(f"{t:12s} mean dep by alpha: {D.mean(0)}  best hist: {np.bincount(best, minlength=9)}")
    P = np.array([r["proxy"] for r in res]); print(f"proxy        mean by alpha: {P.mean(0)}  best hist: {np.bincount(P.argmax(1), minlength=9)}")
    # mixed population: half/half latent type; population prior = mean over types
    Dmix = np.array([r[(hname, "exploiter" if i % 2 else "equilibrator")] for i, r in enumerate(res)])
    mu = Dmix.mean(0)
    v_pop = Dmix[:, mu.argmax()].mean(); v_oracle = Dmix.max(1).mean(); v_proxy = Dmix[np.arange(len(res)), P.argmax(1)].mean()
    print(f"MIXED population: prior-choice value {v_pop:.4f} | oracle {v_oracle:.4f} | proxy-choice {v_proxy:.4f} | value of world-specific info {v_oracle - v_pop:.4f} | pop-prior picks alpha={ALPHAS[mu.argmax()]}")
    # within-type ceilings
    for t in ("exploiter", "equilibrator"):
        D = np.array([r[(hname, t)] for r in res]); mu = D.mean(0)
        print(f"   {t:12s} within-type: prior-choice {D[:, mu.argmax()].mean():.4f} oracle {D.max(1).mean():.4f} -> world-specific info {D.max(1).mean()-D[:, mu.argmax()].mean():.4f}")
