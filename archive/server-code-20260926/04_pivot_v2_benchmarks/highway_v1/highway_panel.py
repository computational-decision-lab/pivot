"""HighwayEnv reactive-traffic panel with a latent driver-population type per world.

Benchmark leg: "reactive physical environment" (Ledo priority 2; MetaDrive -> HighwayEnv fallback rule).

World r (root seed): straight 4-lane highway with N traffic vehicles driven by IDM+MOBIL whose parameters
come from a LATENT type (e.g. yielding vs assertive) fixed per root and hidden from all selectors.

Incumbent / candidates: a parametric ego controller (IDM+MOBIL) indexed by aggressiveness a on a fixed
grid (incumbent a=0.5). Fixed operator; no outcome-dependent proposal.

Proxy world : REPLAYED traffic. Trajectories are recorded while the incumbent drives (per episode seed)
              and replayed kinematically; they do not react to the ego.
Deployment  : REACTIVE traffic. A fraction phi of vehicles run their typed IDM/MOBIL live and react to the
              ego; the rest replay. phi is the response strength (short = weak, long = strong).
Return      : sum over policy steps of speed_reward - crash_penalty; episode ends at ego crash or horizon.

Per-seed outputs follow the same contract as melting_native_v4_hetero / openspiel_v2_panel and are consumed
unchanged by analyze_v4.py: proxy_deltas, selection_bank[phi_key][c] (sealed, no decisions), mechanism_audit
rows (blocks A/B), responder_type, status complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle

VERSION = "highway_v1_latent_driver_type_2026-09-18"


# ----------------------------------------------------------------------------- vehicles
class TypedIDM(IDMVehicle):
    """Traffic vehicle whose IDM/MOBIL class parameters come from the latent population type."""

    @classmethod
    def make_class(cls, params: dict):
        attrs = {k.upper(): float(v) for k, v in params.items() if k != "name"}
        return type("TypedIDM_" + str(params.get("name", "x")), (cls,), attrs)


class EgoIDM(IDMVehicle):
    """Parametric ego controller: aggressiveness a in [0,1] linearly maps to IDM/MOBIL parameters."""

    TARGET_SPEED_A = 30.0

    @classmethod
    def make_class(cls, a: float, spec: dict):
        def lerp(key):
            lo, hi = spec[key]; return float(lo + (hi - lo) * a)
        attrs = {"TARGET_SPEED_A": lerp("target_speed"), "TIME_WANTED": lerp("time_wanted"), "POLITENESS": lerp("politeness"),
                 "LANE_CHANGE_MIN_ACC_GAIN": lerp("lc_min_gain"), "LANE_CHANGE_MAX_BRAKING_IMPOSED": lerp("lc_max_braking"),
                 "COMFORT_ACC_MAX": lerp("comfort_acc"), "DISTANCE_WANTED": lerp("distance_wanted")}
        return type(f"EgoIDM_{a:.3f}".replace(".", "_"), (cls,), attrs)


class ReplayVehicle(Vehicle):
    """Kinematic vehicle replaying a recorded trajectory; never reacts. Ego can still crash into it."""

    def __init__(self, road, traj: np.ndarray):
        super().__init__(road, traj[0, :2].copy(), float(traj[0, 2]), float(traj[0, 3]))
        self.traj = traj; self.k = 0
        self.check_collisions = False

    def act(self, action=None):
        return

    def step(self, dt: float):
        self.k = min(self.k + 1, len(self.traj) - 1)
        r = self.traj[self.k]
        self.position = r[:2].copy(); self.heading = float(r[2]); self.speed = float(r[3])
        self.impact = None
        self.on_state_update()


# ----------------------------------------------------------------------------- world
def stable_u(*parts) -> float:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def build_road(cfg: dict, ep_seed: int, traffic_cls, ego_cls):
    rng = np.random.RandomState(ep_seed)
    net = RoadNetwork.straight_road_network(lanes=cfg["lanes"], length=cfg["road_length"], speed_limit=cfg["speed_limit"])
    road = Road(network=net, np_random=rng, record_history=False)
    for _ in range(cfg["n_behind_ego"]):
        road.vehicles.append(traffic_cls.create_random(road, spacing=cfg["spacing"]))
    ego = ego_cls.create_random(road, speed=cfg["ego_speed0"], lane_id=cfg["ego_lane"], spacing=cfg["spacing"])
    ego.target_speed = ego_cls.TARGET_SPEED_A
    road.vehicles.append(ego)
    for _ in range(cfg["n_vehicles"] - cfg["n_behind_ego"]):
        road.vehicles.append(traffic_cls.create_random(road, spacing=cfg["spacing"]))
    for v in road.vehicles:
        if v is not ego:
            v.target_speed = float(rng.uniform(*cfg["traffic_target_speed"]))
    return road, ego


def run_episode(cfg: dict, ep_seed: int, traffic_cls, ego_cls, *, record: bool = False,
                replay: np.ndarray | None = None, reactive_mask: np.ndarray | None = None):
    """-> (ego_return, ego_crashed, mean speed of LIVE traffic, recording (N,T+1,4) or None)."""
    road, ego = build_road(cfg, ep_seed, traffic_cls, ego_cls)
    traffic = [v for v in road.vehicles if v is not ego]
    sim_dt = 1.0 / cfg["sim_hz"]; per_policy = int(round(cfg["sim_hz"] / cfg["policy_hz"])); T = int(cfg["duration_s"] * cfg["sim_hz"])
    if replay is not None:
        new = [v if (reactive_mask is not None and reactive_mask[i]) else ReplayVehicle(road, replay[i]) for i, v in enumerate(traffic)]
        road.vehicles = [ego] + new; traffic = new
    rec = np.zeros((len(traffic), T + 1, 4)) if record else None
    if record:
        for i, v in enumerate(traffic):
            rec[i, 0] = [v.position[0], v.position[1], v.heading, v.speed]
    ret, crashed, speeds = 0.0, False, []
    lo, hi = cfg["speed_reward_range"]
    for k in range(1, T + 1):
        road.act(); road.step(sim_dt)
        if record:
            for i, v in enumerate(traffic):
                rec[i, k] = [v.position[0], v.position[1], v.heading, v.speed]
        if k % per_policy == 0:
            live = [v.speed for v in traffic if not isinstance(v, ReplayVehicle)]
            if live:
                speeds.append(float(np.mean(live)))
            if not crashed:
                ret += cfg["speed_reward"] * float(np.clip((ego.speed - lo) / (hi - lo), 0.0, 1.0))
                if ego.crashed:
                    crashed = True; ret -= cfg["crash_penalty"]
                    if not record:
                        break
    return ret, crashed, (float(np.mean(speeds)) if speeds else float("nan")), rec


# ----------------------------------------------------------------------------- panel
def latent_type(protocol: dict, seed: int) -> str:
    names = sorted(protocol["traffic_types"]); assert len(names) == 2, names
    rule = protocol.get("responder_type_rule", "parity")
    if rule == "parity":
        return names[seed % 2]
    if rule == "bernoulli":
        return names[0] if stable_u("type", seed) < float(protocol["p_first_type"]) else names[1]
    raise ValueError(rule)


def write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n")


def run_panel(protocol: dict, protocol_path: Path, seed: int, out: Path, max_seconds: int):
    t0 = time.monotonic()
    cfg = protocol["sim"]; grid = [float(x) for x in protocol["candidate_alphas"]]; inc = float(protocol["incumbent_alpha"])
    kind = latent_type(protocol, seed)
    traffic_cls = TypedIDM.make_class({**protocol["traffic_types"][kind], "name": kind})
    ego_inc = EgoIDM.make_class(inc, protocol["ego_spec"])
    ego_of = {str(i): EgoIDM.make_class(al, protocol["ego_spec"]) for i, al in enumerate(grid)}
    phis = {str(k): float(v) for k, v in protocol["response_levels"].items()}     # e.g. {"25": .25, "100": 1.0}
    n_rec, n_q, n_a = int(protocol["n_proxy_episodes"]), int(protocol["n_query_pairs"]), int(protocol["n_audit_pairs"])
    counter = {"episodes": 0}

    def deadline():
        if time.monotonic() - t0 > max_seconds:
            raise TimeoutError(f"max_seconds {max_seconds} exceeded")

    def ep(cls, tag, j, *, record=False, replay=None, mask=None):
        deadline(); counter["episodes"] += 1
        return run_episode(cfg, int(stable_u("ep", seed, tag, j) * 2 ** 31), traffic_cls, cls, record=record, replay=replay, reactive_mask=mask)

    # ---- 1. record traffic around the incumbent for every episode seed used anywhere
    write(out / "status.json", {"status": "recording", "responder_type": kind})
    tags = [("proxy", j) for j in range(n_rec)] + [(f"q{p}", j) for p in phis for j in range(n_q)] + [(f"a{b}", j) for b in "AB" for j in range(n_a)]
    recordings = {}
    for tag, j in tags:
        recordings[(tag, j)] = ep(ego_inc, tag, j, record=True)[3]

    def proxy_ret(cls, tag, j):
        rec = recordings[(tag, j)]
        return ep(cls, tag, j, replay=rec, mask=np.zeros(len(rec), bool))[0]

    # ---- 2. proxy deltas: replayed traffic
    write(out / "status.json", {"status": "proxy", "responder_type": kind})
    inc_proxy = float(np.mean([proxy_ret(ego_inc, "proxy", j) for j in range(n_rec)]))
    proxies = {i: float(np.mean([proxy_ret(c, "proxy", j) for j in range(n_rec)])) - inc_proxy for i, c in ego_of.items()}
    write(out / "features.json", [{"id": i, "alpha": grid[int(i)], "proxy_delta": proxies[i]} for i in ego_of])

    # ---- deployment helpers with incumbent caching (paired CRN: same episode seed, same reactive mask)
    inc_cache, inc_proxy_cache = {}, {}

    def mask_for(tag, j, phi):
        n = len(recordings[(tag, j)])
        return np.array([stable_u("react", seed, tag, j, i) < phi for i in range(n)])

    def deploy_delta(cls, tag, j, phi):
        rec, m = recordings[(tag, j)], mask_for(tag, j, phi)
        key = (tag, j, phi)
        if key not in inc_cache:
            r, _, sp, _ = ep(ego_inc, tag, j, replay=rec, mask=m); inc_cache[key] = (r, sp)
        r_new, _, sp_new, _ = ep(cls, tag, j, replay=rec, mask=m)
        r_old, sp_old = inc_cache[key]
        own = (sp_new - sp_old) if (np.isfinite(sp_new) and np.isfinite(sp_old)) else 0.0
        return r_new - r_old, own

    def proxy_delta_same_seed(cls, tag, j):
        if (tag, j) not in inc_proxy_cache:
            inc_proxy_cache[(tag, j)] = proxy_ret(ego_inc, tag, j)
        return proxy_ret(cls, tag, j) - inc_proxy_cache[(tag, j)]

    # ---- 3. sealed selection bank (HF observations only; no selector runs here)
    write(out / "status.json", {"status": "selection_bank", "responder_type": kind})
    bank = {hs: {i: float(np.mean([deploy_delta(c, f"q{hs}", j, phi)[0] for j in range(n_q)])) for i, c in ego_of.items()}
            for hs, phi in phis.items()}
    write(out / "decisions_frozen.json", {"version": VERSION, "kind": "sealed_selection_bank_not_decisions", "selection_bank": bank, "proxy_deltas": proxies})
    write(out / "selection_seal.json", {"decisions_sha256": hashlib.sha256((out / "decisions_frozen.json").read_bytes()).hexdigest(), "audit_generated": False})

    # ---- 4. held-out audit on independent episode seeds, blocks A / B
    write(out / "status.json", {"status": "held_out_audit", "responder_type": kind})
    mechanism, labels = [], {}
    proxy_audit = {(i, b): float(np.mean([proxy_delta_same_seed(c, f"a{b}", j) for j in range(n_a)])) for i, c in ego_of.items() for b in "AB"}
    for hs, phi in phis.items():
        labels[hs] = {"incumbent": 0.0}
        for i, c in ego_of.items():
            per_block = []
            for b in "AB":
                dd_own = [deploy_delta(c, f"a{b}", j, phi) for j in range(n_a)]
                dep, own, prox = float(np.mean([d for d, _ in dd_own])), float(np.mean([o for _, o in dd_own])), proxy_audit[(i, b)]
                per_block.append(dep)
                mechanism.append({"candidate": i, "alpha": grid[int(i)], "adaptation": int(hs), "block": b,
                                  "proxy_delta_audit": prox, "deployment_delta_audit": dep, "gap": dep - prox, "responder_own_gain": own})
            labels[hs][i] = float(np.mean(per_block))
    summary = {"status": "complete", "version": VERSION, "seed": seed, "responder_type": kind,
               "duration_seconds": time.monotonic() - t0, "episodes_simulated": counter["episodes"],
               "source_protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
               "proxy_deltas": proxies, "selection_bank": bank, "audit_gains": labels, "mechanism_audit": mechanism,
               "incumbent_proxy_return": inc_proxy,
               "scope": "Single-round reactive-traffic ISR; eight fixed ego-aggressiveness edits; latent driver population type is a world property."}
    write(out / "summary.json", summary)
    write(out / "status.json", {"status": "complete", "responder_type": kind})
    hl = max(phis, key=lambda k: phis[k])
    print(json.dumps({"seed": seed, "type": kind, "proxy": {k: round(v, 3) for k, v in proxies.items()},
                      "labels_long": {k: round(v, 3) for k, v in labels[hl].items()}, "episodes": counter["episodes"],
                      "seconds": round(time.monotonic() - t0, 1)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-seconds", type=int, default=7200)
    a = ap.parse_args()
    protocol = json.loads(a.protocol.read_text())
    assert protocol["status"] in ("FROZEN", "PILOT_FROZEN", "DISCOVERY"), protocol["status"]
    assert a.seed in protocol["seeds"], a.seed
    a.output.mkdir(parents=True, exist_ok=True)
    st = a.output / "status.json"
    if st.exists() and json.loads(st.read_text()).get("status") == "complete":
        print("already complete"); return
    try:
        run_panel(protocol, a.protocol, a.seed, a.output, a.max_seconds)
    except TimeoutError as exc:
        write(st, {"status": "timeout", "error": str(exc)}); sys.exit(3)
    except Exception as exc:  # noqa: BLE001
        write(st, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}); raise


if __name__ == "__main__":
    main()
