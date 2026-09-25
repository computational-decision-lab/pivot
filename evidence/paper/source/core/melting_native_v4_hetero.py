"""Melting Pot adaptive extension, cohort v4: heterogeneous (typed) responders.

One root seed = one response world. The responder type is a latent property of the
world, drawn by a pre-registered rule from the seed (see protocol), NEVER exposed to
any selection method. Two types:
  cooperative : responder maximises its own return              (identical to v3)
  competitive : responder maximises own return minus focal return (relative payoff)

Everything else is byte-for-byte the v3 native pipeline (official frozen stag/hare
skill networks, episodic Bernoulli mixture responder, plain SGD logit learning,
stratified paired evaluation blocks, 704 native episodes per root):
  response training  : (1 incumbent + 8 candidates) x 32 episodes            = 288
  proxy block        : frozen responder q=.5, 8 replicates x 4 strata        =  32
  selection bank     : 8 candidates x (8 replicates x 4 strata)              = 256
  audit A, B         : 2 x (16 replicates x 4 strata)                        = 128

This worker produces the PANEL only (proxy deltas, selection bank, audit labels).
All selection methods are replayed offline by analyze_v4.py from the sealed bank, so
method comparison costs zero extra native episodes and cannot read audit labels.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import melting_hierarchical_e5 as v1
import melting_hierarchical_e5_stratified as v2

VERSION = "melting_native_hetero_v4"
GRID = [.125, .225, .325, .425, .575, .675, .775, .875]
TYPES = {"cooperative": 0.0, "competitive": 1.0}   # lambda in reward = own - lambda * focal


def logit(p):
    return math.log(p / (1 - p))


def responder_type(seed: int, protocol: dict) -> str:
    rule = protocol["responder_type_rule"]
    assert rule == "competitive_if_seed_odd_else_cooperative", rule
    return "competitive" if seed % 2 == 1 else "cooperative"


def train_logit_typed(episodes, *, other_logit, key, template, count, learning_rate, lam):
    """v1.train_logit for role='response' with reward = response_return - lam * focal_return.

    lam = 0 reproduces v1.train_logit exactly (same episodes, same arithmetic order).
    """
    config = episodes.config
    theta = 0.0
    past = 0.0
    history = []
    for index in range(count):
        row = episodes.run((*key, index), (*template, index), other_logit, theta)
        p = v1.probability(theta)
        z = row["sampled_specialists"][1]
        raw_own = float(row["response_return"])
        reward = raw_own - lam * float(row["focal_return"])
        scaled = float(np.clip(reward / config["reward_scale"], -config["reward_clip"], config["reward_clip"]))
        baseline = past / index if index and config["baseline"] == "running_mean" else 0.0
        advantage = scaled - baseline
        gradient = advantage * (z - p)
        updated = float(np.clip(theta + learning_rate * gradient, -config["logit_clip"], config["logit_clip"]))
        history.append({"episode_id": row["episode_id"], "logit_before": theta, "probability": p, "sampled_stag": z,
                        "native_return": raw_own, "focal_return": float(row["focal_return"]), "typed_reward": reward,
                        "scaled_clipped_return": scaled, "baseline_before": baseline, "advantage": advantage,
                        "gradient": gradient, "learning_rate": learning_rate, "logit_after": updated})
        theta = updated
        past += scaled
        v1.write(episodes.output / "training" / (v1.digest(key)[:24] + ".json"),
                 {"role": "response", "initial_logit": 0.0, "other_logit": other_logit, "lambda": lam,
                  "optimizer": "plain_SGD_no_momentum", "baseline": config["baseline"],
                  "baseline_uses_only_previous_training_episodes": True, "past_scaled_reward_sum": past,
                  "training_episode_count": index + 1, "history": history, "final_logit": theta,
                  "complete": index + 1 == count})
    return theta


def execute(a, backend):
    out = a.output
    out.mkdir(parents=True, exist_ok=bool(a.resume))
    protocol = v2.read(a.protocol)
    assert protocol["status"] in ("FROZEN", "PILOT_FROZEN"), protocol["status"]
    assert protocol["candidate_probabilities"] == GRID
    assert a.seed in protocol["seeds"], a.seed
    kind = responder_type(a.seed, protocol)
    lam = TYPES[kind]
    c = {"task": "melting_stag", "seed": a.seed, "horizon": 2500, "reward_scale": 100., "reward_clip": 2., "logit_clip": 8.,
         "baseline": "running_mean", "stratum_replicates": 8, "proxy_stratum_replicates": 8}
    v2.write(out / "protocol.json", {"version": VERSION, "seed": a.seed, "config": c, "responder_type": kind, "lambda": lam,
                                      "batch_protocol_sha256": v2.sha(a.protocol), "backend": backend.metadata})
    v2.write(out / "candidates.json", {"probabilities": GRID, "incumbent": .5,
                                        "operator": "fixed symmetric mixture edits, no outcome-dependent proposal selection"})
    episodes = v2.Episodes(out, backend, c, a.max_seconds, 2500 * 704)
    start = time.monotonic()
    histories, q = {}, {}
    for j, p in [("incumbent", .5)] + [(str(i), p) for i, p in enumerate(GRID)]:
        v2.write(out / "status.json", {"status": "response_training", "target": j, "responder_type": kind})
        key = ("response_training", j)
        train_logit_typed(episodes, other_logit=logit(p), key=key, template=(VERSION, "response_training"),
                          count=32, learning_rate=protocol["response_learning_rate"], lam=lam)
        path = out / "training" / (v2.digest(key)[:24] + ".json")
        history = v2.read(path)
        assert history["complete"]
        histories[str(path.relative_to(out))] = v2.sha(path)
        q[j] = {str(h): v2.probability(history["history"][h - 1]["logit_after"]) for h in protocol["adaptation_episodes"]}
    v2.write(out / "training_seal.json", {"history_hashes": histories, "response_probabilities": q,
                                           "responder_type": kind, "all_frozen_before_proxy": True})
    seal_hash = v2.sha(out / "training_seal.json")

    def block(stage, key):
        context = {"training_seal_sha256": seal_hash, "candidate_sha256": v2.sha(out / "candidates.json")}
        return v2.evaluation_block(episodes, (stage, key), (VERSION, stage, key), out / stage / (str(key) + ".json"), context)

    v2.write(out / "status.json", {"status": "proxy"})
    proxy = block("proxy", "common")
    old = v2.weighted_estimate(proxy, .5, .5)["focal_return"]
    proxies = {str(i): v2.weighted_estimate(proxy, p, .5)["focal_return"] - old for i, p in enumerate(GRID)}
    v2.write(out / "features.json", [{"id": str(i), "p": p, "features": [(p - .5) / .25, abs(p - .5) / .25],
                                       "proxy_delta": proxies[str(i)]} for i, p in enumerate(GRID)])
    v2.write(out / "status.json", {"status": "selection_bank"})
    bank = {}
    for h in protocol["adaptation_episodes"]:
        hs = str(h)
        bank[hs] = {}
        for i, p in enumerate(GRID):
            b = block("selection", str(i))
            bank[hs][str(i)] = (v2.weighted_estimate(b, p, q[str(i)][hs])["focal_return"]
                                - v2.weighted_estimate(b, .5, q["incumbent"][hs])["focal_return"])
    # Freeze the complete method input (proxy + bank) BEFORE any audit episode exists.
    v2.write(out / "decisions_frozen.json", {"version": VERSION, "kind": "sealed_selection_bank_not_decisions",
                                              "selection_bank": bank, "proxy_deltas": proxies,
                                              "note": "All selectors are replayed offline from this bank by analyze_v4.py; audit never enters."})
    v2.write(out / "selection_seal.json", {"decisions_sha256": v2.sha(out / "decisions_frozen.json"),
                                            "candidate_sha256": v2.sha(out / "candidates.json"),
                                            "features_sha256": v2.sha(out / "features.json"),
                                            "training_seal_sha256": seal_hash, "audit_generated": False})
    v2.write(out / "status.json", {"status": "held_out_audit"})
    episodes.config = {**c, "stratum_replicates": 16}
    audits = [block("audit", b) for b in ["A", "B"]]
    v2.verify_seal(out)
    assert v2.sha(out / "training_seal.json") == seal_hash
    labels, mechanism = {}, []
    for h in protocol["adaptation_episodes"]:
        hs = str(h)
        labels[hs] = {"incumbent": 0.0}
        for i, p in enumerate(GRID):
            j = str(i)
            per_block = []
            for bname, b in zip(["A", "B"], audits):
                new = v2.weighted_estimate(b, p, q[j][hs]); old_ = v2.weighted_estimate(b, .5, q["incumbent"][hs])
                fixed_new = v2.weighted_estimate(b, p, .5); fixed_old = v2.weighted_estimate(b, .5, .5)
                dep = new["focal_return"] - old_["focal_return"]; prox = fixed_new["focal_return"] - fixed_old["focal_return"]
                per_block.append(dep)
                mechanism.append({"candidate": j, "probability": p, "adaptation": h, "block": bname,
                                  "proxy_delta_audit": prox, "deployment_delta_audit": dep, "gap": dep - prox,
                                  "responder_own_gain": new["response_return"] - fixed_new["response_return"]})
            labels[hs][j] = float(np.mean(per_block))
    assert len(episodes.costs) == 704 and all(r["complete"] for r in episodes.costs.values())
    summary = {"status": "complete", "version": VERSION, "seed": a.seed, "responder_type": kind, "lambda": lam,
               "actual_native_episodes": len(episodes.costs), "actual_native_frames": episodes.frames,
               "duration_seconds": time.monotonic() - start, "source_protocol_sha256": v2.sha(a.protocol),
               "response_probabilities": q, "proxy_deltas": proxies, "selection_bank": bank,
               "audit_gains": labels, "mechanism_audit": mechanism,
               "scope": "Single-round conditional response-world ISR; eight fixed skill-mixture updates; typed responder is a latent world property."}
    v2.write(out / "summary.json", summary)
    v2.write(out / "status.json", {"status": "complete"})
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--crossbench-root", type=Path, required=True); p.add_argument("--max-seconds", type=int, default=14400)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    from melting_reference_diagnostic import OfficialSpecialistRollout
    engine = OfficialSpecialistRollout(a.model_path, horizon=2500, crossbench_root=a.crossbench_root)
    try:
        print(json.dumps(execute(a, engine)), flush=True)
    except v2.BudgetStop as exc:
        v2.write(a.output / "status.json", {"status": "timeout", "error": str(exc), "partial_evidence_retained": True})
        raise SystemExit(124)
    except Exception as exc:
        v2.write(a.output / "status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "partial_evidence_retained": True})
        raise
    finally:
        engine.close()


if __name__ == "__main__":
    main()
