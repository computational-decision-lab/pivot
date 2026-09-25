"""Generate pilot / calibration / confirm.draft protocols for the HighwayEnv leg from the discovery
protocol that PASSED the discovery rule. Copies world definition verbatim (traffic_types, ego_spec, sim,
candidates, response_levels); only seeds, sample sizes, gates and bookkeeping differ. Deterministic:
same input -> same output. Usage:
    python make_highway_protocols.py --from protocols/highway_v1_discovery.json --out protocols/
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

WORLD_KEYS = ["traffic_types", "ego_spec", "candidate_alphas", "incumbent_alpha", "response_levels", "sim",
              "responder_type_rule", "responder_type_is_latent", "hf_caps", "primary_hf_cap", "query_cost",
              "adaptation_steps", "methods"]
PILOT_SEEDS = list(range(61000, 61004))
CAL_SEEDS = list(range(61000, 61012))
CONF_SEEDS = list(range(62000, 62030))
SAMPLES = {"n_proxy_episodes": 16, "n_query_pairs": 8, "n_audit_pairs": 16}   # query_cost 24 = 8 pairs x 2 sims (+ setup 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    disc = json.loads(a.src.read_text())
    world = {k: copy.deepcopy(disc[k]) for k in WORLD_KEYS}
    src_sha = hashlib.sha256(a.src.read_bytes()).hexdigest()
    base = {**world, **SAMPLES, "discovery_protocol": a.src.name, "discovery_protocol_sha256": src_sha,
            "hypothesis_chain": "Replayed (non-responsive) traffic is the proxy; reactive traffic with a latent driver-population type is deployment. "
                                "Aggressive ego edits look bad against replay (followers never brake) but good against yielding drivers and bad against assertive ones, "
                                "so the deployed optimum is world-specific; paired HF queries carry decision value and decision-sensitive allocation should beat uniform "
                                "allocation of the same budget, more so when a larger fraction of traffic reacts (long) than when few vehicles react (short)."}

    pilot = {"protocol_id": "highway_v1_pilot", "status": "PILOT_FROZEN", "seeds": PILOT_SEEDS, **base,
             "pilot_go_criteria": {"min_info_ceiling_fraction_of_oracle": 0.2},
             "pilot_go_rule": "analyze_v4.py --mode pilot: best_candidate_differs_by_type AND world_specific_info_ceiling_at_least_fraction_of_oracle (generic two-type criteria). pilot_go false -> STOP, report, do not tune."}
    cal = {"protocol_id": "highway_v1_calibration", "status": "PILOT_FROZEN", "seeds": CAL_SEEDS, **base,
           "reuses_pilot_roots": PILOT_SEEDS, "primary_method": "pivot_kg", "primary_comparator": "uniform_v2",
           "preflight_gate_before_confirmation_freeze": {
               "loro_long_kg_minus_uniform_mean_positive": True,
               "loro_long_kg_not_worse_than_no_hf_v2_by_more_than": 0.0,
               "coverage_in_range": [0.80, 1.00]},
           "note": "Gate mirrors melting_v4_calibration: mean(long, cap primary, kg - uniform) > 0, kg not worse than calibrated no-HF, LORO predictive coverage in [0.80, 1.00]."}
    conf = {"protocol_id": "highway_v1_confirm", "status": "DRAFT_FREEZE_ONLY_AFTER_PILOT_GO_AND_PREFLIGHT_PASS", "seeds": CONF_SEEDS, **base,
            "seed_disjointness": "62000-62029 disjoint from discovery 60000-60007 and calibration 61000-61011",
            "calibration_source": "highway_v1_calibration (12 roots)", "primary_method": "pivot_kg", "primary_comparator": "uniform_v2",
            "primary_contrasts": ["H2: long (phi=1.0), cap primary: gain(pivot_kg) - gain(uniform_v2), paired by root",
                                  "H3: [gain(pivot_kg) - gain(uniform_v2)]_long - [same]_short, paired by root"],
            "primary_success_rule": "Both H2 and H3 root-bootstrap 95% CI lower endpoints > 0 (10,000 draws, seed 20260917). Mechanism (H1) reported alongside: squared-gap long-minus-short CI lower > 0 and best-candidate identity differs by latent type.",
            "secondary_prespecified": ["long smallest cap (single query) kg - uniform", "kg - ivr_v2", "kg - no_hf_v2 (query value over calibration)",
                                       "no_hf_v2 - proxy_only (calibration value)", "kg_stop cost and gain vs kg", "short kg - uniform", "per-type stratified ISR (descriptive)"],
            "if_null": "Retain and report. Do not add seeds, do not change presets/phi/caps, do not re-tune the posterior on 62xxx.",
            "freeze_procedure": "Codex: after pilot_go == true AND preflight PASS run freeze_confirm.py --draft this file --also-hash code/highway_v1/*.py; it sets status FROZEN and records hashes. Only then run any 62xxx seed."}
    a.out.mkdir(parents=True, exist_ok=True)
    for name, obj in [("highway_v1_pilot.json", pilot), ("highway_v1_calibration.json", cal), ("highway_v1_confirm.draft.json", conf)]:
        (a.out / name).write_text(json.dumps(obj, indent=2) + "\n")
        print("wrote", a.out / name, hashlib.sha256((a.out / name).read_bytes()).hexdigest()[:16])


if __name__ == "__main__":
    main()
