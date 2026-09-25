"""Generate the frozen Leduc v3 generalization suite (dose-response + robustness) from the v2b template.

v2b (eta=0.15, 50/50 parity types, 4096 hands) is the confirmed primary result and is NOT rerun.
The suite varies exactly ONE factor per point relative to v2b and runs calibration(12) -> LORO -> confirmation(30)
for EVERY point regardless of LORO outcome. There is no gate and no stopping rule: all points are reported, so the
suite is a descriptive dose-response, not a search. Predictions are written into suite_manifest.json before any run.

Usage:  python make_leduc_v3_suite.py --template ../../protocols/openspiel_v2b_leduc_confirm.json \
                                      --out ../../protocols/leduc_v3_suite
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

POINTS = [
    # id,          factor,        value,  seed_base
    ("eta_0.05",   "eta",         0.05,   100000),
    ("eta_0.10",   "eta",         0.10,   101000),
    ("eta_0.25",   "eta",         0.25,   102000),
    ("eta_0.40",   "eta",         0.40,   103000),
    ("pex_0.80",   "p_exploiter", 0.80,   104000),
    ("pex_0.95",   "p_exploiter", 0.95,   105000),
    ("mix_cont",   "mixture",     None,   106000),
]

PREDICTIONS = {
    "axis_eta": {
        "points": ["eta_0.05", "eta_0.10", "v2b eta_0.15 (existing)", "eta_0.25", "eta_0.40"],
        "mechanism_prediction": "VWI(long) is ~0 at eta=0.05 (w_long=0.34: the exploiter's response is too weak to flip the "
                                "sign of large updates, both types share best alpha=1), becomes positive between 0.08 and 0.15 "
                                "(discovery: eta=0.08 gave a split exploiter), and saturates for eta>=0.25 (w_long>=0.90).",
        "method_prediction": "PIVOT-Uniform (long, 2 queries) tracks VWI: null at eta=0.05, positive and roughly flat for eta>=0.15. "
                             "PIVOT-noHF likewise. Under short response all methods coincide at every eta.",
        "what_a_violation_means": "If PIVOT-Uniform is positive where VWI~0, the gain is not coming from world-specific information "
                                  "and Prop.7 is wrong; if it is null where VWI is large and the instrument resolves the margin, "
                                  "the allocation rule is at fault.",
    },
    "axis_type_imbalance": {
        "points": ["v2b 50/50 (existing)", "pex_0.80", "pex_0.95"],
        "mechanism_prediction": "As P(exploiter) -> 1 the population prior already picks the incumbent in almost every world; VWI "
                                "shrinks proportionally to P(equilibrator) x equilibrator gain (~0.15 x 0.2 = 0.03 at 0.80, ~0.007 at 0.95).",
        "method_prediction": "PIVOT-Uniform shrinks toward 0 and PIVOT-noHF shrinks toward 0; at 0.95 the calibrated no-query rule is "
                             "within noise of PIVOT and of all-HF. This is the 'calibration suffices' boundary approached continuously.",
        "caveat": "With 12 calibration roots and p=0.95 the calibration set may contain 0-1 equilibrator worlds; posterior coverage may "
                  "fall below 0.80. That is reported, not repaired.",
    },
    "axis_continuous_type": {
        "points": ["mix_cont"],
        "mechanism_prediction": "m~U(0,1) per world; deployment-optimal alpha moves from 1 (m~0) through interior values to 0 (m~1). "
                                "Best candidate is no longer binary; 'type' bins are for reporting only.",
        "method_prediction": "PIVOT-Uniform (long) remains positive (the two largest updates remain the most informative about m), "
                             "but PIVOT's ISR is higher than in v2b because interior optima require more than type identification.",
        "why_it_matters": "Shows the result does not depend on a clean two-type construction.",
    },
    "cross_point_check": "Plot realised PIVOT-Uniform (long, primary cap) against VWI ceiling estimated from each cohort's own audit "
                         "labels, for all 7 suite points + v2b Leduc + v2b Kuhn + v2 Kuhn + frozen Kuhn + Melting Pot v3/v4. "
                         "Prop.7 predicts a monotone relation with zero intercept.",
    "not_done": "No re-tuning of shrinkage, noise floor (r_floor=0.25 retained for comparability; scale-matched sensitivity is a "
                "separate local analysis), candidate grid, query_hands, or seeds after seeing any result.",
}


def build(template: dict, pid: str, factor: str, value, seed_base: int):
    cal = copy.deepcopy(template)
    for k in ("protocol_sha256", "script_sha256", "open_spiel", "hypothesis_chain", "primary_contrasts",
              "primary_success_rule", "secondary_prespecified", "if_null", "inference_unit", "boundary_results_retained",
              "why_v2b", "eta_selection_evidence", "calibration_seeds", "seeds"):
        cal.pop(k, None)
    cal["suite"] = "leduc_v3_generalization_20260918"
    cal["suite_point"] = pid
    cal["varied_factor"] = factor
    cal["status"] = "FROZEN"
    if factor == "eta":
        cal["eta"] = value
        cal["response_weights"] = {h: 1 - (1 - value) ** t for h, t in cal["adaptation_steps"].items()}
    elif factor == "p_exploiter":
        cal["responder_type_rule"] = "bernoulli"
        cal["p_exploiter"] = value
    elif factor == "mixture":
        cal["responder_type_rule"] = "continuous_mixture"
    else:
        raise ValueError(factor)
    cal["responder_type_is_latent"] = "never shown to selectors (nor is mixture_m)"
    cal_seeds = list(range(seed_base, seed_base + 12))
    con_seeds = list(range(seed_base + 500, seed_base + 530))

    cal["protocol_id"] = f"leduc_v3_{pid}_calibration_20260918"
    cal["seeds"] = cal_seeds
    cal["preflight_gate_before_confirmation_freeze"] = {
        "note": "DESCRIPTIVE ONLY for the suite: LORO summary is reported; confirmation runs for every point regardless."}

    con = copy.deepcopy(cal)
    con["protocol_id"] = f"leduc_v3_{pid}_confirm_20260918"
    con["seeds"] = con_seeds
    con["calibration_seeds"] = cal_seeds
    con.pop("preflight_gate_before_confirmation_freeze", None)
    con["primary_contrasts"] = template["primary_contrasts"]
    con["primary_success_rule"] = ("Reported as in v2b (root-bootstrap 95% CI, 10000 draws, seed 20260917) but NOT used as a gate; "
                                   "the suite's claim is the dose-response across points, pre-stated in suite_manifest.json.")
    con["inference_unit"] = "root seed (30 per point)"
    return cal, con


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    template = json.loads(a.template.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {"suite": "leduc_v3_generalization_20260918", "template": str(a.template.name), "template_eta": template["eta"],
                "points": [], "predictions": PREDICTIONS}
    for pid, factor, value, base in POINTS:
        cal, con = build(template, pid, factor, value, base)
        (a.out / f"{pid}_calibration.json").write_text(json.dumps(cal, indent=1))
        (a.out / f"{pid}_confirm.json").write_text(json.dumps(con, indent=1))
        manifest["points"].append({"id": pid, "factor": factor, "value": value, "calibration_seeds": cal["seeds"],
                                   "confirm_seeds": con["seeds"], "eta": cal["eta"],
                                   "responder_type_rule": cal["responder_type_rule"], "p_exploiter": cal.get("p_exploiter")})
    (a.out / "suite_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("written", sorted(p.name for p in a.out.iterdir()))


if __name__ == "__main__":
    main()
