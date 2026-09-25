#!/usr/bin/env bash
# Leduc v4 (PIVOT component study): unit tests -> generator selftest -> pilot (gate) -> calibration -> LORO (gate)
#                                   -> confirmation -> analysis -> analyze_v4 cross-check.
# Every stage is resumable; rerunning skips finished stages. Gates write STOP_<stage> files and exit 3.
#
# Required env:  OSPY   python with pyspiel (the openspiel_env used for v2b)
#                CODE   directory holding code/ and protocols/leduc_v4/ from this package
#                OSOUT  output root (e.g. /root/autodl-tmp/openspiel_v4_20260921)
# Optional:      WORKERS (default 8)
set -euo pipefail
: "${OSPY:?set OSPY}"; : "${CODE:?set CODE}"; : "${OSOUT:?set OSOUT}"
WORKERS="${WORKERS:-8}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$CODE/code:${PYTHONPATH:-}"
PANEL="$CODE/code/openspiel_v4_panel.py"; ANALYZE="$CODE/code/analyze_v5.py"; PROT="$CODE/protocols/leduc_v4"
OUT="$OSOUT"; mkdir -p "$OUT"; LOG="$OUT/v4_suite.log"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "$(stamp) $*" | tee -a "$LOG"; }
jget() { "$OSPY" - "$1" "$2" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."):
    d = d[k]
print(json.dumps(d))
EOF
}

# ---------------------------------------------------------------- 0. tests + selftest
if [ ! -f "$OUT/.stage0_ok" ]; then
  say "STAGE0 unit tests + generator selftest"
  ( cd "$CODE/code" && "$OSPY" -m unittest test_pivot_v3 -v ) >> "$LOG" 2>&1
  "$OSPY" "$PANEL" --selftest --output "$OUT/selftest_kuhn" >> "$LOG" 2>&1
  grep -q SELFTEST_OK "$LOG" || { say "SELFTEST FAILED"; exit 2; }
  touch "$OUT/.stage0_ok"
fi

# ---------------------------------------------------------------- 1. pilot (4 roots) + gate
PILOT="$OUT/leduc_v4_pilot"
if [ ! -f "$PILOT/status.json" ]; then
  say "STAGE1 pilot generation (4 roots)"
  "$OSPY" "$PANEL" --protocol "$PROT/pilot.json" --output "$PILOT" --workers 4 --resume >> "$LOG" 2>&1
fi
if [ ! -f "$PILOT/analysis/pilot_report.json" ]; then
  "$OSPY" "$ANALYZE" --mode pilot --protocol "$PROT/pilot.json" --calibration "$PILOT" --output "$PILOT/analysis" >> "$LOG" 2>&1
fi
if [ "$(jget "$PILOT/analysis/pilot_report.json" pilot_go)" != "true" ]; then
  say "PILOT GATE FAILED: $(jget "$PILOT/analysis/pilot_report.json" pilot_checks)"; touch "$OUT/STOP_pilot_gate"; exit 3
fi
say "pilot gate passed: $(jget "$PILOT/analysis/pilot_report.json" pilot_checks)"

# ---------------------------------------------------------------- 2. calibration (12 roots; pilot roots reused)
CAL="$OUT/leduc_v4_calibration"
if [ ! -f "$CAL/status.json" ]; then
  say "STAGE2 calibration generation (12 roots, 4 reused from pilot)"
  mkdir -p "$CAL"
  for r in 97000 97001 97002 97003; do [ -d "$CAL/seed_$r" ] || cp -R "$PILOT/seed_$r" "$CAL/seed_$r"; done
  "$OSPY" "$PANEL" --protocol "$PROT/calibration.json" --output "$CAL" --workers "$WORKERS" --resume >> "$LOG" 2>&1
fi

# ---------------------------------------------------------------- 3. LORO pre-flight + gate
LORO="$OUT/leduc_v4_loro"
if [ ! -f "$LORO/summary.json" ]; then
  say "STAGE3 leave-one-root-out replay"
  "$OSPY" "$ANALYZE" --mode loro --protocol "$PROT/calibration.json" --calibration "$CAL" --output "$LORO" >> "$LOG" 2>&1
fi
say "LORO gates: $(jget "$LORO/summary.json" gates)"
if [ "$(jget "$LORO/summary.json" gates.all_pass)" != "true" ]; then
  say "LORO GATE FAILED (C1 coverage or C2 not-worse-than-calibration)"; touch "$OUT/STOP_loro_gate"; exit 3
fi
if [ "$(jget "$LORO/summary.json" gates.C3_pass)" != "true" ]; then
  say "NOTE: C3 (acquisition discriminability) failed -> H_B will be reported as not testable in this design; confirmation continues"
fi

# ---------------------------------------------------------------- 4. confirmation (30 roots) + analysis
CON="$OUT/leduc_v4_confirm"
if [ ! -f "$CON/status.json" ]; then
  say "STAGE4 confirmation generation (30 roots)"
  "$OSPY" "$PANEL" --protocol "$PROT/confirm.json" --output "$CON" --workers "$WORKERS" --resume >> "$LOG" 2>&1
fi
if [ ! -f "$CON/analysis/summary.json" ]; then
  say "STAGE4 confirmation analysis"
  "$OSPY" "$ANALYZE" --mode confirm --protocol "$PROT/confirm.json" --calibration "$CAL" --test "$CON" --output "$CON/analysis" >> "$LOG" 2>&1
fi

# ---------------------------------------------------------------- 5. cross-check with the corrected analyze_v4 (fixed-size methods only)
if [ ! -f "$CON/analysis_v4_crosscheck/summary.json" ]; then
  say "STAGE5 analyze_v4 cross-check"
  "$OSPY" - "$PROT/confirm.json" "$CON/protocol_v4crosscheck.json" <<'EOF'
import json, sys
p = json.load(open(sys.argv[1])); p["primary_method"] = "pivot_kg"; p["primary_comparator"] = "uniform_v2"
json.dump(p, open(sys.argv[2], "w"), indent=1)
EOF
  "$OSPY" "$CODE/code/analyze_v4.py" --mode confirm --protocol "$CON/protocol_v4crosscheck.json" --calibration "$CAL" --test "$CON" --output "$CON/analysis_v4_crosscheck" >> "$LOG" 2>&1 || say "cross-check failed (non-fatal)"
fi

say "V4_COMPLETE"
"$OSPY" - "$CON/analysis/summary.json" <<'EOF' | tee -a "$LOG"
import json, sys
s = json.load(open(sys.argv[1])); h = s["hypotheses"]
def f(b): return None if not b else [round(b["mean"], 4), round(b["lo"], 4), round(b["hi"], 4)]
print(json.dumps({"n": s["n_roots"], "types": s["types"],
                  "H_A_pivot_minus_E_uniform": f(h["H_A_allocation_pivot_minus_expected_uniform"]),
                  "H_B_pivot_minus_ivr_menu": f(h["H_B_acquisition_pivot_minus_ivr_menu"]), "H_B_testable": h["H_B_testable_by_design_gate"],
                  "H_B2_pivot_minus_lucb": f(h["H_B2_pivot_minus_lucb_fixed"]),
                  "H_C_pivot_minus_unpaired": f(h["H_C_pairing_pivot_minus_unpaired"]),
                  "H_D_exact_minus_noisy_calibration": f(h["H_D_exact_minus_noisy_calibration"]),
                  "H_E_stop_hands_minus_menu": f(h["H_E_stop_hands_minus_menu_hands"]), "H_E_stop_gain_minus_menu": f(h["H_E_stop_gain_minus_menu"]),
                  "H_F_menu_minus_fixed": f(h["H_F_menu_minus_fixed_size"]),
                  "acquisition_discriminability": h["acquisition_discriminability"],
                  "se_ratio_unpaired_over_paired_long": [round(x, 2) for x in s["mechanism"][str(max(int(k) for k in s["mechanism"]))]["se_ratio_unpaired_over_paired"]]}, indent=1))
EOF
