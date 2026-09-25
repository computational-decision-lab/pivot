#!/usr/bin/env bash
# E1: finish the pre-registered Leduc v3 generalisation suite (eta_0.05, eta_0.10, eta_0.25, pex_0.80, pex_0.95, mix_cont;
#     eta_0.40 already has rollouts) with the FROZEN analysis (as pre-registered, via the original suite script), then add a
#     CORRECTED re-analysis (calibration-estimated noise) of every point, including eta_0.15 v2b, into *_corrected directories.
#
# Required env: OSPY (python with pyspiel), CODE_V3 (assembled dir, see CODEX_PROMPT.md), OSOUT (= /root/autodl-tmp/openspiel_v2_20260918)
#               FIXED_ANALYZE (= /root/autodl-tmp/leduc_fix_20260920/code/analyze_v4.py)
# Optional: WORKERS (default 8), POINTS
set -euo pipefail
: "${OSPY:?}"; : "${CODE_V3:?}"; : "${OSOUT:?}"; : "${FIXED_ANALYZE:?}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKERS="${WORKERS:-8}"
POINTS="${POINTS:-eta_0.05 eta_0.10 eta_0.25 pex_0.80 pex_0.95 mix_cont eta_0.40}"
SUITE_OUT="$OSOUT/leduc_v3_suite"; LOG="$SUITE_OUT/suite_rest.log"; mkdir -p "$SUITE_OUT"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "$(stamp) $*" | tee -a "$LOG"; }

# 1) frozen (pre-registered) pipeline through the original suite runner: generation -> LORO -> confirm -> frozen analysis
say "E1 frozen pipeline for: $POINTS"
OSPY="$OSPY" CODE="$CODE_V3" OSOUT="$OSOUT" WORKERS="$WORKERS" POINTS="$POINTS" bash "$CODE_V3/openspiel_v2/run_leduc_v3_suite.sh" 2>&1 | tee -a "$LOG"

# 2) corrected re-analysis of every point (zero rollouts)
export PYTHONPATH="$(dirname "$FIXED_ANALYZE"):/root/autodl-tmp/colin_pivot/src:/root/autodl-tmp/colin_pivot:${PYTHONPATH:-}"
for P in $POINTS; do
  CAL="$SUITE_OUT/${P}_calibration"; CON="$SUITE_OUT/${P}_confirm"
  [ -f "$CAL/status.json" ] && [ -f "$CON/status.json" ] || { say "skip $P (incomplete)"; continue; }
  if [ ! -f "$SUITE_OUT/${P}_loro_corrected/summary.json" ]; then
    "$OSPY" "$FIXED_ANALYZE" --mode loro --protocol "$CAL/protocol.json" --calibration "$CAL" --output "$SUITE_OUT/${P}_loro_corrected" >> "$LOG" 2>&1
  fi
  if [ ! -f "$CON/analysis_corrected/summary.json" ]; then
    "$OSPY" "$FIXED_ANALYZE" --mode confirm --protocol "$CON/protocol.json" --calibration "$CAL" --test "$CON" --output "$CON/analysis_corrected" >> "$LOG" 2>&1
  fi
  "$OSPY" - "$CON/analysis/summary.json" "$CON/analysis_corrected/summary.json" "$P" <<'EOF' | tee -a "$LOG"
import json, sys
out = {"point": sys.argv[3]}
for tag, path in (("frozen", sys.argv[1]), ("corrected", sys.argv[2])):
    try:
        s = json.load(open(path)); h = s["hypotheses"]["H2_long_primary_gain_minus_comparator_cap192"]
        hl = max(int(k) for k in s["mechanism"] if k.isdigit()); m = s["mechanism"][str(hl)]
        row = {r["method"]: round(r["mean_isr"], 4) for r in s["method_table_cap192"] if r["adaptation"] == hl}
        out[tag] = {"H2": [round(h["mean"], 4), round(h["lo"], 4), round(h["hi"], 4)], "VWI": round(m["value_of_world_specific_information_ceiling"], 4),
                    "ISR": {k: row.get(k) for k in ("pivot_kg", "uniform_v2", "no_hf_v2", "all_hf_reference", "proxy_only")}}
    except Exception as exc:  # noqa: BLE001
        out[tag] = f"missing: {exc}"
print(json.dumps(out))
EOF
done
say "E1_COMPLETE"
