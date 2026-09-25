#!/usr/bin/env bash
# Leduc v3 generalization suite runner: for each point, calibration(12) -> LORO -> confirmation(30) -> analyze.
# Every stage is resumable; rerunning the script skips finished stages. No gate: every point runs to completion.
#
# Required env:  OSPY   python with pyspiel (same one used for v2b)
#                CODE   directory holding analyze_v4.py + openspiel_v2/ + protocols/leduc_v3_suite/
#                OSOUT  output root (same as v2b, e.g. $BASE/openspiel_v2_20260918)
# Optional:      WORKERS (default 8), POINTS (space-separated subset, default all in suite_manifest order)
set -euo pipefail
: "${OSPY:?set OSPY}"; : "${CODE:?set CODE}"; : "${OSOUT:?set OSOUT}"
WORKERS="${WORKERS:-8}"
SUITE="$CODE/protocols/leduc_v3_suite"
PANEL="$CODE/openspiel_v2/openspiel_v2_panel.py"
ANALYZE="$CODE/analyze_v4.py"
OUT="$OSOUT/leduc_v3_suite"; mkdir -p "$OUT"
LOG="$OUT/suite.log"
POINTS="${POINTS:-$($OSPY -c "import json;print(' '.join(p['id'] for p in json.load(open('$SUITE/suite_manifest.json'))['points']))")}"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
for P in $POINTS; do
  CAL="$OUT/${P}_calibration"; CON="$OUT/${P}_confirm"
  echo "$(stamp) START $P" | tee -a "$LOG"
  if [ ! -f "$CAL/status.json" ]; then
    $OSPY "$PANEL" --protocol "$SUITE/${P}_calibration.json" --output "$CAL" --workers "$WORKERS" --resume >> "$LOG" 2>&1
  fi
  if [ ! -f "$OUT/${P}_loro/summary.json" ]; then
    $OSPY "$ANALYZE" --mode loro --protocol "$SUITE/${P}_calibration.json" --calibration "$CAL" --output "$OUT/${P}_loro" >> "$LOG" 2>&1
  fi
  if [ ! -f "$CON/status.json" ]; then
    $OSPY "$PANEL" --protocol "$SUITE/${P}_confirm.json" --output "$CON" --workers "$WORKERS" --resume >> "$LOG" 2>&1
  fi
  if [ ! -f "$CON/analysis/summary.json" ]; then
    $OSPY "$ANALYZE" --mode confirm --protocol "$SUITE/${P}_confirm.json" --calibration "$CAL" --test "$CON" --output "$CON/analysis" >> "$LOG" 2>&1
  fi
  echo "$(stamp) DONE  $P" | tee -a "$LOG"
  $OSPY - "$CON/analysis/summary.json" "$P" <<'EOF' | tee -a "$LOG"
import json, sys
s = json.load(open(sys.argv[1])); h = s["hypotheses"]; v = h["H2_long_primary_gain_minus_comparator_cap192"]
hl = max(int(k) for k in s["mechanism"] if k.isdigit()); m = s["mechanism"][str(hl)]
row = {r["method"]: r["mean_isr"] for r in s["method_table_cap192"] if r["adaptation"] == hl}
print(json.dumps({"point": sys.argv[2], "n": s["n_roots"], "types": s["types"],
                  "H2_kg_minus_uniform": [round(v["mean"], 4), round(v["lo"], 4), round(v["hi"], 4)],
                  "VWI_ceiling_long": round(m["value_of_world_specific_information_ceiling"], 4),
                  "ISR_long": {k: round(row[k], 4) for k in ("pivot_kg", "uniform_v2", "no_hf_v2", "all_hf_reference", "proxy_only") if k in row},
                  "author_baselines": s.get("author_baselines")}))
EOF
done
echo "$(stamp) SUITE_COMPLETE" | tee -a "$LOG"
