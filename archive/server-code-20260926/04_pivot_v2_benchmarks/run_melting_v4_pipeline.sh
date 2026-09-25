#!/usr/bin/env bash
# Melting Pot v4 一键流水线：calibration -> LORO 预飞行 -> 冻结 -> confirmation -> confirm 分析。
# 幂等：任何阶段已完成会被跳过；被打断后用同一命令重跑即可续上。
# 不做任何设计决定：gate 不过就退出并打印报告。协议/种子/参数全部来自冻结的 json。
#
# 用法（服务器上，nohup 跑）：
#   nohup bash run_melting_v4_pipeline.sh > $BASE/melting_v4_pipeline.log 2>&1 &
# 双机分片时机器 B：SEED_SUBSET="$(seq -s, 49015 49029)" CONF_WORKERS=15 bash run_melting_v4_pipeline.sh --confirm-only
set -uo pipefail

BASE=${BASE:-/root/autodl-tmp}
PY=${PY:-$BASE/benchmark_extensions/melting_env/bin/python}
MODEL=${MODEL:-$BASE/colin_pivot_cloud/melting_recovery_20260915/reference_smoke/reference_assets/assets/saved_models/stag_hunt_in_the_matrix__repeated/puppet_1}
CROSS=${CROSS:-$BASE/colin_melting}
CODE=${CODE:-$BASE/colin_pivot_cloud/melting_v4_hetero}
OUT=${OUT:-$BASE/melting_v4_calibration_20260918}
CONF=${CONF:-$BASE/melting_v4_confirm_20260918}
CAL_WORKERS=${CAL_WORKERS:-8}        # 8 个新种子，种子内部不可拆
CONF_WORKERS=${CONF_WORKERS:-16}     # 16 vCPU 全开（8 物理核 x2 超线程）
CONF_MAX_SECONDS=${CONF_MAX_SECONDS:-43200}   # 超线程下单种子约 3h，两波需 >6h
SEED_SUBSET=${SEED_SUBSET:-}         # 分片用，逗号分隔；空 = 全部 30 个
SEED_SUBSET=${SEED_SUBSET%,}         # 去掉可能的尾逗号
CONFIRM_ONLY=0; FREEZE_ONLY=0
[ "${1:-}" = "--confirm-only" ] && CONFIRM_ONLY=1   # 跳过 calibration/LORO/freeze，只跑 confirmation(+分析)
[ "${1:-}" = "--freeze-only" ] && FREEZE_ONLY=1     # 跑到冻结为止，不启动 confirmation（双机分片用）

export PYTHONPATH=$BASE/benchmark_extensions/meltingpot_source:$BASE/colin_melting:$BASE/colin_pivot/src:$BASE/colin_pivot:$BASE/colin_pivot_cloud:$CODE
export CUDA_VISIBLE_DEVICES='' TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=3 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "STOP: $*"; echo "PIPELINE_STOPPED"; exit 1; }

n_complete() {  # n_complete <dir> <protocol.json> -> 打印 "已完成数 / 协议种子数"
  "$PY" - "$1" "$2" <<'PYEOF'
import json, sys, pathlib
d, proto = pathlib.Path(sys.argv[1]), json.load(open(sys.argv[2]))
done = sum(1 for s in proto["seeds"] if (d / f"seed_{s}" / "status.json").exists()
           and json.loads((d / f"seed_{s}" / "status.json").read_text()).get("status") == "complete")
print(done, len(proto["seeds"]))
PYEOF
}

failed_seeds() {  # 列出 job.json 里 failed 的种子
  "$PY" - "$1" <<'PYEOF'
import json, sys, pathlib
for p in sorted(pathlib.Path(sys.argv[1]).glob("seed_*.job.json")):
    j = json.load(open(p))
    if j.get("status") == "failed": print(j["seed"], end=" ")
PYEOF
}

run_cohort() {  # run_cohort <protocol> <outdir> <workers> <max_seconds> [subset]
  local proto=$1 out=$2 workers=$3 maxs=$4 subset=${5:-}
  mkdir -p "$out"
  log "launch $(basename "$proto") -> $out workers=$workers max_seconds=$maxs subset=${subset:-ALL}"
  if [ -n "$subset" ]; then
    "$PY" "$CODE/run_melting_v4.py" --protocol "$proto" --output "$out" --resume \
        --model-path "$MODEL" --crossbench-root "$CROSS" --workers "$workers" --max-seconds "$maxs" --seed-subset "$subset" \
        >> "$out.driver.log" 2>&1
  else
    "$PY" "$CODE/run_melting_v4.py" --protocol "$proto" --output "$out" --resume \
        --model-path "$MODEL" --crossbench-root "$CROSS" --workers "$workers" --max-seconds "$maxs" \
        >> "$out.driver.log" 2>&1
  fi
  local rc=$?
  local f; f=$(failed_seeds "$out")
  [ -n "$f" ] && { for s in $f; do log "seed $s FAILED, log tail:"; tail -n 40 "$out/seed_$s.log"; done; die "cohort $(basename "$proto") has failed seeds: $f"; }
  read -r done total < <(n_complete "$out" "$proto")
  log "cohort $(basename "$proto"): $done/$total complete overall (driver rc=$rc)"
  if [ -n "$subset" ]; then
    local missing=""
    for s in ${subset//,/ }; do
      [ -f "$out/seed_$s/status.json" ] && grep -q '"complete"' "$out/seed_$s/status.json" || missing="$missing $s"
    done
    [ -z "$missing" ] || die "shard seeds not complete:$missing. If status is timeout, rerun this same command (it resumes)."
  elif [ "$done" -ne "$total" ]; then
    die "cohort incomplete ($done/$total). If status is timeout, rerun this same script (it resumes)."
  fi
}

cd "$CODE" || die "CODE dir missing: $CODE"
for f in "$PY" "$MODEL" "$CODE/run_melting_v4.py" "$CODE/melting_native_v4_hetero.py" "$CODE/analyze_v4.py" "$CODE/freeze_confirm.py" \
         "$CODE/melting_v4_calibration.json" "$CODE/melting_v4_confirm.draft.json"; do
  [ -e "$f" ] || die "missing: $f"
done
log "PIPELINE START  host=$(hostname)  nproc=$(nproc 2>/dev/null || echo '?')  confirm_only=$CONFIRM_ONLY"

if [ "$CONFIRM_ONLY" -eq 0 ]; then
  # ---------- 0. pilot 必须已完成且 pilot_go ----------
  PR=$OUT/pilot_analysis/pilot_report.json
  [ -f "$PR" ] || die "pilot_report.json missing; run pilot first (codex_执行说明 第 2 节)"
  "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$PR')).get('pilot_go') is True else 1)" || die "pilot_go is not true"
  log "pilot_go=true"

  # ---------- 1. calibration（补 8 个种子，--resume 跳过 pilot 的 4 个） ----------
  read -r done total < <(n_complete "$OUT" "$CODE/melting_v4_calibration.json")
  if [ "$done" -lt "$total" ]; then
    run_cohort "$CODE/melting_v4_calibration.json" "$OUT" "$CAL_WORKERS" 14400
  else
    log "calibration already complete ($done/$total), skip"
  fi

  # ---------- 2. LORO 预飞行 ----------
  if [ ! -f "$OUT/loro_preflight/summary.json" ]; then
    log "LORO preflight"
    "$PY" "$CODE/analyze_v4.py" --mode loro --protocol "$CODE/melting_v4_calibration.json" \
        --calibration "$OUT" --output "$OUT/loro_preflight" || die "LORO analysis failed"
  fi
  echo "===== LORO_PREFLIGHT_SUMMARY (hypotheses / coverage) ====="
  "$PY" - "$OUT/loro_preflight/summary.json" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
print(json.dumps({k: s.get(k) for k in ("hypotheses", "root_held_out_predictive_coverage_95")}, indent=1, ensure_ascii=False))
hl = max(int(r["adaptation"]) for r in s["method_table_cap192"])
for r in s["method_table_cap192"]:
    if int(r["adaptation"]) == hl: print(r)
PYEOF

  # ---------- 3. 冻结（脚本自判 gate） ----------
  if [ ! -f "$CODE/melting_v4_confirm.json" ]; then
    log "freeze confirm protocol"
    "$PY" "$CODE/freeze_confirm.py" --draft "$CODE/melting_v4_confirm.draft.json" \
        --pilot-report "$PR" --loro-summary "$OUT/loro_preflight/summary.json" \
        --calibration-protocol "$CODE/melting_v4_calibration.json" \
        --calibration-dir "$OUT" --out "$CODE/melting_v4_confirm.json" | tee "$CODE/freeze.log"
    grep -q FROZEN_CONFIRM_PROTOCOL "$CODE/freeze.log" || die "pre-flight gate FAILED or freeze error; see above. Do NOT tune anything."
  else
    log "confirm protocol already frozen: $(sha256sum "$CODE/melting_v4_confirm.json")"
  fi
  if [ "$FREEZE_ONLY" -eq 1 ]; then
    log "freeze-only mode: stop here. Launch confirmation shards with --confirm-only."
    echo "FREEZE_COMPLETE"; exit 0
  fi
fi

# ---------- 4. confirmation ----------
[ -f "$CODE/melting_v4_confirm.json" ] || die "melting_v4_confirm.json not frozen"
"$PY" -c "import json,sys; sys.exit(0 if json.load(open('$CODE/melting_v4_confirm.json'))['status']=='FROZEN' else 1)" || die "confirm protocol status != FROZEN"
read -r done total < <(n_complete "$CONF" "$CODE/melting_v4_confirm.json")
if [ "$done" -lt "$total" ]; then
  run_cohort "$CODE/melting_v4_confirm.json" "$CONF" "$CONF_WORKERS" "$CONF_MAX_SECONDS" "$SEED_SUBSET"
else
  log "confirmation already complete ($done/$total), skip"
fi
if [ -n "$SEED_SUBSET" ]; then
  log "shard done. rsync $CONF/seed_* to machine A, then run there:  bash run_melting_v4_pipeline.sh --confirm-only"
  echo "SHARD_COMPLETE"; exit 0
fi
read -r done total < <(n_complete "$CONF" "$CODE/melting_v4_confirm.json")
[ "$done" -eq "$total" ] || die "confirmation incomplete ($done/$total); if sharded, rsync other shard first"

# ---------- 5. confirm 分析（含作者 baselines） ----------
log "confirm analysis"
"$PY" "$CODE/analyze_v4.py" --mode confirm --protocol "$CODE/melting_v4_confirm.json" \
    --calibration "$OUT" --test "$CONF" --output "$CONF/analysis" || die "confirm analysis failed"
echo "===== CONFIRM_SUMMARY (hypotheses / long rows / coverage / author_baselines) ====="
"$PY" - "$CONF/analysis/summary.json" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
print(json.dumps({k: s.get(k) for k in ("hypotheses", "prespecified_success", "root_held_out_predictive_coverage_95", "author_baselines")}, indent=1, ensure_ascii=False))
hl = max(int(r["adaptation"]) for r in s["method_table_cap192"])
for r in s["method_table_cap192"]:
    if int(r["adaptation"]) == hl: print(r)
PYEOF
sha256sum "$CONF/analysis/summary.json" "$CONF/analysis/selection_seal.json" 2>/dev/null
log "PIPELINE DONE"
echo "PIPELINE_COMPLETE"
