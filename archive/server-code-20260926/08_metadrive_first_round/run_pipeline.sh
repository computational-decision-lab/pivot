#!/usr/bin/env bash
set -Eeuo pipefail
BASE=/root/autodl-tmp/metadrive_pivot_20260923
PY=/root/autodl-tmp/metadrive_env_20260923/bin/python
export PYTHONPATH="$BASE/code:$BASE/analysis_tools:/root/autodl-tmp/colin_pivot/src:/root/autodl-tmp/colin_pivot"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
trap 'rc=$?; echo "PIPELINE_ERROR rc=$rc line=$LINENO $(date -u +%FT%TZ)"' ERR
cd "$BASE"
echo "PIPELINE_STARTED $(date -u +%FT%TZ)"
# This wrapper only follows the existing pilot; it never starts a duplicate.
while :; do
 state=$($PY -c 'import json; print(json.load(open("pilot/status.json"))["status"])')
 if [[ "$state" == COMPLETE ]]; then break; fi
 if [[ "$state" == FAILED ]]; then echo 'PIPELINE_STOPPED pilot failed'; exit 2; fi
 if ! kill -0 2027720 2>/dev/null; then echo 'PIPELINE_STOPPED pilot process absent before completion'; exit 3; fi
 sleep 15
done
if ! $PY -c 'import json; assert json.load(open("checks/checks.json"))["checks"].get("saved_bank_replay_three_episodes",False)'; then
  $PY analysis_tools/replay_check.py --out pilot --checks checks/checks.json > checks/saved_bank_replay.log 2>&1
fi
if [[ ! -f pilot/pilot_report.json ]]; then
  $PY analysis_tools/report_pilot.py --out pilot --protocol code/pilot_protocol.json --checks checks/checks.json > pilot.analysis.log 2>&1
fi
$PY -c 'import json; d=json.load(open("pilot/pilot_report.json")); print("PILOT_GATES",json.dumps(d["pilot_checks"])); print("PILOT_GO",d["pilot_go"])'
go=$($PY -c 'import json; print(json.load(open("pilot/pilot_report.json"))["pilot_go"])')
if [[ "$go" != True ]]; then echo 'PIPELINE_STOPPED pilot_no_go; no additional seeds or parameter changes'; exit 0; fi
# All following protocol values were registered before viewing the pilot result.
$PY -c 'import json,author_adapter; print(author_adapter.self_test(json.load(open("code/pilot_protocol.json"))))'
$PY -u code/run_benchmark.py --protocol formal_protocols/calibration_protocol.json --out calibration --workers 16 > calibration.launch.log 2>&1
$PY analysis_tools/analyze_formal.py --mode calibration --calibration calibration --out calibration --protocol formal_protocols/calibration_protocol.json > calibration.analysis.log 2>&1
echo "CALIBRATION_COMPLETE $(date -u +%FT%TZ)"
sha256sum formal_protocols/*.json code/*.py analysis_tools/*.py calibration/analysis/frozen.json > checks/CONFIRMATION_FREEZE.sha256
echo "CONFIRMATION_FROZEN $(date -u +%FT%TZ)"
$PY -u code/run_benchmark.py --protocol formal_protocols/confirmation_protocol.json --out confirmation --workers 16 > confirmation.launch.log 2>&1
$PY analysis_tools/analyze_formal.py --mode confirmation --calibration calibration --out confirmation --protocol formal_protocols/confirmation_protocol.json > confirmation.analysis.log 2>&1
echo "PIPELINE_COMPLETE $(date -u +%FT%TZ)"
