#!/usr/bin/env bash
set -Eeuo pipefail
BASE=/root/autodl-tmp/metadrive_precision_20260923
PY=/root/autodl-tmp/metadrive_env_20260923/bin/python
export PYTHONPATH="$BASE/code:$BASE/analysis_tools:/root/autodl-tmp/colin_pivot/src:/root/autodl-tmp/colin_pivot"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
trap 'rc=$?; echo "PIPELINE_ERROR rc=$rc line=$LINENO $(date -u +%FT%TZ)"' ERR
cd "$BASE"
echo "PIPELINE_STARTED second_independent_precision_cohort $(date -u +%FT%TZ)"
$PY -c 'import metadrive, native_adapter,run_benchmark,selector_analysis,pivot_v2,author_adapter,analyze_formal; print("IMPORT_OK")'
$PY analysis_tools/precision_preflight.py --base "$BASE" | tee checks/runtime_preflight.log
# No pilot and no outcome-based branch; exactly the frozen calibration roots.
$PY -u code/run_benchmark.py --protocol formal_protocols/calibration_protocol.json --out calibration --workers 16 > calibration.launch.log 2>&1
$PY analysis_tools/analyze_formal.py --mode calibration --calibration calibration --out calibration --protocol formal_protocols/calibration_protocol.json > calibration.analysis.log 2>&1
echo "CALIBRATION_COMPLETE $(date -u +%FT%TZ)"
sha256sum formal_protocols/*.json code/*.py analysis_tools/*.py calibration/analysis/frozen.json > checks/CONFIRMATION_FREEZE.sha256
sha256sum -c SOURCE_PROTOCOL_SHA256SUMS.txt > checks/preconfirmation_source_check.log 2>&1
echo "CONFIRMATION_FROZEN $(date -u +%FT%TZ)"
$PY -u code/run_benchmark.py --protocol formal_protocols/confirmation_protocol.json --out confirmation --workers 16 > confirmation.launch.log 2>&1
$PY analysis_tools/analyze_formal.py --mode confirmation --calibration calibration --out confirmation --protocol formal_protocols/confirmation_protocol.json > confirmation.analysis.log 2>&1
$PY analysis_tools/precision_decision.py --base "$BASE" | tee confirmation.precision_decision.log
echo "PIPELINE_COMPLETE $(date -u +%FT%TZ)"
