#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/colin_pivot_cloud
export PYTHONPATH=/root/autodl-tmp/benchmark_extensions/meltingpot_source:/root/autodl-tmp/colin_melting:/root/autodl-tmp/colin_pivot/src:/root/autodl-tmp/colin_pivot:/root/autodl-tmp/colin_pivot_cloud
export CUDA_VISIBLE_DEVICES='' TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=3 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
py=/root/autodl-tmp/benchmark_extensions/melting_env/bin/python
"$py" - <<'PY'
import datetime,hashlib,json
from pathlib import Path
p=Path('protocols/melting_short_long_20260917.json')
s=json.loads(p.read_text());r=json.loads(Path('replay_avatar_updater_v2_20260917/replay_gate.json').read_text());assert r['status']=='PASS'
s.update(status='FROZEN_BEFORE_FRESH_COHORT',frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runtime_file_hashes=r['runtime_file_hashes'],
    native_determinism_patch=[json.loads(Path('determinism_patch_20260917/'+name+'.json').read_text()) for name in ['avatars','updaters']],
    patch_scope='Adaptive runtime extension: stable avatar identity order and equal-priority updater dispatch order. Historical outcomes not reused.',
    prepared_protocol_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
target=Path('protocols/melting_short_long_20260917_frozen.json');assert not target.exists();target.write_text(json.dumps(s,indent=2)+'\n')
print('FROZEN_PROTOCOL',hashlib.sha256(target.read_bytes()).hexdigest(),flush=True)
PY
model=/root/autodl-tmp/colin_pivot_cloud/melting_recovery_20260915/reference_smoke/reference_assets/assets/saved_models/stag_hunt_in_the_matrix__repeated/puppet_1
batch=/root/autodl-tmp/colin_pivot_cloud/melting_mechanism_20260917
args=(--protocol protocols/melting_short_long_20260917_frozen.json --replay-gate replay_avatar_updater_v2_20260917/replay_gate.json --model-path "$model" --crossbench-root /root/autodl-tmp/colin_melting --output "$batch" --workers 12 --max-seconds 7200)
"$py" run_melting_mechanism_gate.py "${args[@]}" --plan-only > mechanism_launch_plan_20260917.json
"$py" run_melting_mechanism_gate.py "${args[@]}"
"$py" analyze_melting_mechanism_gate.py --batch "$batch" --sources . --output "$batch/analysis"
"$py" plot_melting_mechanism_gate.py --analysis "$batch/analysis"
