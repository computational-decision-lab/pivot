#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/colin_pivot_cloud
export PYTHONPATH=/root/autodl-tmp/benchmark_extensions/meltingpot_source:/root/autodl-tmp/colin_melting:/root/autodl-tmp/colin_pivot/src:/root/autodl-tmp/colin_pivot:/root/autodl-tmp/colin_pivot_cloud
export CUDA_VISIBLE_DEVICES='' TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=3 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
py=/root/autodl-tmp/benchmark_extensions/melting_env/bin/python
"$py" test_sequential_pivot_extension.py
"$py" test_melting_method_native_v3.py
"$py" - <<'PY'
from pathlib import Path
import datetime,json,hashlib
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
mechanism=Path('melting_mechanism_20260917')
gate=json.loads((mechanism/'analysis/summary.json').read_text());assert gate['status']=='PASS'
opportunity=json.loads((mechanism/'analysis/decision_relevance.json').read_text());assert opportunity['effects']['32']['ci95'][0]>0
draft=Path('protocols/melting_method_v3_20260917_draft.json');s=json.loads(draft.read_text())
replay=json.loads((mechanism/'replay_gate.json').read_text())
names=['melting_method_native_v3.py','sequential_pivot_extension.py','verify_melting_method_v3.py','run_melting_method_v3.py','analyze_melting_method_v3.py','plot_melting_method_v3.py','verify_melting_mechanism.py','audit_melting_short_long.py','test_sequential_pivot_extension.py','test_melting_method_native_v3.py','launch_melting_method_v3_20260917.sh']
names+=list(json.loads((mechanism/'protocol.json').read_text())['source_hashes_before_replay_fix'])
sources=[Path(name).resolve() for name in names]+list(Path('/root/autodl-tmp/colin_pivot/src').rglob('*.py'))+list(Path('/root/autodl-tmp/colin_pivot/experiments').rglob('*.py'))
calibration=[(mechanism/f'seed_{seed}/results.json').resolve() for seed in range(43000,43012)]+[(mechanism/'analysis/summary.json').resolve(),(mechanism/'analysis/decision_relevance.json').resolve()]
s.update(status='FROZEN',frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prepared_protocol_sha256=sha(draft),
    mechanism_gate_sha256=sha(mechanism/'analysis/summary.json'),runtime_file_hashes=replay['runtime_file_hashes'],
    source_file_hashes={str(p):sha(p) for p in sorted(set(sources))},calibration_file_hashes={str(p):sha(p) for p in calibration},
    author_reference_commit='693585913c17ff8359739e7dada633b65e41f0ba',
    native_runtime_extension='Stable avatar identity order and equal-priority updater order; full diagnostic traces retained.')
target=Path('protocols/melting_method_v3_20260917_frozen.json');assert not target.exists();target.write_text(json.dumps(s,indent=2)+'\n')
print('FROZEN_METHOD_PROTOCOL',sha(target),flush=True)
PY
model=/root/autodl-tmp/colin_pivot_cloud/melting_recovery_20260915/reference_smoke/reference_assets/assets/saved_models/stag_hunt_in_the_matrix__repeated/puppet_1
batch=/root/autodl-tmp/colin_pivot_cloud/melting_method_confirm_20260917
"$py" run_melting_method_v3.py --protocol protocols/melting_method_v3_20260917_frozen.json --output "$batch" --model-path "$model" --crossbench-root /root/autodl-tmp/colin_melting --calibration /root/autodl-tmp/colin_pivot_cloud/melting_mechanism_20260917 --workers 30 --max-seconds 14400
"$py" analyze_melting_method_v3.py --batch "$batch" --output "$batch/analysis"
"$py" plot_melting_method_v3.py --analysis "$batch/analysis"
