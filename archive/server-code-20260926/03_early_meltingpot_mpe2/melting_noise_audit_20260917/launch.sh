#!/bin/bash
set -Eeuo pipefail
base=/root/autodl-tmp
code=$base/colin_pivot_cloud/melting_noise_audit_20260917
out=$base/melting_noise_audit_results_20260917
export PYTHONPATH=$base/benchmark_extensions/meltingpot_source:$base/colin_melting:$base/colin_pivot/src:$base/colin_pivot:$base/colin_pivot_cloud:$code
export CUDA_VISIBLE_DEVICES='' TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=3 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
py=$base/benchmark_extensions/melting_env/bin/python
cd "$code"
"$py" - <<'PY'
import numpy as np
from run_audit import noise_metrics,weights
import matplotlib
# Identity, covariance sign, and equal-physical-cost bookkeeping.
rng=np.random.default_rng(1909); x=rng.normal(size=(1000,4))
assert np.isclose(weights(.4,.7).sum(),1)
r=noise_metrics(x,.5,.7,.7);assert r['paired_variance']==0 and r['independent_variance']>0
r=noise_metrics(x,.8,.2,.7);assert r['covariance_identity_error']<1e-9
assert r['independent_equal_cost_variance']==2*r['independent_variance']
print('NOISE_ESTIMATOR_CHECKS_PASS',flush=True)
PY
"$py" -m unittest test_sequential_pivot_extension test_melting_method_native_v3
model=$base/colin_pivot_cloud/melting_recovery_20260915/reference_smoke/reference_assets/assets/saved_models/stag_hunt_in_the_matrix__repeated/puppet_1
exec "$py" run_audit.py --protocol "$code/protocol.json" --output "$out" --model "$model" --crossbench "$base/colin_melting"
