"""Run a fixed three-checkpoint development diagnostic; no training."""
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path('/root/autodl-tmp/pivot_author_20260915/colin_pivot_cloud')
OUT = BASE / 'melting_recovery_20260915'
PY = '/root/autodl-tmp/benchmark_extensions/melting_env/bin/python'
STUDY = Path('/root/autodl-tmp/colin_melting/runs/melting_cross_main/melting_stag/test')


def one(seed):
    env = os.environ.copy()
    env.update(PYTHONPATH='/root/autodl-tmp/colin_melting/vendor/meltingpot_source:/root/autodl-tmp/colin_melting',
               CUDA_VISIBLE_DEVICES='', TF_CPP_MIN_LOG_LEVEL='3', TF_ENABLE_ONEDNN_OPTS='0')
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS']:
        env[key] = '1'
    cmd = [PY, str(BASE / 'melting_behavior_diagnostic.py'),
           '--crossbench-root', '/root/autodl-tmp/colin_melting',
           '--run-dir', str(STUDY / f'seed_{seed}'), '--output', str(OUT / f'quality_{seed}'),
           '--seed', str(202609150 + seed), '--horizons', '1000,native', '--episodes', '32',
           '--max-seconds', '600', '--max-env-steps', '384000']
    begin = time.monotonic()
    with (OUT / f'quality_{seed}.log').open('w') as log:
        p = subprocess.Popen(cmd, env=env, cwd='/root/autodl-tmp/colin_melting',
                             stdout=log, stderr=log, stdin=subprocess.DEVNULL)
        try:
            code = p.wait(timeout=630)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()
            code = -1
    return {'checkpoint_seed': seed, 'returncode': code, 'seconds': time.monotonic()-begin,
            'new_training_steps': 0, 'command': cmd}


def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    protocol = {'started_at': started, 'checkpoint_seeds': [1300,1301,1302],
                'role': 'development_quality_gate', 'new_training_steps': 0,
                'episodes_per_policy_per_horizon': 32,
                'primary_horizon': 1000, 'native_horizon': 'separate_extrapolation_diagnostic',
                'historical_checkpoint_selection': 'first_three_in_prespecified_old_seed_order',
                'fresh_evaluation_seeds': True, 'formal_benchmark_complete': False}
    (OUT/'quality_batch_protocol.json').write_text(json.dumps(protocol, indent=2))
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for future in concurrent.futures.as_completed([pool.submit(one, s) for s in [1300,1301,1302]]):
            row=future.result(); rows.append(row)
            (OUT/'quality_batch_status.json').write_text(json.dumps({'jobs': rows,
                    'complete':len(rows)==3, 'all_passed_execution':all(x['returncode']==0 for x in rows)},indent=2))
            print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
