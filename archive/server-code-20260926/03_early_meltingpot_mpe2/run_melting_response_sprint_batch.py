"""Bounded orchestration for the 2026-09-16 native Melting Pot response study."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PY = '/root/autodl-tmp/benchmark_extensions/melting_env/bin/python'
MODEL = ROOT / 'melting_recovery_20260915/reference_smoke/reference_assets/assets/saved_models/stag_hunt_in_the_matrix__repeated/puppet_1'
SEEDS = list(range(42000, 42012))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline-utc', default='2026-09-16T11:25:00+00:00')
    parser.add_argument('--max-seconds', type=int, default=2400)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    deadline = min(datetime.fromisoformat(args.deadline_utc).timestamp(), time.time() + args.max_seconds)
    files = ['melting_response_sprint.py', 'melting_hierarchical_e5_stratified.py',
             'melting_hierarchical_e5.py', 'melting_reference_diagnostic.py',
             'melting_behavior_diagnostic.py', 'melting_e5_adapter.py', Path(__file__).name]
    protocol = {
        'study': 'Native Melting Pot learned responder horizon mechanism extension',
        'registered_at_utc': now(), 'seed_list': SEEDS, 'seeds': SEEDS, 'workers': 12,
        'task': 'stag_hunt_in_the_matrix__repeated',
        'focal_probability_probes': [.25, .5, .75], 'initial_response_probability': .5,
        'response_train_episodes_per_probe': 32, 'checkpoints': [0, 4, 16, 32],
        'evaluation_blocks': ['A', 'B'], 'evaluation_episodes_per_stratum': 4,
        'native_episodes_per_seed': 128, 'planned_native_episodes': 1536,
        'seed_independence_unit': 'training/environment random seed, conditional on one fixed low-level official network',
        'paired_probe_training_templates': True,
        'evaluation_independent_of_training': True,
        'primary_diagnostic': '32-episode responder own-return gain averaged over outer probes p=.25,.75 and A/B within seed',
        'response_direction_diagnostic': 'q32(p=.75) minus q32(p=.25) within seed',
        'secondary': 'all fixed probe learning curves, A/B agreement, proxy versus responsive deployment probe gains',
        'statistics': 'paired seed-level bootstrap 95% intervals; descriptive development study, not confirmatory PIVOT superiority',
        'scope': 'One official Melting Pot task; real REINFORCE trains episode-level skill mixture only; frozen low-level network; preset focal probes; no PIVOT selection comparison in this study',
        'outcome_independent_stop': 'Fixed twelve seeds, no replacement/extension based on outcomes; terminate at time cap; preserve failures and partial seeds',
        'deadline_utc': datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
        'hardware_cpu_quota': 25, 'hardware_memory_gib': 92,
        'gpu_or_llm_calls': False,
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
    }
    write(output / 'protocol.json', protocol)
    state = {'status': 'running', 'started_at_utc': now(), 'jobs': []}
    write(output / 'status.json', state)
    env = os.environ.copy()
    env.update(PYTHONPATH=':'.join([
        str(ROOT.parent / 'colin_pivot/src'), str(ROOT.parent / 'colin_pivot'), str(ROOT),
        '/root/autodl-tmp/colin_melting/vendor/meltingpot_source', '/root/autodl-tmp/colin_melting']),
        CUDA_VISIBLE_DEVICES='', TF_CPP_MIN_LOG_LEVEL='3', TF_ENABLE_ONEDNN_OPTS='0')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'

    def execute(seed):
        begin = time.monotonic()
        remaining = max(1, int(deadline - time.time()))
        cmd = [PY, str(ROOT / 'melting_response_sprint.py'), '--seed', str(seed),
               '--output', str(output / f'seed_{seed}'), '--model-path', str(MODEL),
               '--crossbench-root', '/root/autodl-tmp/colin_melting',
               '--train-episodes', '32', '--eval-episodes-per-stratum', '4',
               '--max-seconds', str(remaining)]
        with (output / f'seed_{seed}.log').open('w') as handle:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            timed_out = False
            try:
                code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    code = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    code = process.wait()
        result = {'seed': seed, 'returncode': code, 'timeout': timed_out,
                  'elapsed_seconds': time.monotonic() - begin, 'command': cmd,
                  'finished_at_utc': now(), 'output': str(output / f'seed_{seed}')}
        write(output / f'seed_{seed}.job.json', result)
        return result

    with ThreadPoolExecutor(max_workers=12) as pool:
        for future in as_completed([pool.submit(execute, seed) for seed in SEEDS]):
            result = future.result()
            state['jobs'].append(result)
            write(output / 'status.json', state)
            print(json.dumps(result), flush=True)
    state['status'] = 'complete' if all(j['returncode'] == 0 and not j['timeout'] for j in state['jobs']) else 'incomplete_or_failed'
    state['finished_at_utc'] = now()
    write(output / 'status.json', state)


if __name__ == '__main__':
    main()
