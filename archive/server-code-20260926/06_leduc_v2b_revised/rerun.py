import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

RUN = Path('/root/autodl-tmp/leduc_fix_20260920')
CODE = RUN / 'code'
DATA = Path('/root/autodl-tmp/openspiel_v2_20260918')
CAL = DATA / 'leduc_v2b_calibration'
TEST = DATA / 'leduc_v2b_confirm'
PY = '/root/autodl-tmp/openspiel_env/bin/python'
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    PYTHONPATH=f'{CODE}:/root/autodl-tmp/colin_pivot/src:/root/autodl-tmp/colin_pivot')
sys.path[:0] = [str(CODE), '/root/autodl-tmp/colin_pivot/src', '/root/autodl-tmp/colin_pivot']

def now():
    return datetime.datetime.now().astimezone().isoformat()

started = now()

def status(stage, **extra):
    d = dict(stage=stage, started_at=started, updated_at=now(), **extra)
    (RUN / 'status.json').write_text(json.dumps(d, indent=2))
    print(json.dumps(d), flush=True)

def command(args, logfile):
    print('COMMAND', args, flush=True)
    with (RUN / logfile).open('w') as log:
        subprocess.run(args, cwd=CODE, env=os.environ, stdout=log,
                       stderr=subprocess.STDOUT, check=True)

try:
    status('unit_tests')
    command([PY, '-m', 'unittest', 'test_pivot_v2', '-v'], 'unit_tests.log')
    import numpy as np
    import analyze_v4 as fixed
    spec = importlib.util.spec_from_file_location('original_analyze', CODE / 'analyze_v4.before.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    protocol = json.loads((CAL / 'protocol.json').read_text())
    fixed.set_candidate_geometry(protocol)
    assert np.array_equal(fixed.author_features(9)[0], [0, 0])
    assert np.array_equal(fixed.author_features(9)[4], [2, 2])
    train = fixed.load_cohort(CAL, [1, 8])
    assert sorted(w['root'] for w in train) == list(range(95000, 95012))
    for h in [1, 8]:
        post = fixed.fit_spec(train, h)
        assert np.array_equal(post.R, np.maximum(post.notes['raw_R'], 1e-8))
    for directory, seeds in [(CAL, range(95000, 95012)), (TEST, range(96000, 96030))]:
        assert sorted(int(d.name[5:]) for d in directory.glob('seed_*') if d.is_dir()) == list(seeds)
        for seed in seeds:
            assert json.loads((directory / f'seed_{seed}/status.json').read_text())['status'] == 'complete'
    mp = {'candidate_probabilities': [.125, .225, .325, .425, .575, .675, .775, .875]}
    fixed.set_candidate_geometry(mp)
    original.set_candidate_geometry(mp)
    assert np.array_equal(fixed.author_features(8), original.author_features(8))
    rng = np.random.default_rng(11)
    worlds = []
    for _ in range(6):
        proxy = rng.normal(size=8)
        a = proxy + rng.normal(size=8)
        b = a + rng.normal(0, .1, size=8)
        worlds.append({'proxy': proxy, '_audit': {1: {'label': (a+b)/2, 'A': a, 'B': b}},
                       'per_h': {1: {'S': a+rng.normal(0, .2, size=8)}}})
    assert fixed.fit_spec(worlds, 1).__dict__ == original.fit_spec(worlds, 1).__dict__
    fixed.set_candidate_geometry(protocol)
    assert 'error' not in fixed.load_author(), fixed.AUTHOR
    files = [p for d in [CAL, TEST] for p in d.rglob('*') if p.is_file()
             and p.name in ['protocol.json', 'summary.json', 'status.json']]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (RUN / 'input_hashes.json').write_text(json.dumps(hashes, indent=2))
    (RUN / 'regression_checks.json').write_text(json.dumps(dict(
        leduc_zero_anchor=True, leduc_noise_from_calibration=True,
        other_environment_features_and_posterior_unchanged=True,
        exact_roots_12_and_30=True, author_import=True), indent=2))
    print('REGRESSION_OK', flush=True)
    status('loro')
    command([PY, str(CODE / 'analyze_v4.py'), '--mode', 'loro', '--protocol',
             str(CAL / 'protocol.json'), '--calibration', str(CAL), '--output',
             str(RUN / 'loro')], 'loro.log')
    status('confirm_replay')
    command([PY, str(CODE / 'analyze_v4.py'), '--mode', 'confirm', '--protocol',
             str(TEST / 'protocol.json'), '--calibration', str(CAL), '--test', str(TEST),
             '--output', str(RUN / 'confirm')], 'confirm.log')
    for phase, seeds in [('loro', range(95000, 95012)), ('confirm', range(96000, 96030))]:
        directory = RUN / phase
        summary = json.loads((directory / 'summary.json').read_text())
        rows = json.loads((directory / 'scored_decisions.json').read_text())
        seal = json.loads((directory / 'selection_seal.json').read_text())
        assert summary['n_roots'] == len(seeds)
        assert seal['test_roots'] == list(seeds)
        assert summary['author_baselines'] == 'included'
        assert not any(str(r['stop_reason']).startswith('author_error') for r in rows)
        assert all(r['cap'] is None or r['hf_episode_cost'] <= r['cap'] for r in rows)
        assert hashlib.sha256((directory / 'decisions_sealed.json').read_bytes()).hexdigest() == seal['decisions_sha256']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in hashes.items())
    status('complete', regression_ok=True, original_inputs_unchanged=True, new_rollouts=0)
except BaseException:
    status('failed', traceback=traceback.format_exc())
    raise
