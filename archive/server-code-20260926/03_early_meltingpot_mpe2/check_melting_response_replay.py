"""Two extra native rollouts check exact replay; excluded from study estimates."""
import hashlib
import json
import math
from pathlib import Path
import time
from melting_reference_diagnostic import OfficialSpecialistRollout

ROOT = Path(__file__).resolve().parent
BATCH = ROOT / 'melting_response_sprint_20260916'
PANEL = BATCH / 'seed_42000'
metadata = json.loads((PANEL / 'protocol.json').read_text())['backend']
rows = []
for path in (PANEL / 'episodes').glob('*.json'):
    if path.name.endswith('.events.json'):
        continue
    row = json.loads(path.read_text())
    if row['signature']['key'] in (['training', 0, 0], ['training', 0, 1]):
        rows.append((path, row))
rows.sort(key=lambda item: item[1]['signature']['key'])
if len(rows) != 2:
    raise RuntimeError('Two planned replay originals are not available')
backend = OfficialSpecialistRollout(Path(metadata['model_path']), task='melting_stag', horizon=2500,
                                   crossbench_root=Path('/root/autodl-tmp/colin_melting'))
checks = []
try:
    for path, original in rows:
        sig = original['signature']
        replay, events = backend.episode(sig['matchup'], sig['environment_seed'], sig['policy_seeds'],
                                         deadline=time.monotonic()+120, allowance=2500)
        match = (replay['complete'] and replay['trajectory_sha256'] == original['trajectory_sha256']
                 and replay['env_steps'] == original['env_steps']
                 and all(math.isclose(replay[key], original[key], rel_tol=1e-12, abs_tol=1e-12)
                         for key in ('focal_return', 'response_return')))
        checks.append({'original_path': str(path.relative_to(BATCH)),
                       'original_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'match': match, 'original_trajectory': original['trajectory_sha256'],
                       'replay_trajectory': replay['trajectory_sha256'],
                       'original_frames': original['env_steps'], 'replay_frames': replay['env_steps'],
                       'original_returns': [original['focal_return'], original['response_return']],
                       'replay_returns': [replay['focal_return'], replay['response_return']],
                       'replay_complete': replay['complete']})
finally:
    backend.close()
result = {'status': 'PASS' if all(row['match'] for row in checks) else 'FAIL',
          'additional_native_replay_episodes': len(checks),
          'additional_native_frames': sum(row['replay_frames'] for row in checks),
          'used_in_training_or_effect_estimation': False, 'checks': checks}
(BATCH / 'replay_check.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
