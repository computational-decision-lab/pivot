"""Produce figures and a small evidence archive after the bounded batch ends."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parent
BATCH = ROOT / 'melting_response_sprint_20260916'
PY = '/root/autodl-tmp/benchmark_extensions/melting_env/bin/python'
deadline = datetime.fromisoformat('2026-09-16T11:27:00+00:00').timestamp()
while time.time() < deadline:
    try:
        state = json.loads((BATCH / 'status.json').read_text())
        if state['status'] != 'running':
            break
    except (OSError, ValueError, KeyError):
        pass
    time.sleep(15)
BATCH.mkdir(exist_ok=True)
env = os.environ.copy()
env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           MPLCONFIGDIR='/tmp/melting_response_sprint_mpl', MPLBACKEND='Agg')
analysis = BATCH / 'analysis'
try:
    result = subprocess.run([PY, str(ROOT / 'analyze_melting_response_sprint.py'),
                             '--batch-root', str(BATCH), '--output', str(analysis),
                             '--source-dir', str(ROOT)], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
    status = {'analysis_exit': result.returncode, 'analysis_output': result.stdout[-10000:]}
except Exception as exc:
    status = {'analysis_error': type(exc).__name__ + ': ' + str(exc)}
sources = BATCH / 'source_snapshot'
sources.mkdir(exist_ok=True)
for name in ('melting_response_sprint.py', 'run_melting_response_sprint_batch.py',
             'analyze_melting_response_sprint.py', 'finalize_melting_response_sprint.py',
             'melting_hierarchical_e5_stratified.py', 'melting_hierarchical_e5.py',
             'melting_reference_diagnostic.py', 'melting_behavior_diagnostic.py', 'melting_e5_adapter.py'):
    shutil.copy2(ROOT / name, sources / name)
status['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
(BATCH / 'finalization.json').write_text(json.dumps(status, indent=2) + '\n')
archive = ROOT / 'melting_response_sprint_20260916_results.tar.gz'
with tarfile.open(archive, 'w:gz') as handle:
    handle.add(BATCH, arcname=BATCH.name)
status.update(archive=str(archive), archive_bytes=archive.stat().st_size,
              archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
(ROOT / 'melting_response_sprint_20260916_finalization.json').write_text(json.dumps(status, indent=2) + '\n')
print(json.dumps(status), flush=True)
