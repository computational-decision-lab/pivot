"""Finish analysis and compact archival after the one authorized batch ends.

Does not train, resume, replace seeds, alter the frozen protocol, or shut down.
The separately armed cloud watchdog remains responsible for the time limit.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path('/root/autodl-tmp/pivot_author_20260915/colin_pivot_cloud')
AUTHOR = ROOT.parent / 'colin_pivot'
STUDY = ROOT / 'melting_recovery_20260915'
BATCH = STUDY / 'hierarchical_main_v2'
DEST = STUDY / 'completed_main_package'
DEADLINE = dt.datetime.fromisoformat('2026-09-15T16:43:00+00:00').timestamp()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(status, **extra):
    value = {'status': status, 'updated_at': dt.datetime.now(dt.timezone.utc).isoformat(), **extra}
    path = STUDY / 'finalization_status.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)
    print(json.dumps(value), flush=True)


def run(args, log):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               MPLBACKEND='Agg', PYTHONPATH=f'{AUTHOR / "src"}:{AUTHOR}:{ROOT}')
    with log.open('w') as out:
        subprocess.run([sys.executable, *map(str, args)], check=True, stdout=out,
                       stderr=subprocess.STDOUT, env=env,
                       timeout=max(1, min(240, DEADLINE-time.time())))


def main():
    state('waiting_for_complete_batch')
    while time.time() < DEADLINE:
        status = json.loads((BATCH / 'status.json').read_text())['status']
        if status == 'complete':
            break
        if status != 'running':
            state('batch_not_complete', batch_status=status)
            return 2
        time.sleep(5)
    else:
        state('finalization_deadline_reached')
        return 2
    DEST.mkdir(exist_ok=False)
    state('validating_all_raw_records')
    run([ROOT/'analyze_melting_hierarchical.py', '--batch-root', BATCH,
         '--output', DEST/'analysis', '--source-dir', ROOT, '--author-root', AUTHOR],
        DEST/'analysis.log')
    state('rendering_registered_figures')
    run([ROOT/'plot_melting_recovery.py', '--analysis', DEST/'analysis',
         '--output', DEST/'figures'], DEST/'figures.log')
    state('archiving_compact_evidence')
    archive = DEST/'compact_evidence.tar.gz'
    members = []
    with tarfile.open(archive, 'w:gz', compresslevel=6) as tar:
        for path in sorted(BATCH.rglob('*.json')):
            if path.name.endswith('.events.json') or not path.is_file() or path.is_symlink():
                continue
            name = str(Path('batch') / path.relative_to(BATCH))
            tar.add(path, arcname=name, recursive=False)
            members.append({'path': name, 'bytes': path.stat().st_size, 'sha256': digest(path)})
        protocol = json.loads((BATCH/'protocol.json').read_text())
        sources = list(protocol['sources_sha256']) + [
            'analyze_melting_hierarchical.py', 'plot_melting_recovery.py', 'finalize_melting_batch.py']
        for name in sources:
            path = ROOT / name
            tar.add(path, arcname='source/' + name, recursive=False)
            members.append({'path': 'source/' + name, 'bytes': path.stat().st_size, 'sha256': digest(path)})
        for name in ('experiments/v9/e5c_efficiency.py', 'src/pivot/acquisition/pivot_voi.py'):
            tar.add(AUTHOR/name, arcname='author_source/' + name, recursive=False)
        wrapper = Path('/root/autodl-tmp/colin_melting/crossbench/melting.py')
        if wrapper.is_file():
            tar.add(wrapper, arcname='source/crossbench/melting.py', recursive=False)
    if archive.stat().st_size > 8*1024*1024:
        state('archive_exceeds_local_size_target', archive_bytes=archive.stat().st_size)
        return 2
    (DEST/'evidence_manifest.json').write_text(json.dumps({
        'archive_sha256': digest(archive), 'archive_bytes': archive.stat().st_size,
        'members': members, 'note': 'Actual episode summaries and frozen records, excluding raw event streams, models, credentials and large logs.'
    }, indent=2) + '\n')
    package = STUDY/'completed_main_package.tar.gz'
    with tarfile.open(package, 'w:gz', compresslevel=6) as tar:
        tar.add(DEST, arcname='main', recursive=True)
    state('ready_for_download', package=str(package), bytes=package.stat().st_size,
          sha256=digest(package), raw_records=len(members),
          analysis_sha256=digest(ROOT/'analyze_melting_hierarchical.py'),
          plotting_sha256=digest(ROOT/'plot_melting_recovery.py'))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        state('finalization_failed', error=type(exc).__name__+': '+str(exc))
        raise
