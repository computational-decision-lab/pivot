"""Import single-sided edits using the hash baseline shared with forward sync."""
import argparse
import hashlib
import json
from pathlib import Path

ALLOWED = {'.tex', '.bib', '.sty', '.bst', '.cls', '.pdf', '.png', '.jpg', '.jpeg', '.eps'}
SKIP = {'archive', 'build', 'supplementary', 'snapshot', '.git'}


def eligible(name):
    p = Path(name)
    return (not p.is_absolute() and '..' not in p.parts
            and not any(x in SKIP for x in p.parts)
            and p.suffix in ALLOWED
            and not p.name.startswith('pivot_iclr2027_submission'))


def read(path):
    if path.is_symlink():
        raise ValueError(f'Symlink is not supported: {path}')
    return path.read_bytes() if path.is_file() else None


def digest(data):
    return hashlib.sha256(data).hexdigest() if data is not None else None


def canonical(name, data):
    if data is not None and name == 'main.tex':
        return data.replace(b'../tables/', b'tables/')
    return data


def import_edits(remote, paper):
    state_path = remote / '.github-sync.json'
    state = json.loads(state_path.read_text())
    baseline = state['files']
    # Never import build configuration from Overleaf as executable code.
    if digest(read(remote / 'latexmkrc')) != baseline.get('latexmkrc'):
        raise ValueError('Overleaf latexmkrc changed; manual reconciliation required')
    names = {n for n in baseline if eligible(n)}
    names.update(p.relative_to(remote).as_posix() for p in remote.rglob('*')
                 if eligible(p.relative_to(remote).as_posix()) and p.is_file())
    changes = []
    for name in sorted(names):
        # Reject parent symlinks before any read/write.
        for root in (paper, remote):
            target = root / name
            if any(p.is_symlink() for p in [target, *target.parents] if p != root.parent):
                raise ValueError(f'Symlink path: {name}')
        data = read(remote / name)
        old_hash = baseline.get(name)
        remote_hash = digest(data)
        if remote_hash == old_hash:
            continue  # GitHub-only edits are left for forward sync.
        local = read(paper / name)
        local_hash = digest(canonical(name, local))
        if local_hash not in (old_hash, remote_hash):
            raise ValueError(f'Both sides changed {name}; reconcile manually')
        if name == 'main.tex' and data is None:
            raise ValueError('Refusing to delete main.tex')
        changes.append((name, data, local, local_hash == remote_hash))
    # Validate every file before modifying either tree.
    for name, data, local, identical in changes:
        path = paper / name
        if not identical:
            if data is None:
                path.unlink()
            else:
                if name == 'main.tex':
                    data = data.replace(b'../tables/', b'tables/').replace(b'tables/', b'../tables/')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        remote_data = read(remote / name)
        if remote_data is None:
            baseline.pop(name, None)
        else:
            baseline[name] = digest(remote_data)
    if changes:
        state_path.write_text(json.dumps(state, indent=2) + '\n')
    return len(changes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('remote', type=Path)
    parser.add_argument('--paper', type=Path, default=Path('paper'))
    args = parser.parse_args()
    print(f'Overleaf changes reconciled: {import_edits(args.remote, args.paper)}')
