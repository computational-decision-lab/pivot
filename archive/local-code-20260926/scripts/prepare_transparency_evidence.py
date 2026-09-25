#!/usr/bin/env python3
"""Collect selected immutable inputs; the submitted builder works offline."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def collect(root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    entries = []

    def put(data: bytes, destination: str, origin: str):
        target = output / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(data).hexdigest()
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f'Refusing to replace changed evidence: {destination}')
        target.write_bytes(data)
        entries.append(dict(path=destination, sha256=digest, bytes=len(data), origin=origin))

    def copy(source: Path, destination: str):
        put(source.read_bytes(), destination, str(source.relative_to(root)))

    core = root / 'artifacts/revision/iclr_20260918/source/07_下一步_PIVOTv2_异质响应'
    for name in ['leduc', 'kuhn']:
        for phase in ['calibration', 'confirm']:
            src = core / f'server_results/openspiel_v2_20260918/{name}_v2b_{phase}'
            for file in src.glob('seed_*/summary.json'):
                copy(file, f'core/{name}/{phase}/{file.parent.name}/summary.json')
            copy(core / f'protocols/openspiel_v2b_{name}_{phase}.json', f'core/{name}/{phase}/protocol.json')
        src = core / f'server_results/openspiel_v2_20260918/reanalysis_v12/{name}_v2b_confirm'
        for file in src.iterdir():
            if file.suffix in {'.json', '.csv'}:
                copy(file, f'core/{name}/analysis/{file.name}')
    for phase, sub in [('calibration', 'melting_v4_calibration_20260918'), ('analysis', 'melting_v4_confirm_analysis')]:
        src = core / 'server_results' / sub
        for file in src.rglob('*.json'):
            copy(file, f'core/melting/{phase}/{file.relative_to(src)}')
    for file in (core / 'protocols').glob('melting_v4*.json'):
        copy(file, f'core/melting/protocols/{file.name}')
    archive = core / 'server_results/melting_v4_hetero_20260918.tar.gz'
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            if ('melting_v4_confirm_20260918/seed_' in member.name and member.name.endswith('/summary.json') and member.isfile()):
                destination = f'core/melting/confirm/{PurePosixPath(member.name).parent.name}/summary.json'
                put(handle.extractfile(member).read(), destination, f'melting-v4-archive:{member.name}')
    for file in (core / 'code').rglob('*.py'):
        copy(file, f'source/core/{file.relative_to(core / "code")}')

    leduc = root / '实验结果/Leduc_研究成果_20260923'
    raw = leduc / '03_完整实验结果/root/autodl-tmp/openspiel_v4_20260921'
    for phase in ['pilot', 'calibration', 'confirm']:
        src = raw / f'leduc_v4_{phase}'
        for file in src.glob('seed_*/*'):
            if file.name in {'summary.json', 'status.json', 'bank.npz'}:
                copy(file, f'leduc_v4/{phase}/{file.parent.name}/{file.name}')
        copy(src / 'protocol.json', f'leduc_v4/{phase}/protocol.json')
    for file in (raw / 'leduc_v4_confirm/analysis').iterdir():
        if file.suffix in {'.json', '.csv'}:
            copy(file, f'leduc_v4/analysis/{file.name}')
    for file in (raw / 'leduc_v4_loro').glob('*.json'):
        copy(file, f'leduc_v4/loro/{file.name}')
    src = leduc / '05_代码与协议/Leduc_v4_原始执行包'
    for sub, glob in [('code', '*.py'), ('protocols', '*.json')]:
        for file in (src / sub).rglob(glob):
            copy(file, f'source/leduc_v4/{file.relative_to(src)}')
    for file in (leduc / '02_独立图表').glob('*.py'):
        copy(file, f'source/leduc_v4/figures/{file.name}')
    copy(leduc / '02_独立图表/plot_data.csv', 'leduc_v4/figure_table.csv')

    meta = root / '实验结果/MetaDrive_第一轮成果_20260923'
    archive = meta / '06_完整原始数据/metadrive_pivot_20260923_complete.tar.gz'
    selected_top = {'analysis_tools', 'author_source_snapshot', 'code', 'formal_protocols'}
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            p = PurePosixPath(member.name)
            parts = p.parts[1:]
            if not member.isfile() or len(parts) < 2 or '..' in parts:
                continue
            source = parts[0] in selected_top and p.suffix in {'.py', '.json', '.yaml', '.yml', '.txt'}
            evidence = (parts[0] in {'calibration', 'confirmation', 'pilot'} and
                        (parts[1] in {'roots', 'analysis'} or parts[-1] == 'protocol.json') and
                        p.suffix in {'.json', '.csv'})
            if source or evidence:
                put(handle.extractfile(member).read(), 'metadrive/' + '/'.join(parts),
                    'metadrive-original-archive:' + '/'.join(parts))

    highway = root / 'artifacts/scientific-audit/20260925-overleaf-9fef05d/highway/github-source'
    for file in (highway / 'evidence/highway').rglob('*.json'):
        copy(file, str(file.relative_to(highway / 'evidence')))
    for variant in ['highway', 'highway_redesign']:
        base = highway / 'reproduction' / variant
        for file in base.rglob('*'):
            if file.is_file() and file.suffix in {'.py', '.json', '.yaml', '.yml', '.txt', '.md'}:
                copy(file, f'source/{variant}/{file.relative_to(base)}')

    for study in ['e2c', 'e3c', 'e4c', 'e5c', 'e7c']:
        source = root / f'results/v9/{study}-confirmatory'
        for file in source.glob('*'):
            if file.is_file() and (file.suffix in {'.json', '.csv'} or file.name in {'group_metrics.jsonl.gz', 'trajectory_rows.jsonl.gz'}):
                copy(file, f'controlled/{study}/{file.name}')
        cfg = root / f'configs/v9/{study}.yaml'
        if cfg.exists(): copy(cfg, f'controlled/configs/{cfg.name}')
    for name in ['fig1_improvement_reversal', 'fig2_operator_shift', 'fig4_evidence_efficiency',
                 'fig5_closed_loop', 'figA_response_footprint', 'figC_posterior_robustness']:
        file = root / f'paper/figures/v10/{name}.csv'
        if file.exists(): copy(file, f'controlled/figure_tables/{file.name}')
    for folder in ['src/pivot', 'experiments/v9']:
        for file in (root / folder).rglob('*.py'):
            copy(file, f'source/controlled/{file.relative_to(root)}')
    public = [{k: v for k, v in row.items() if k != 'origin'} for row in entries]
    (output / 'manifest.json').write_text(json.dumps({'schema': 1, 'files': public}, indent=2) + '\n')
    return {'files': len(entries), 'bytes': sum(r['bytes'] for r in entries), 'origins': entries}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    info = collect(args.root.resolve(), args.output.resolve())
    origin_file = args.root / 'artifacts/scientific-audit/20260925-transparency-revision/evidence-origins.json'
    origin_file.write_text(json.dumps(info, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in info.items() if k != 'origins'}))
