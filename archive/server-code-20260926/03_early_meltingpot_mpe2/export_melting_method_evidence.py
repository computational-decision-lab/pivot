"""Export completed formal evidence without models or verbose native event logs."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.batch.resolve();status=json.loads((root/'status.json').read_text())
    assert status['status']=='complete' and status['frozen_inputs_unchanged']
    analysis=json.loads((root/'analysis/summary.json').read_text());assert analysis['seed_count']==30
    paths=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.name.endswith('.events.json') or path.suffix not in ['.json','.csv','.png','.pdf']:continue
        paths.append(path)
    manifest={'scope':'Complete formal JSON evidence, analyses and figures; verbose per-event logs, process logs, model weights and runtime excluded. Full originals remain on server.',
              'protocol_sha256':analysis['protocol_sha256'],'native_episodes':analysis['accounting']['native_episodes'],
              'files':[{'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]}
    manifest_path=root/'delivery_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    assert not a.output.exists(),'Do not overwrite an existing evidence bundle'
    with tarfile.open(a.output,'w:gz',compresslevel=6) as archive:
        for path in [*paths,manifest_path]:archive.add(path,arcname=str(Path(root.name)/path.relative_to(root)),recursive=False)
    print(json.dumps({'bundle':str(a.output),'bytes':a.output.stat().st_size,'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'files':len(paths)+1}))


if __name__=='__main__':main()
