"""Require repeatable native traces across process history before cohort launch."""
from __future__ import annotations
import argparse,gzip,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native-source',type=Path,required=True);p.add_argument('--crossbench-root',type=Path,required=True);p.add_argument('--dependency-freeze',type=Path,required=True)
    a=p.parse_args();reports=[a.root/'process_1/summary.json',a.root/'process_2/summary.json'];groups={};checks=[];n=0;models=[]
    for path in reports:
        s=json.loads(path.read_text());assert s['status']=='complete';models.append(s['model']['model_file_sha256'])
        for r in s['records']:
            assert r['rollout']['complete'];n+=1
            trace_file=path.parent/r['trace_file'];assert sha(trace_file)==r['trace_sha256']
            with gzip.open(trace_file,'rt') as f:trace=json.load(f)
            case=r['case'];previous=groups.get(case)
            if previous is not None:
                old,oldpath=previous;idx=next((i for i,(x,y) in enumerate(zip(old,trace)) if x!=y),None)
                checks.append({'case':case,'reference':oldpath,'current':str(trace_file),'equal':old==trace,
                    'first_difference':None if idx is None else {'index':idx,'reference':old[idx],'current':trace[idx]},'lengths':[len(old),len(trace)]})
            else:groups[case]=(trace,str(trace_file))
    assert n>=8 and len(checks)>=5 and all(m==models[0] for m in models)
    names=['melting_reference_diagnostic.py','melting_behavior_diagnostic.py']
    native_paths=list(a.native_source.rglob('*.lua'))+list(a.native_source.rglob('*.py'))+list((a.crossbench_root/'crossbench').glob('*.py'))+[a.dependency_freeze]
    result={'status':'PASS' if all(c['equal'] for c in checks) else 'FAIL','native_episodes':n,
        'model_hashes':models[0],'source_hashes':{name:sha(a.sources/name) for name in names},
        'runtime_file_hashes':{str(path.resolve()):sha(path) for path in sorted(native_paths)},
        'checks':checks,'scope':'Two fresh processes, multiple within-process repetitions and changed preceding episode history. Diagnostic sample, not proof of universal determinism.',
        'used_in_effect_estimation':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
