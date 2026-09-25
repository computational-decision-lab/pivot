"""Dependent postprocessing for the currently running, bounded audit batch."""
import json,os,pathlib,subprocess,tarfile,time
b=pathlib.Path(__file__).resolve().parent;old=pathlib.Path('/root/autodl-tmp/colin_melting');study=old/'runs/melting_cross_main';status={'state':'waiting_for_primary_audit'}
def save():(b/'melting_postprocess.json').write_text(json.dumps(status,indent=2)+'\n')
save();start=time.monotonic()
def ready():return all((study/task/'test'/f'seed_{seed}'/'audit'/f'candidate_{j}'/f'replicate_{r}'/'paired.json').exists() for task in ['melting_pd','melting_stag'] for seed in range(1300,1306) for j in range(4) for r in range(4))
while not ready() and time.monotonic()-start<660:time.sleep(10)
if not ready():status.update(state='incomplete_at_deadline');save();raise SystemExit(1)
env=dict(os.environ,PYTHONPATH=str(old)+':'+str(old/'vendor/meltingpot_source'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
commands=[('primary_report',['/root/autodl-tmp/benchmark_extensions/melting_env/bin/python',str(b/'complete_melting_primary_v2.py'),'--report-only','--output',str(b/'melting_primary_report')]),('author_score',[str(b/'.venv/bin/python'),str(b/'replay_author_melting.py'),'--phase','score','--base',str(study),'--repo',str(b.parent/'colin_pivot'),'--output',str(b/'author_melting')])]
for label,cmd in commands:
    status.update(state='running',step=label);save()
    with (b/(label+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=old,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    if r.returncode:status.update(state='failed',returncode=r.returncode);save();raise SystemExit(1)
with tarfile.open(b/'melting_completed_small.tar.gz','w:gz') as tar:
    for folder in ['melting_primary_report','author_melting']:
        for name in ['summary.json','protocol.json','scored_decisions.json','selection_seal.json','status.json']:
            p=b/folder/name
            if p.exists():tar.add(p,arcname=str(p.relative_to(b)))
status.update(state='complete',all_short_response_panels=12,author_decisions=168,long_response_completed=False,archive_bytes=(b/'melting_completed_small.tar.gz').stat().st_size);save();print(json.dumps(status),flush=True)
