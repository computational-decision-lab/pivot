"""Finish the current bounded run by comparing results and packing small outputs.
No notifications, external account actions, or recurring jobs.
"""
import json,pathlib,subprocess,sys,tarfile,time
b=pathlib.Path(__file__).resolve().parent
start=time.monotonic();status={}
while time.monotonic()-start<11000:
    p=b/'job_confirmatory/status.json'
    try:status=json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):pass
    if status.get('state') in ['failed','complete']:break
    time.sleep(10)
else:status={'state':'finalizer_deadline'}
summary={'upstream_job_state':status.get('state'),'new_results':str(b/'job_confirmatory/confirmatory'),'comparison_completed':False}
if status.get('state')=='complete':
    result=subprocess.run([str(b/'.venv/bin/python'),str(b/'compare_upstream.py'),'--new',str(b/'job_confirmatory/confirmatory'),'--repo',str(b.parent/'colin_pivot'),'--output',str(b/'upstream_comparison.json')],capture_output=True,text=True,timeout=60)
    (b/'comparison.log').write_text(result.stdout+result.stderr)
    summary['comparison_completed']=result.returncode==0
with tarfile.open(b/'completed_results_small.tar.gz','w:gz') as tar:
    paths=[b/'upstream_comparison.json',b/'deepseek_check.json',b/'melting_quality/summary.json',b/'melting_quality/protocol.json',b/'job_confirmatory/status.json',b/'job_confirmatory/confirmatory/run_summary.json',b/'replay_mpe30/summary.json',b/'replay_mpe30/protocol.json',b/'replay_mpe30/selection_seal.json']
    for name in ['e3c','e5c','e7c']:
        folder=b/'job_confirmatory/confirmatory'/name
        paths.extend(folder.glob('*summary.json'))
        paths.extend(folder/x for x in ['scientific_decision.json','provenance.json','manifest.json'])
    for p in paths:
        if p.is_file():tar.add(p,arcname=str(p.relative_to(b)))
summary['archive_bytes']=(b/'completed_results_small.tar.gz').stat().st_size
(b/'finalization.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
