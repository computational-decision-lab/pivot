"""Detached, bounded upstream CPU job; no cloud lifecycle or purchasing actions."""
import argparse,datetime,json,os,pathlib,subprocess,sys,time,traceback
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['setup','confirmatory'],required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
here=pathlib.Path(__file__).resolve().parent
job=here/('job_'+a.stage)
if not a.worker:
    job.mkdir(exist_ok=False)
    with (job/'launcher.log').open('w') as log:
        child=subprocess.Popen([sys.executable,str(pathlib.Path(__file__).resolve()),'--stage',a.stage,'--worker'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (job/'pid').write_text(str(child.pid)+'\n')
    print(json.dumps({'pid':child.pid,'job':str(job),'stage':a.stage}),flush=True)
    raise SystemExit()
status={'stage':a.stage,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'steps':[],'state':'running'}
def save():(job/'status.json').write_text(json.dumps(status,indent=2)+'\n')
env=dict(os.environ,PATH='/root/miniconda3/bin:/usr/local/bin:/usr/bin:/bin',PIVOT_PYTHON='/root/miniconda3/bin/python',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
py=str(here/'.venv/bin/python');runner=str(here/'run_cpu.py')
steps=[('bootstrap',['bash',str(here/'bootstrap_cpu.sh')],600),('smoke',[py,runner,'--profile','smoke','--output',str(job/'smoke')],900),('dev',[py,runner,'--profile','dev','--output',str(job/'dev'),'--timeout-seconds','1200'],3700)] if a.stage=='setup' else [('confirmatory',[py,runner,'--profile','confirmatory','--output',str(job/'confirmatory'),'--timeout-seconds','3600'],10900)]
save()
try:
    for name,cmd,timeout in steps:
        status['current_step']=name;save();start=time.monotonic()
        with (job/(name+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=here,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        status['steps'].append({'name':name,'returncode':r.returncode,'seconds':round(time.monotonic()-start,3)});save()
        if r.returncode:raise RuntimeError('Step failed: '+name)
    status['state']='complete'
except Exception as exc:
    status.update(state='failed',error_type=type(exc).__name__,error=str(exc));traceback.print_exc()
finally:
    status['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print(json.dumps(status),flush=True)
