"""Bounded E2C/E4C replication with unchanged source/config contents."""
import datetime,json,os,pathlib,subprocess,sys,time
b=pathlib.Path(__file__).resolve().parent;repo=b.parent/'colin_pivot';out=b/'remaining_v9';out.mkdir(exist_ok=False)
env=dict(os.environ,PIP_INDEX_URL='https://pypi.tuna.tsinghua.edu.cn/simple',PIP_CONFIG_FILE='/dev/null',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(repo/'src')+':'+str(repo),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
py=str(b/'.venv/bin/python');status={'state':'running','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'steps':[],'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()}
def save():(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
def execute(label,command,timeout):
    status['step']=label;save();start=time.monotonic()
    with (out/(label+'.log')).open('w') as log:
        r=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    status['steps'].append({'label':label,'returncode':r.returncode,'seconds':round(time.monotonic()-start,3)});save()
    if r.returncode:raise RuntimeError(label+' failed')
def check(path):
    sys.path.insert(0,str(b));from run_cpu import verify
    result=verify(path);assert result['artifacts_valid'],result
    status.setdefault('verification',{})[path.name]=result;save()
try:
    assert status['source_commit']=='9e3be723dbe895101c13dd3a6782d3e058b91558'
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip()
    execute('external_dependencies',[py,'-m','pip','install','-c',str(b/'requirements-cpu.txt'),'mpe2==1.1.0','pettingzoo==1.27.0','gymnasium==1.3.0'],240)
    (out/'environment.freeze.txt').write_text(subprocess.check_output([py,'-m','pip','freeze'],text=True))
    execute('e2c_smoke',[py,'-m','experiments.v9.run','--experiment','e2c','--profile','smoke','--root',str(repo),'--output',str(out/'e2c_smoke')],180);check(out/'e2c_smoke')
    execute('e2c_confirmatory',[py,'-m','experiments.v9.run','--experiment','e2c','--profile','confirmatory','--root',str(repo),'--output',str(out/'e2c_confirmatory')],1200);check(out/'e2c_confirmatory')
    # An untracked context inside the repository resolves the unchanged E4C
    # relative input path to this run, while git provenance resolves to parent.
    context=repo/'.local_replication_context_20260915';context.mkdir(exist_ok=False);(context/'configs').symlink_to(repo/'configs',target_is_directory=True)
    (context/'results/v9').mkdir(parents=True);(context/'results/v9/e2c-confirmatory').symlink_to(out/'e2c_confirmatory',target_is_directory=True)
    execute('e4c_confirmatory',[py,'-m','experiments.v9.run','--experiment','e4c','--profile','confirmatory','--root',str(context),'--output',str(out/'e4c_confirmatory')],600);check(out/'e4c_confirmatory')
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip()
    status.update(state='complete',input_binding='Unchanged E4C config; untracked context symlink points to newly generated E2C artifact.')
except Exception as exc:status.update(state='failed',error_type=type(exc).__name__,error=str(exc))
finally:status['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print(json.dumps(status),flush=True)
