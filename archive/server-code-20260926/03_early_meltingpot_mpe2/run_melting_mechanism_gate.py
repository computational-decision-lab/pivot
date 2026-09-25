"""Run only the frozen mechanism cohort, after an explicit native replay PASS."""
from __future__ import annotations
import argparse,concurrent.futures,datetime,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--replay-gate',type=Path,required=True)
    p.add_argument('--model-path',type=Path,required=True);p.add_argument('--crossbench-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=12)
    p.add_argument('--max-seconds',type=int,default=7200);p.add_argument('--plan-only',action='store_true')
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());m=protocol['mechanism']
    assert m['adaptation_episodes']=={'fixed':0,'short':4,'long':32}
    assert m['checkpoints']==[0,4,32] and m['seeds']==list(range(43000,43012))
    assert m['eval_replicates_per_skill_stratum_per_block']==16 and not m['seed_extension']
    assert 1<=a.workers<=12 and 0<a.max_seconds<=14400
    for name,expected in protocol['source_hashes_before_replay_fix'].items():
        assert sha(ROOT/name)==expected,f'Source changed: {name}'
    replay=json.loads(a.replay_gate.read_text())
    assert replay['status']=='PASS' and replay['native_episodes']>=8,'Native replay gate has not passed'
    assert replay['runtime_file_hashes']==protocol['runtime_file_hashes'],'Native runtime not frozen in protocol'
    for path,expected in replay['runtime_file_hashes'].items():
        assert sha(path)==expected,f'Native runtime changed since replay: {path}'
    for name,expected in replay['source_hashes'].items():
        assert sha(ROOT/name)==expected,f'Replay uses different source: {name}'
    for name,expected in replay['model_hashes'].items():
        assert sha(a.model_path/name)==expected,'Low-level model changed since replay'
    commands=[]
    for seed in m['seeds']:
        cmd=[sys.executable,str(ROOT/'melting_response_sprint.py'),'--seed',str(seed),'--output',str(a.output/f'seed_{seed}'),
             '--model-path',str(a.model_path),'--crossbench-root',str(a.crossbench_root),'--train-episodes','32',
             '--checkpoints','0','4','32','--eval-episodes-per-stratum','16','--max-seconds',str(a.max_seconds)]
        commands.append((seed,cmd))
    if a.plan_only:
        print(json.dumps({'native_episodes':2688,'seeds':m['seeds'],'commands':commands}));return
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'protocol.json',protocol);write(a.output/'replay_gate.json',replay)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',TF_CPP_MIN_LOG_LEVEL='3',TF_ENABLE_ONEDNN_OPTS='0',PYTHONHASHSEED='0')
    for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS']:env[k]='1'
    start=time.monotonic();jobs=[]
    write(a.output/'status.json',{'status':'running','jobs':jobs,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    def run(item):
        seed,cmd=item;remaining=max(1,int(a.max_seconds-(time.monotonic()-start)));cmd[-1]=str(remaining)
        with (a.output/f'seed_{seed}.log').open('w') as f:
            process=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            timed_out=False
            try:code=process.wait(timeout=remaining+20)
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(process.pid,signal.SIGTERM)
                try:code=process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
        r={'seed':seed,'returncode':code,'timed_out':timed_out}
        write(a.output/f'seed_{seed}.job.json',r);return r
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for f in concurrent.futures.as_completed([ex.submit(run,item) for item in commands]):
            r=f.result();jobs.append(r);write(a.output/'status.json',{'status':'running','jobs':jobs});print(json.dumps(r),flush=True)
    ok=all(x['returncode']==0 and not x['timed_out'] for x in jobs)
    runtime_unchanged=all(sha(path)==expected for path,expected in replay['runtime_file_hashes'].items())
    ok=ok and runtime_unchanged
    write(a.output/'status.json',{'status':'complete' if ok else 'incomplete','jobs':jobs,'seconds':time.monotonic()-start,'runtime_unchanged':runtime_unchanged})
    raise SystemExit(0 if ok else 2)
if __name__=='__main__':main()
