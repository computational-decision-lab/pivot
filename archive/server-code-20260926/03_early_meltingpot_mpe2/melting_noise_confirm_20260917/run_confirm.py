"""Launch the frozen fresh-seed method cohort; stop the cohort on worker failure."""
import argparse,concurrent.futures,datetime,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
from verify_melting_mechanism import sha

def write(p,s):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(s,indent=2)+'\n');q.replace(p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-path',type=Path,required=True);p.add_argument('--crossbench-root',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--workers',type=int,default=14);p.add_argument('--max-seconds',type=int,default=14400);a=p.parse_args()
    protocol=json.loads(a.protocol.read_text());assert protocol['status']=='FROZEN' and protocol['test_seeds']==list(range(47000,47030))
    assert 1<=a.workers<=14 and 0<a.max_seconds<=14400
    assert protocol['seed_extension'] is False
    hashes={**protocol['runtime_file_hashes'],**protocol['source_file_hashes'],**protocol['calibration_file_hashes']}
    for path,h in hashes.items():assert sha(path)==h,f'Frozen input changed: {path}'
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'protocol.json',protocol)
    root=Path(__file__).resolve().parent;env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',TF_ENABLE_ONEDNN_OPTS='0',TF_CPP_MIN_LOG_LEVEL='3',PYTHONHASHSEED='0')
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS']:env[key]='1'
    stopped=threading.Event();lock=threading.Lock();active={};results=[];start=time.monotonic()
    def run(seed):
        if stopped.is_set():return {'seed':seed,'status':'not_started_after_failure'}
        if time.monotonic()-start>=a.max_seconds:
            stopped.set();return {'seed':seed,'status':'not_started_walltime_exhausted'}
        remaining=max(1,int(a.max_seconds-(time.monotonic()-start)))
        cmd=[sys.executable,str(root/'native_confirm.py'),'--protocol',str(a.protocol.resolve()),'--seed',str(seed),'--output',str((a.output/f'seed_{seed}').resolve()),
             '--calibration',str(a.calibration.resolve()),'--model-path',str(a.model_path.resolve()),'--crossbench-root',str(a.crossbench_root.resolve()),'--max-seconds',str(remaining)]
        with (a.output/f'seed_{seed}.log').open('w') as f:
            process=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            with lock:active[seed]=process
            try:code=process.wait(timeout=remaining+30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:code=process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
            finally:
                with lock:active.pop(seed,None)
        r={'seed':seed,'status':'complete' if code==0 else 'failed','returncode':code}
        write(a.output/f'seed_{seed}.job.json',r);return r
    write(a.output/'status.json',{'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':[]})
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run,s) for s in protocol['test_seeds']]):
            r=future.result();results.append(r)
            if r['status']=='failed':
                stopped.set()
                with lock:
                    for proc in active.values():
                        if proc.poll() is None:
                            try:os.killpg(proc.pid,signal.SIGTERM)
                            except ProcessLookupError:pass
            write(a.output/'status.json',{'status':'running' if not stopped.is_set() else 'stopping_after_failure','jobs':results});print(json.dumps(r),flush=True)
    unchanged=all(sha(path)==h for path,h in hashes.items());ok=unchanged and all(r['status']=='complete' for r in results)
    write(a.output/'status.json',{'status':'complete' if ok else 'incomplete','jobs':results,'frozen_inputs_unchanged':unchanged,'seconds':time.monotonic()-start})
    raise SystemExit(0 if ok else 2)
if __name__=='__main__':main()
