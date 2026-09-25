"""Native MetaDrive paired response benchmark. Protocol controls scientific choices."""
import argparse, concurrent.futures, hashlib, json, multiprocessing as mp, os, pathlib, time, traceback
import numpy as np


def atomic_json(path, data):
    path=pathlib.Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n'); tmp.replace(path)


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


_EVALUATOR=None
_ECONFIG=None

def episode_worker(task):
    global _EVALUATOR,_ECONFIG
    from native_adapter import NativeEvaluator
    if _EVALUATOR is None or _ECONFIG!=task['environment']:
        if _EVALUATOR is not None: _EVALUATOR.close()
        _EVALUATOR=NativeEvaluator(task['environment']); _ECONFIG=task['environment']
    r=_EVALUATOR.run_episode(task['seed'],task['ego'],task['background'])
    r['task']=task
    return r


class Bank:
    def __init__(self,out,protocol,workers,one_episode_per_process=False):
        self.out=pathlib.Path(out); self.out.mkdir(parents=True,exist_ok=True)
        self.p=protocol; self.completed=0; self.started=time.time()
        self.native_sha=hashlib.sha256(pathlib.Path(__file__).with_name('native_adapter.py').read_bytes()).hexdigest()
        kwargs={'max_workers':workers,'mp_context':mp.get_context('spawn')}
        if one_episode_per_process: kwargs['max_tasks_per_child']=1
        self.pool=concurrent.futures.ProcessPoolExecutor(**kwargs)
        self.pending={}; self.known={}; self.stage='init'
        self.load_count=0
    def register(self,root,seed,ego,bg):
        stride=int(self.p['seed_layout']['root_stride'])
        env=dict(self.p['environment'],start_seed=root*stride,num_scenarios=stride)
        t={'seed':seed,'ego':ego,'background':bg,'environment':env,'native_adapter_sha256':self.native_sha,'upstream_commit':self.p['upstream_commit']}
        key=digest(t); path=self.out/'episodes'/f'{key}.json'
        if key not in self.known:
            if path.exists():
                data=json.loads(path.read_text())
                if data['task']!=t: raise RuntimeError('Cache task mismatch '+str(path))
                self.known[key]=data; self.load_count+=1
            else:
                if len(self.known) >= int(self.p['runtime_limits']['per_root_registered_episode_upper_bound'])*len(self.p['root_seeds']):
                    raise RuntimeError('Prespecified phase episode cap exceeded; no automatic resume')
                self.known[key]=None; self.pending[key]=(t,path)
        return key
    def execute(self,stage):
        self.stage=stage
        fs={self.pool.submit(episode_worker,t):(key,path) for key,(t,path) in self.pending.items()}
        expected=len(fs); self.pending={}
        self.status('running',stage_pending=expected)
        for f in concurrent.futures.as_completed(fs):
            key,path=fs[f]
            try: data=f.result()
            except BaseException:
                self.status('FAILED',traceback=traceback.format_exc())
                for x in fs: x.cancel()
                raise
            atomic_json(path,data); self.known[key]=data; self.completed+=1
            if self.completed%20==0:
                self.status('running',stage_pending=sum(not z.done() for z in fs))
                print(json.dumps({'stage':stage,'new_episodes_completed':self.completed,'elapsed_seconds':round(time.time()-self.started,1)}),flush=True)
        self.status('running',stage_pending=0)
    def status(self,status,**extra):
        atomic_json(self.out/'status.json',{'status':status,'stage':self.stage,'new_episodes_completed':self.completed,'episodes_reused':self.load_count,'wallclock_started':self.started,'updated':time.time(),'elapsed_seconds':time.time()-self.started,**extra})
    def get(self,key): return self.known[key]
    def close(self): self.pool.shutdown(wait=True,cancel_futures=True)


def scene(p,root,stream,candidate=0,j=0):
    d=p['seed_layout']; return root*d['root_stride']+d[stream]+candidate*d['candidate_stride']+j


def response_pool(p,root):
    rng=np.random.default_rng(root+p['response']['pool_seed_offset'])
    pool=[p['background_initial']]
    for _ in range(p['response']['pool_size']-1):
        pool.append({k:float(rng.uniform(*v)) for k,v in p['response']['bounds'].items()})
    return pool


def run(args):
    p=json.loads(pathlib.Path(args.protocol).read_text()); out=pathlib.Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    old=out/'protocol.json'
    if old.exists() and json.loads(old.read_text())!=p: raise RuntimeError('Output protocol differs; refusing overwrite')
    atomic_json(old,p)
    source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in pathlib.Path(__file__).parent.glob('*.py')}
    previous=out/'run_provenance.json'
    if previous.exists() and json.loads(previous.read_text())['source_sha256']!=source_hashes: raise RuntimeError('Source hashes changed; refusing mixed-code resume')
    atomic_json(out/'run_provenance.json',{'source_sha256':source_hashes,'protocol_sha256':hashlib.sha256(pathlib.Path(args.protocol).read_bytes()).hexdigest(),'workers':args.workers,'one_episode_per_process':args.one_episode_per_process,'started':time.time()})
    roots=p['root_seeds']; policies=[p['incumbent']]+p['candidates']; K=len(p['candidates']); H=p['response']['adaptation_steps']
    bank=Bank(out,p,args.workers,args.one_episode_per_process)
    refs={}
    try:
        # Discovery/calibration/confirmation have disjoint roots. All policy
        # responses in a root use precisely the same pool and training scenes.
        for root in roots:
            rd=out/'roots'/f'seed_{root}'; rd.mkdir(parents=True,exist_ok=True)
            pool=response_pool(p,root); rr={'pool':pool,'training':[]}
            for pol in policies:
                rr['training'].append([[bank.register(root,scene(p,root,'train',j=j),pol,bg) for j in range(p['response']['training_episodes_per_profile'])] for bg in pool])
            refs[root]=rr
        bank.execute('response_training')
        for root,rr in refs.items():
            train=np.array([[[bank.get(key)['background_mean_return'] for key in episodes] for episodes in profiles] for profiles in rr['training']])
            response_indices=[[int(np.argmax(train[i,:h].mean(axis=1))) for h in H] for i in range(K+1)]
            rr['response_indices']=response_indices
            rr['responses']=[[p['background_initial']]+[rr['pool'][idx] for idx in ids] for ids in response_indices]
            atomic_json(out/'roots'/f'seed_{root}'/'response_training.json',{'seed':root,'pool':rr['pool'],'training_returns':train.tolist(),'response_indices':response_indices,'training_episode_references':rr['training']})
            # All proxy comparisons use same four scenes and frozen background.
            rr['proxy']=[[bank.register(root,scene(p,root,'proxy',j=j),pol,p['background_initial']) for j in range(p['proxy_episodes'])] for pol in policies]
            # Held-out response validation shares scenes over all 3 response
            # conditions; its rewards never choose profiles or selectors.
            rr['validation']=[[[bank.register(root,scene(p,root,'response_validation',j=j),pol,bg) for j in range(p['response_validation_episodes'])] for bg in rr['responses'][i]] for i,pol in enumerate(policies)]
            for stream,n in [('selection',p['selection_episodes']),('audit_a',p['audit_episodes_per_split']),('audit_b',p['audit_episodes_per_split'])]:
                conditions=range(1,3) if stream=='selection' else range(3)
                rr[stream]=[]
                for h in conditions:
                    rr[stream].append([[[bank.register(root,scene(p,root,stream,candidate=i,j=j),policies[i+1],rr['responses'][i+1][h]),bank.register(root,scene(p,root,stream,candidate=i,j=j),policies[0],rr['responses'][0][h])] for j in range(n)] for i in range(K)])
        bank.execute('proxy_selection_independent_audits')
        for root,rr in refs.items():
            def gain(pair):
                a,b=[bank.get(key) for key in pair]
                if a['initial_positions']!=b['initial_positions']: raise RuntimeError('Paired scene initial conditions mismatch')
                return a['ego_return']-b['ego_return']
            proxy=np.array([[bank.get(key)['ego_return'] for key in keys] for keys in rr['proxy']])
            selection=np.array([[[gain(pair) for pair in cand] for cand in cond] for cond in rr['selection']])
            aa=np.array([[[gain(pair) for pair in cand] for cand in cond] for cond in rr['audit_a']])
            ab=np.array([[[gain(pair) for pair in cand] for cand in cond] for cond in rr['audit_b']])
            val=np.array([[[bank.get(key)['background_mean_return'] for key in keys] for keys in cond] for cond in rr['validation']])
            outcomes={}
            for stream in ('audit_a','audit_b'):
                outcomes[stream]={m:[[[int(bank.get(pair[0])[m]) for pair in cand] for cand in cond] for cond in rr[stream]] for m in ('ego_crash','ego_arrival','ego_out_of_road')}
            data={'seed':root,'adaptation_steps':H,'candidate_names':p['candidate_names'],'proxy':(proxy[1:]-proxy[0]).mean(axis=1).tolist(),'proxy_paired_episodes':(proxy[1:]-proxy[0]).tolist(),'selection':selection.tolist(),'audit_a':aa[1:].tolist(),'audit_b':ab[1:].tolist(),'audit_fixed_a':aa[0].tolist(),'audit_fixed_b':ab[0].tolist(),'response_validation':val.tolist(),'response_indices':rr['response_indices'],'native_outcomes':outcomes}
            atomic_json(out/'roots'/f'seed_{root}'/'root.json',data)
            atomic_json(out/'roots'/f'seed_{root}'/'episode_references.json',rr)
            print(json.dumps({'ROOT_COMPLETE':root,'path':str(out/'roots'/f'seed_{root}'/'root.json')}),flush=True)
        bank.status('COMPLETE',roots_completed=len(roots),unique_native_episodes=len(bank.known))
        print('BANK_COMPLETE',flush=True)
    finally:
        bank.close()


if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--protocol',required=True);a.add_argument('--out',required=True);a.add_argument('--workers',type=int,default=16);a.add_argument('--one-episode-per-process',action='store_true')
    run(a.parse_args())
