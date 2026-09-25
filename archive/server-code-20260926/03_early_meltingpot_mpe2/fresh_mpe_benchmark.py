"""New PPO candidate panels with prospectively frozen selection and separate audits."""
from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,datetime,hashlib,importlib.metadata,json,math,multiprocessing,subprocess,sys,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from pivot.acquisition.pivot_voi import BayesianLinearDeltaPosterior,select_pivot_voi
from pivot.algorithms.pivot import run_pivot_voi_round,run_pivot_round
from sequential_validation import run as sequential_run,exchangeable_discrepancy

BASE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');temp.replace(p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def panel(out,task,seed):return Path(out)/task/f'seed_{seed}'

def prepare(spec):
    from crossbench.experiment import train,load,evaluate,collect,delta,Seeds
    from pivot_v2.features import footprint
    task,seed,cfg,out=spec;b=panel(out,task,seed);b.mkdir(parents=True,exist_ok=True)
    seeds=Seeds(b/'seeds.json',seed);started=time.monotonic()
    write(b/'manifest.json',{'task':task,'seed':seed,'config':cfg,'started_at':now()})
    write(b/'status.json',{'phase':'pretraining'})
    p0,_=train(task,'response',None,b/'initial/response0',cfg['pretrain_steps']//2,seeds.get('initial','response0'),cfg)
    f0,_=train(task,'focal',load(p0),b/'initial/focal0',cfg['pretrain_steps']//2,seeds.get('initial','focal0'),cfg)
    op,_=train(task,'response',load(f0),b/'initial/response',cfg['pretrain_steps']//2,seeds.get('initial','response'),cfg,p0)
    oldp,_=train(task,'focal',load(op),b/'initial/focal',cfg['pretrain_steps']//2,seeds.get('initial','focal'),cfg,f0)
    old,opp=load(oldp),load(op);candidates=[]
    write(b/'status.json',{'phase':'candidate_training'})
    for j in range(cfg['candidates']):
        path,_=train(task,'focal',opp,b/f'candidate_{j}',cfg['candidate_steps'],seeds.get('candidate',j),cfg,oldp,cfg['learning_rate']*[.5,1,2,1][j])
        candidates.append(path)
    es=[seeds.get('proxy',e) for e in range(cfg['proxy_episodes'])]
    old_eval=evaluate(task,old,opp,es,cfg['horizon']);proxy=[];features=[]
    state_seed=seeds.get('footprint');old_states=collect(task,old,opp,state_seed,cfg)
    for j,path in enumerate(candidates):
        new=load(path);new_eval=evaluate(task,new,opp,es,cfg['horizon']);d=delta(old_eval,new_eval)
        proxy.append({'delta':d,'old':old_eval,'new':new_eval})
        features.append(footprint(old,new,old_states,collect(task,new,opp,state_seed,cfg),d))
    write(b/'proxy.json',proxy);write(b/'features.json',features)
    es=[seeds.get('quality',e) for e in range(cfg['query_episodes'])]
    a=evaluate(task,load(b/'initial/focal0_untrained.zip'),opp,es,cfg['horizon']);c=evaluate(task,old,opp,es,cfg['horizon'])
    write(b/'quality.json',{'gain':delta(a,c),'trained':c,'untrained':a})
    write(b/'model_hashes.json',{str(p.relative_to(b)):sha(p) for p in [oldp,op]+candidates})
    training=sum(read(p)['actual_steps'] for p in list((b/'initial').glob('*.json'))+list(b.glob('candidate_*.json')))
    evaluation=(5*cfg['proxy_episodes']+5*cfg['footprint_episodes']+2*cfg['query_episodes'])*cfg['horizon']
    result={'task':task,'seed':seed,'phase':'prepared','seconds':time.monotonic()-started,'env_steps':training+evaluation,'training_steps':training,'evaluation_steps':evaluation}
    write(b/'status.json',result);return result

def response(spec):
    from crossbench.experiment import train,load,evaluate,delta
    task,seed,cfg,out,stream,j,r,training_seed,es=spec;b=panel(out,task,seed)
    folder=b/stream/f'candidate_{j}'/f'replicate_{r}';target=folder/'paired.json'
    if target.exists():raise RuntimeError('Fresh response already exists')
    old,new,opp=load(b/'initial/focal.zip'),load(b/f'candidate_{j}.zip'),load(b/'initial/response.zip')
    start=time.monotonic()
    a,am=train(task,'response',old,folder/'old_response',cfg['adapt_steps'],training_seed,cfg,b/'initial/response.zip')
    c,cm=train(task,'response',new,folder/'new_response',cfg['adapt_steps'],training_seed,cfg,b/'initial/response.zip')
    ar=evaluate(task,old,load(a),es,cfg['horizon']);cr=evaluate(task,new,load(c),es,cfg['horizon'])
    assert am['signature']['start_hash']==cm['signature']['start_hash']==sha(b/'initial/response.zip')
    assert am['signature']['seed']==cm['signature']['seed']==training_seed
    row={'stream':stream,'task':task,'seed':seed,'candidate':j,'replicate':r,'training_seed':training_seed,'evaluation_seeds':es,
         'delta':delta(ar,cr),'old':ar,'new':cr,'total_env_steps':am['actual_steps']+cm['actual_steps']+2*len(es)*cfg['horizon']}
    if stream=='audit':
        fa=evaluate(task,old,opp,es,cfg['horizon']);fc=evaluate(task,new,opp,es,cfg['horizon'])
        crossed=evaluate(task,new,load(a),es,cfg['horizon'])
        row.update(frozen_delta=delta(fa,fc),frozen_old=fa,frozen_new=fc,new_against_old_response=crossed,
            candidate_specific_effect=delta(crossed,cr),
            response_own_reward_gain=float(np.mean([x['response_return']-y['response_return'] for x,y in zip(cr,fc)])),
            audit_extra_eval_steps=3*len(es)*cfg['horizon'])
    write(target,row)
    return {'task':task,'seed':seed,'phase':stream,'candidate':j,'replicate':r,'seconds':time.monotonic()-start,'env_steps':row['total_env_steps']+row.get('audit_extra_eval_steps',0)}

def fit(rows):
    raw=np.array([r['features'][1:] for r in rows]);y=np.array([r['target_correction'] for r in rows])
    mu=raw.mean(0);sd=raw.std(0);sd[sd<1e-8]=1
    noise=max(1e-6,float(np.mean([r['noise'] for r in rows])))
    m=BayesianLinearDeltaPosterior(prior_precision=1.,noise_variance=noise/4).fit(np.c_[np.ones(len(raw)),(raw-mu)/sd],y)
    return m,mu,sd,noise

def calibrate(protocol,out):
    for task in protocol['tasks']:
        paths=[Path(protocol['calibration_source'])/task/'calibration'/f'seed_{s}'/'summary.json' for s in protocol['calibration_seeds']]
        rows=[r for p in paths for r in read(p)['rows']];errors=[];known=[]
        assert {r['seed'] for r in rows}==set(protocol['calibration_seeds'])
        for seed in protocol['calibration_seeds']:
            train=[r for r in rows if r['seed']!=seed];held=[r for r in rows if r['seed']==seed]
            m,mu,sd,noise=fit(train);z=np.c_[np.ones(4),(np.array([r['features'][1:] for r in held])-mu)/sd]
            errors.append(np.array([r['target_correction'] for r in held])-m.predict(z))
            known.append(z@m.covariance@z.T+np.diag([r['noise']/4 for r in held]))
        discrepancy,info=exchangeable_discrepancy(errors,known);m,mu,sd,noise=fit(rows)
        write(Path(out)/task/'calibration.json',{'feature_mean':mu.tolist(),'feature_scale':sd.tolist(),'mean':m.mean.tolist(),'covariance':m.covariance.tolist(),
            'single_query_noise':noise,'calibration_mean_noise':noise/4,'discrepancy':discrepancy.tolist(),'discrepancy_diagnostic':info,
            'source_hashes':{str(p):sha(p) for p in paths},'n_calibration_seeds':len(paths),'test_labels_used':False})

def select_panel(spec):
    task,seed,cfg,out,protocol=spec;b=panel(out,task,seed);cal=read(Path(out)/task/'calibration.json')
    raw=np.array(read(b/'features.json'));proxy=raw[:,0];z=np.c_[np.ones(4),(raw[:,1:]-cal['feature_mean'])/cal['feature_scale']]
    model=BayesianLinearDeltaPosterior(prior_precision=1.,noise_variance=cal['single_query_noise'],mean=np.array(cal['mean']),covariance=np.array(cal['covariance']))
    mu=model.predict(z)+proxy;cov=z@model.covariance@z.T;cal_cov=cov+np.array(cal['discrepancy'])
    cost=2*math.ceil(cfg['adapt_steps']/(cfg['n_envs']*cfg['n_steps']))*(cfg['n_envs']*cfg['n_steps'])+2*cfg['query_episodes']*cfg['horizon']
    candidates=[{'transition_id':str(j),'delta_proxy':float(proxy[j]),'features':z[j].tolist(),'hf_query_cost':cost} for j in range(4)]
    accessed=[]
    def query(j,r):
        assert 0<=r<cfg['selection_replicates_available']
        p=b/'selection'/f'candidate_{j}'/f'replicate_{r}'/'paired.json';d=read(p);assert d['stream']=='selection'
        accessed.append({'candidate':j,'replicate':r,'delta':d['delta'],'cost':d['total_env_steps'],'sha256':sha(p)});return d
    def hf(c):
        d=query(int(c['transition_id']),0);return {'delta_true':d['delta'],'hf_query_cost':d['total_env_steps']}
    records=[]
    for method in protocol['methods']:
        for budget in ([0] if method in ['proxy_only','correction_only'] else protocol['query_budgets']):
            accessed.clear()
            if method in ['proxy_only','correction_only']:
                values=proxy if method=='proxy_only' else mu
                d={'selected':int(np.argmax(values)),'estimates':values.tolist(),'stop_reason':method,'query_cost':0}
            elif method in ['author_batch','author_batch_no_stop','random_batch']:
                if method=='author_batch':
                    result=run_pivot_voi_round(None,candidates,hf,model,budget,seed=seed,delta=.05,eta=0.,fantasies=64,posterior_samples=256)
                elif method=='author_batch_no_stop':
                    result=run_pivot_round(None,candidates,None,hf,select_pivot_voi,budget,model=model,acquisition_kwargs={'seed':seed,'fantasies':64,'posterior_samples':256})
                else:
                    order=np.random.default_rng(seed+11).permutation(4)
                    result=run_pivot_round(None,candidates,None,hf,lambda rows,model,budget:[str(j) for j in order[:budget]],budget,model=model)
                d={'selected':int(result.selected_candidate_id),'selected_estimate':result.selected_delta_estimate,'stop_reason':result.stop_reason,'query_cost':result.hf_cost}
            elif method=='uniform_sample_mean':
                order=np.random.default_rng(seed+11).permutation(4);values=mu.copy()
                for j in order[:budget]:values[j]=query(int(j),0)['delta']
                d={'selected':int(np.argmax(values)),'estimates':values.tolist(),'query_cost':budget*cost,'stop_reason':'budget'}
            else:
                c=cov if method=='sequential_linear' else cal_cov
                d=sequential_run(mu,c,cal['single_query_noise'],query,[cost]*4,budget,seed+23,
                    strategy='random' if method.startswith('random_sequential') else 'voi',stop=not method.endswith('_no_stop'))
            assert d['query_cost']==sum(q['cost'] for q in accessed)
            d.update(task=task,seed=seed,method=method,budget=budget,queries=list(accessed),query_count=len(accessed))
            records.append(d)
    write(b/'decisions_frozen.json',records)
    write(b/'selection_seal.json',{'sha256':sha(b/'decisions_frozen.json'),'audit_opened':False,'frozen_at':now()})
    return {'task':task,'seed':seed,'phase':'decisions_frozen','n_decisions':len(records)}

def ci(x):
    x=np.array(x,float);rng=np.random.default_rng(20260915);d=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    return {'mean':float(x.mean()),'ci95':np.quantile(d,[.025,.975]).tolist(),'n_training_seeds':len(x)}

def report(protocol,out):
    tables={};all_records=[];all_mechanisms=[];audit_hashes={}
    for task in protocol['tasks']:
        records=[];mechanisms=[];candidates_rows=[]
        for seed in protocol['test_seeds']:
            b=panel(out,task,seed);assert sha(b/'decisions_frozen.json')==read(b/'selection_seal.json')['sha256']
            for path,digest in read(b/'model_hashes.json').items():assert sha(b/path)==digest
            selection_es=set();selection_ts=set()
            for p in (b/'selection').glob('candidate_*/replicate_*/paired.json'):
                q=read(p);selection_es.update(q['evaluation_seeds']);selection_ts.add(q['training_seed'])
            audit=[]
            for j in range(4):
                group=[]
                for r in range(protocol['config']['audit_replicates']):
                    p=b/'audit'/f'candidate_{j}'/f'replicate_{r}'/'paired.json';q=read(p)
                    assert not selection_es.intersection(q['evaluation_seeds']) and q['training_seed'] not in selection_ts
                    assert [x['seed'] for x in q['old']]==[x['seed'] for x in q['new']]==q['evaluation_seeds']
                    audit_hashes[str(p.relative_to(out))]=sha(p);group.append(q)
                audit.append(group)
            values=np.array([[r['delta'] for r in rows] for rows in audit]);frozen=np.array([[r['frozen_delta'] for r in rows] for rows in audit])
            means=values.mean(1);left=values[:,:4].mean(1);right=values[:,4:].mean(1)
            cfg=protocol['config'];common={'candidate_training_steps':4*cfg['candidate_steps'],'proxy_evaluation_steps':5*cfg['proxy_episodes']*cfg['horizon'],'footprint_steps':5*cfg['footprint_episodes']*cfg['horizon']}
            for row in read(b/'decisions_frozen.json'):
                row.pop('trace',None);j=row['selected'];row.update(audit_gain=float(means[j]),frozen_gain=float(frozen[j].mean()),
                    crossfit_reference_gap=float(.5*(right[left.argmax()]-right[j]+left[right.argmax()]-left[j])),
                    common_cost=common,decision_cost=sum(v for k,v in common.items() if k!='footprint_steps' or row['method']!='proxy_only')+row['query_cost'])
                records.append(row)
            flat=[q for group in audit for q in group]
            mechanisms.append({'task':task,'seed':seed,'focal_quality_gain':read(b/'quality.json')['gain'],
                'response_effect':float(np.mean([q['delta']-q['frozen_delta'] for q in flat])),
                'candidate_specific_effect':float(np.mean([q['candidate_specific_effect'] for q in flat])),
                'response_own_reward_gain':float(np.mean([q['response_own_reward_gain'] for q in flat])),
                'audit_winner_half_agreement':bool(left.argmax()==right.argmax())})
            candidates_rows.append({'seed':seed,'proxy':np.array(read(b/'features.json'))[:,0].tolist(),'audit_replicates':values.tolist(),'frozen_replicates':frozen.tolist()})
        method_rows=[]
        for method,budget in sorted({(r['method'],r['budget']) for r in records}):
            rows=[r for r in records if r['method']==method and r['budget']==budget]
            method_rows.append({'method':method,'budget':budget,'audit_gain':ci([r['audit_gain'] for r in rows]),'mean_queries':float(np.mean([r['query_count'] for r in rows])),'mean_query_cost':float(np.mean([r['query_cost'] for r in rows])),'mean_decision_cost':float(np.mean([r['decision_cost'] for r in rows]))})
        contrasts=[]
        for m1,m2 in [('author_batch_no_stop','random_batch'),('sequential_calibrated_no_stop','random_sequential_calibrated_no_stop'),('author_batch','random_batch'),('sequential_linear','author_batch'),('sequential_calibrated','sequential_linear'),('sequential_calibrated','proxy_only'),('sequential_calibrated','sequential_calibrated_no_stop')]:
            budget=protocol['primary_budget'] if m1=='author_batch_no_stop' else 4
            a={r['seed']:r for r in records if r['method']==m1 and r['budget']==budget};c={r['seed']:r for r in records if r['method']==m2 and r['budget']==(0 if m2=='proxy_only' else budget)}
            diffs=[a[s]['audit_gain']-c[s]['audit_gain'] for s in protocol['test_seeds']]
            if m1=='sequential_calibrated_no_stop':
                assert all(a[s]['query_count']==c[s]['query_count']==4 and a[s]['query_cost']==c[s]['query_cost'] for s in a)
            if m1=='author_batch_no_stop':
                assert all(a[s]['query_count']==c[s]['query_count']==budget and a[s]['query_cost']==c[s]['query_cost'] for s in a)
            contrasts.append({'contrast':m1+' minus '+m2,'budget':budget,'gain_difference':ci(diffs),'selection_changed_seeds':sum(a[s]['selected']!=c[s]['selected'] for s in a),'mean_query_count_difference':float(np.mean([a[s]['query_count']-c[s]['query_count'] for s in a]))})
        from scipy.stats import ttest_1samp
        primary=contrasts[0];a={r['seed']:r for r in records if r['method']=='author_batch_no_stop' and r['budget']==protocol['primary_budget']};c={r['seed']:r for r in records if r['method']=='random_batch' and r['budget']==protocol['primary_budget']}
        diffs=np.array([a[s]['audit_gain']-c[s]['audit_gain'] for s in a]);p=float(ttest_1samp(diffs,0).pvalue) if diffs.std()>0 else 1.
        tables[task]={'methods':method_rows,'contrasts':contrasts,'primary_approximate_t_p':p,'mechanism':{k:ci([r[k] for r in mechanisms]) for k in ['focal_quality_gain','response_effect','candidate_specific_effect','response_own_reward_gain','audit_winner_half_agreement']}}
        write(Path(out)/task/'candidate_audit.json',candidates_rows);all_records.extend(records);all_mechanisms.extend(mechanisms)
    previous=0.
    for i,task in enumerate(sorted(tables,key=lambda t:tables[t]['primary_approximate_t_p'])):
        previous=max(previous,min(1.,(len(tables)-i)*tables[task]['primary_approximate_t_p']));tables[task]['primary_holm_p']=previous
    write(Path(out)/'scored_decisions.json',all_records);write(Path(out)/'mechanism_by_seed.json',all_mechanisms);write(Path(out)/'audit_hashes.json',audit_hashes)
    jobs=[json.loads(line) for line in (Path(out)/'completed_jobs.jsonl').read_text().splitlines()]
    physical={phase:sum(r.get('env_steps',0) for r in jobs if r['phase']==phase) for phase in ['prepared','selection','audit']}
    summary={'status':'complete','study_id':protocol['study_id'],'n_fresh_training_seeds_per_task':len(protocol['test_seeds']),'n_tasks':len(tables),'n_platforms':1,'n_candidates':len(protocol['test_seeds'])*len(tables)*4,'n_decisions':len(all_records),'physical_environment_steps':physical,'total_physical_environment_steps':sum(physical.values()),'tables':tables,'scope':protocol['purpose'],'limitations':protocol['scope_limits']}
    write(Path(out)/'summary.json',summary);return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    protocol=read(args.protocol);out=args.output;out.mkdir(parents=True,exist_ok=False);write(out/'protocol.json',protocol)
    root=Path('/root/autodl-tmp/colin_baselines');author=BASE.parent/'colin_pivot'
    assert subprocess.check_output(['git','-C',str(author),'rev-parse','HEAD'],text=True).strip()==protocol['author_source_commit']
    assert not subprocess.check_output(['git','-C',str(author),'status','--porcelain','--untracked-files=no'],text=True).strip()
    hashes={str(p):sha(p) for p in [Path(__file__),BASE/'sequential_validation.py',args.protocol]+[root/p for p in ['crossbench/experiment.py','crossbench/envs.py','pivot_v2/features.py','benchmark.py']]}
    write(out/'source_hashes.json',hashes)
    write(out/'environment.json',{'python':sys.version,'packages':{k:importlib.metadata.version(k) for k in ['numpy','mpe2','stable_baselines3','torch','gymnasium']}})
    state={'state':'running','started_at':now(),'completed':{},'failures':[]};write(out/'status.json',state)
    def stage(name,fn,specs):
        state.update(phase=name,phase_total=len(specs));state['completed'][name]=0;write(out/'status.json',state)
        with ProcessPoolExecutor(max_workers=protocol['limits']['max_workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(fn,s) for s in specs]
            for f in as_completed(futures):
                try:
                    result=f.result();state['completed'][name]+=1
                    with (out/'completed_jobs.jsonl').open('a') as log:log.write(json.dumps(result)+'\n')
                except Exception:state['failures'].append(traceback.format_exc())
                write(out/'status.json',state)
        if state['failures']:raise RuntimeError('Stage failed: '+name)
    try:
        cfg=protocol['config'];specs=[(t,s,cfg,str(out)) for t in protocol['tasks'] for s in protocol['test_seeds']]
        calibrate(protocol,out);stage('prepare',prepare,specs)
        from crossbench.experiment import Seeds
        def response_specs(stream):
            result=[]
            for t,s,c,o in specs:
                seeds=Seeds(panel(out,t,s)/'seeds.json',s)
                for j in range(4):
                    for r in range(c['selection_replicates_available'] if stream=='selection' else c['audit_replicates']):
                        training=seeds.get(stream,j,r,'training');es=[seeds.get(stream,j,r,'evaluation',e) for e in range(c['query_episodes'])]
                        result.append((t,s,c,o,stream,j,r,training,es))
            return result
        stage('selection_data',response,response_specs('selection'))
        stage('selection_freeze',select_panel,[tuple(s)+(protocol,) for s in specs])
        seals={str(panel(out,t,s)/'decisions_frozen.json'):sha(panel(out,t,s)/'decisions_frozen.json') for t,s,_,_ in specs}
        write(out/'global_selection_seal.json',{'frozen_at':now(),'audit_opened':False,'decision_hashes':seals})
        stage('independent_audit',response,response_specs('audit'))
        report(protocol,out)
        for p,h in hashes.items():assert sha(p)==h
        for p,h in seals.items():assert sha(p)==h
        state.update(state='complete',finished_at=now())
    except Exception:
        state.update(state='failed',finished_at=now());state['failures'].append(traceback.format_exc());raise
    finally:write(out/'status.json',state)

if __name__=='__main__':main()
