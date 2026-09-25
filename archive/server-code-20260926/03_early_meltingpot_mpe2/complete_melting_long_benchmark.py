"""Complete prespecified long responses and plot-ready cross-horizon evidence.
Author choices were frozen at the short horizon. Longer-response scores are a
deployment stress test of those same choices, not an oracle retuned to the audit.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse,datetime,hashlib,json,math,multiprocessing,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
BASE=Path('/root/autodl-tmp/colin_melting');STUDY=BASE/'runs/melting_cross_main'
CLOUD=Path(__file__).resolve().parent
TASKS=['melting_pd','melting_stag'];SEEDS=list(range(1300,1306))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def ci(x):
    a=np.array(x,float);rng=np.random.default_rng(20260915);samples=a[rng.integers(len(a),size=(10000,len(a)))].mean(1)
    return {'mean':float(a.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),'n_training_seeds':len(a)}

def quality(spec):
    from crossbench.experiment import load,evaluate,delta
    task,seed,cfg,es=spec;b=STUDY/task/'test'/f'seed_{seed}';target=b/'quality.json'
    if target.exists():return {'phase':'quality','task':task,'seed':seed,'cached':True,'env_steps':0}
    old=load(b/'initial/focal.zip');opp=load(b/'initial/response.zip');untrained=load(b/'initial/focal0_untrained.zip')
    a=evaluate(task,untrained,opp,es,cfg['horizon']);c=evaluate(task,old,opp,es,cfg['horizon'])
    write(target,{'untrained':a,'trained':c,'gain':delta(a,c)})
    return {'phase':'quality','task':task,'seed':seed,'env_steps':2*len(es)*cfg['horizon']}

def job(spec):
    from crossbench.experiment import train,load,evaluate,delta
    task,seed,j,r,cfg,training_seed,es=spec;b=STUDY/task/'test'/f'seed_{seed}';folder=b/'long_audit'/f'candidate_{j}'/f'replicate_{r}';target=folder/'paired.json'
    if target.exists():raise RuntimeError('Long response already exists unexpectedly')
    start=time.monotonic();old,new,opp=load(b/'initial/focal.zip'),load(b/f'candidate_{j}.zip'),load(b/'initial/response.zip')
    old_cached=(folder/'old_response.json').exists();new_cached=(folder/'new_response.json').exists()
    a,am=train(task,'response',old,folder/'old_response',cfg['long_adapt_steps'],training_seed,cfg,b/'initial/response.zip')
    c,cm=train(task,'response',new,folder/'new_response',cfg['long_adapt_steps'],training_seed,cfg,b/'initial/response.zip')
    ar=evaluate(task,old,load(a),es,cfg['horizon']);cr=evaluate(task,new,load(c),es,cfg['horizon'])
    fa=evaluate(task,old,opp,es,cfg['horizon']);fc=evaluate(task,new,opp,es,cfg['horizon']);crossed=evaluate(task,new,load(a),es,cfg['horizon'])
    assert am['signature']['seed']==cm['signature']['seed']==training_seed
    assert am['signature']['start_hash']==cm['signature']['start_hash']==sha(b/'initial/response.zip')
    row={'delta':delta(ar,cr),'old':ar,'new':cr,'candidate':j,'replicate':r,'training_seed':training_seed,'evaluation_seeds':es,
         'total_env_steps':am['actual_steps']+cm['actual_steps']+2*len(es)*cfg['horizon'],
         'frozen_delta':delta(fa,fc),'frozen_old':fa,'frozen_new':fc,
         'new_response_own_reward_gain':float(np.mean([x['response_return']-y['response_return'] for x,y in zip(cr,fc)])),
         'new_against_old_response':crossed,'candidate_specific_effect':delta(crossed,cr),'audit_extra_eval_steps':3*len(es)*cfg['horizon']}
    write(target,row)
    physical_training=(0 if old_cached else am['actual_steps'])+(0 if new_cached else cm['actual_steps'])
    return {'phase':'long_response','task':task,'seed':seed,'candidate':j,'replicate':r,'seconds':time.monotonic()-start,'env_steps':physical_training+5*len(es)*cfg['horizon'],'old_response_cached':old_cached,'new_response_cached':new_cached}

def report(out,decisions,seal):
    tables={};scored=[];candidate_rows=[];mechanism_rows=[];hashes={}
    for task in TASKS:
        task_decisions=[r for r in decisions if r['task']==task];mechanisms=[]
        for seed in SEEDS:
            b=STUDY/task/'test'/f'seed_{seed}';cfg=read(b/'manifest.json')['config'];query_training=set();query_evaluation=set()
            for p in (b/'selection').glob('candidate_*/replicate_*/paired.json'):
                q=read(p);query_training.add(q['training_seed']);query_evaluation.update(q['evaluation_seeds'])
            for stream,reps,steps in [('audit',cfg['audit_replicates'],cfg['adapt_steps']),('long_audit',cfg['long_audit_replicates'],cfg['long_adapt_steps'])]:
                audit=[]
                for j in range(4):
                    group=[]
                    for r in range(reps):
                        p=b/stream/f'candidate_{j}'/f'replicate_{r}'/'paired.json';q=read(p);hashes[str(p)]=sha(p)
                        assert q['training_seed'] not in query_training and not query_evaluation.intersection(q['evaluation_seeds'])
                        assert q['evaluation_seeds']==[x['seed'] for x in q['old']]==[x['seed'] for x in q['new']]
                        group.append(q)
                    audit.append(group)
                gains=np.array([[q['delta'] for q in group] for group in audit]);frozen=np.array([[q['frozen_delta'] for q in group] for group in audit])
                flat=[q for group in audit for q in group];proxy=np.array(read(b/'features.json'))[:,0]
                candidate_rows.append({'task':task,'seed':seed,'response_steps':steps,'proxy':proxy.tolist(),'audit_replicates':gains.tolist(),'frozen_replicates':frozen.tolist()})
                for d in task_decisions:
                    if d['seed']!=seed:continue
                    selected=d['selected'];scored.append(dict(d,response_steps=steps,audit_gain=float(gains[selected].mean()),frozen_gain=float(frozen[selected].mean()),audit_response_replicates=reps))
                mechanisms.append({'task':task,'seed':seed,'response_steps':steps,'response_effect':float(np.mean([q['delta']-q['frozen_delta'] for q in flat])),
                    'candidate_specific_effect':float(np.mean([q['candidate_specific_effect'] for q in flat])),
                    'response_own_reward_gain':float(np.mean([q['new_response_own_reward_gain'] for q in flat])),
                    'observed_proxy_sign_disagreement':float(np.mean((proxy>0)!=(gains.mean(1)>0))),
                    'observed_best_candidate_changed':bool(proxy.argmax()!=gains.mean(1).argmax())})
        method_rows=[]
        for steps,method,budget in sorted({(r['response_steps'],r['method'],r['budget_packages']) for r in scored if r['task']==task}):
            group=[r for r in scored if r['task']==task and r['response_steps']==steps and r['method']==method and r['budget_packages']==budget]
            method_rows.append({'response_steps':steps,'method':method,'budget_packages':budget,'audit_gain':ci([r['audit_gain'] for r in group]),'frozen_gain':ci([r['frozen_gain'] for r in group]),'mean_query_cost':float(np.mean([r['query_cost'] for r in group])),'mean_query_packages':float(np.mean([r['query_packages_used'] for r in group]))})
        tables[task]={'n_training_seeds':6,'methods':method_rows,'focal_quality_gain':ci([read(STUDY/task/'test'/f'seed_{s}'/'quality.json')['gain'] for s in SEEDS]),
            'mechanism_by_response_budget':{str(h):{key:ci([r[key] for r in mechanisms if r['response_steps']==h]) for key in ['response_effect','candidate_specific_effect','response_own_reward_gain','observed_proxy_sign_disagreement','observed_best_candidate_changed']} for h in sorted({r['response_steps'] for r in mechanisms})}}
        mechanism_rows.extend(mechanisms)
    assert sha(CLOUD/'author_melting/decisions_frozen.json')==seal
    jobs=[json.loads(x) for x in (out/'completed_jobs.jsonl').read_text().splitlines()] if (out/'completed_jobs.jsonl').exists() else []
    result={'status':'complete','platform':'Melting Pot','n_tasks':2,'n_test_panels':12,'short_pairs':192,'long_pairs':96,'n_scored_author_decisions_across_two_horizons':len(scored),'new_physical_environment_steps':sum(r['env_steps'] for r in jobs),'tables':tables,
      'limits':['Pre-existing candidate policies and short-horizon author choices; new work completes the prespecified long-response stress test.',
                'The author selector is calibrated and queried at the short response horizon; long scores are deployment-response mismatch tests, not long-target optimization.',
                'Four short and two long response replicates per candidate; six training seeds per task. Intervals are descriptive.',
                'Sample-mean sign or ranking changes are noisy observed indicators, not individually confirmed true reversals.',
                'Local 5x5 pooled RGB MLP PPO, finite 1000-frame episodes; not the complete official Melting Pot evaluation suite.']}
    write(out/'summary.json',result);write(out/'scored_decisions.json',scored);write(out/'candidate_audit.json',candidate_rows);write(out/'mechanism_by_seed.json',mechanism_rows);write(out/'audit_hashes.json',hashes)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    from crossbench.experiment import Seeds
    decisions=read(CLOUD/'author_melting/decisions_frozen.json');seal=sha(CLOUD/'author_melting/decisions_frozen.json')
    assert seal==read(CLOUD/'author_melting/selection_seal.json')['decisions_sha256']
    specs=[];quality_specs=[];models={};sources={}
    for task in TASKS:
        for seed in SEEDS:
            b=STUDY/task/'test'/f'seed_{seed}';manifest=read(b/'manifest.json');cfg=manifest['config'];seeds=Seeds(b/'seeds.json',seed)
            for path,digest in manifest['source'].items():assert sha(BASE/path)==digest;sources[str(BASE/path)]=digest
            for path in [b/'initial/focal.zip',b/'initial/response.zip']+[b/f'candidate_{j}.zip' for j in range(4)]:models[str(path)]=sha(path)
            es=[seeds.get('quality',e) for e in range(cfg['query_episodes'])];quality_specs.append((task,seed,cfg,es))
            for j in range(4):
                for r in range(cfg['long_audit_replicates']):
                    if (b/'long_audit'/f'candidate_{j}'/f'replicate_{r}'/'paired.json').exists():continue
                    training_seed=seeds.get('long_audit',j,r,'training');es=[seeds.get('long_audit',j,r,'evaluation',e) for e in range(cfg['query_episodes'])]
                    specs.append((task,seed,j,r,cfg,training_seed,es))
    write(out/'protocol.json',{'started_at':now(),'purpose':'Complete prespecified Melting Pot long responses; author PIVOT benchmark and mechanism/cost/horizon figures, without requiring method superiority','missing_long_pairs':len(specs),'workers':8,'wall_limit_seconds':2700,'author_choices_sha256':seal,'models_before':models,'source_hashes':sources,'helper_sha256':sha(__file__)})
    state={'state':'running','started_at':now(),'completed_quality':0,'completed_long_pairs':0,'planned_long_pairs':len(specs),'failures':[]};write(out/'status.json',state)
    try:
        with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
            for phase,fn,jobs in [('quality',quality,quality_specs),('long_response',job,specs)]:
                state['phase']=phase;write(out/'status.json',state)
                for f in as_completed([pool.submit(fn,x) for x in jobs]):
                    try:
                        r=f.result();state['completed_quality' if phase=='quality' else 'completed_long_pairs']+=1
                        with (out/'completed_jobs.jsonl').open('a') as log:log.write(json.dumps(r)+'\n')
                    except Exception:state['failures'].append(traceback.format_exc())
                    write(out/'status.json',state)
        if state['failures']:raise RuntimeError('Some Melting Pot jobs failed')
        for p,h in models.items():assert sha(p)==h
        report(out,decisions,seal);state.update(state='complete',finished_at=now())
    except Exception:
        state.update(state='failed',finished_at=now());state['failures'].append(traceback.format_exc());raise
    finally:write(out/'status.json',state)

if __name__=='__main__':main()
