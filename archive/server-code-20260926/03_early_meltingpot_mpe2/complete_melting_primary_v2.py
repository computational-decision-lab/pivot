"""Complete only the prespecified short-response audit in the existing study.
Uses original training/evaluation functions and frozen decisions; no tuning.
The separate primary report does not claim that long-response audits finished.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS']:os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse,datetime,hashlib,json,math,multiprocessing,pathlib,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
BASE=pathlib.Path('/root/autodl-tmp/colin_melting')
STUDY=BASE/'runs/melting_cross_main'
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,x):pathlib.Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def job(spec):
    import torch
    torch.set_num_threads(1)
    from crossbench.experiment import train,load,evaluate,delta
    task,seed,j,r,cfg,training_seed,es=spec
    base=STUDY/task/'test'/f'seed_{seed}';folder=base/'audit'/f'candidate_{j}'/f'replicate_{r}';target=folder/'paired.json'
    if target.exists():return {'task':task,'seed':seed,'candidate':j,'replicate':r,'cached':True}
    start=time.monotonic();oldp=base/'initial/focal.zip';op=base/'initial/response.zip';newp=base/f'candidate_{j}.zip'
    old,new,opp=load(oldp),load(newp),load(op)
    oldresponse,am=train(task,'response',old,folder/'old_response',cfg['adapt_steps'],training_seed,cfg,op)
    newresponse,bm=train(task,'response',new,folder/'new_response',cfg['adapt_steps'],training_seed,cfg,op)
    ar=evaluate(task,old,load(oldresponse),es,cfg['horizon']);br=evaluate(task,new,load(newresponse),es,cfg['horizon'])
    fa=evaluate(task,old,opp,es,cfg['horizon']);fb=evaluate(task,new,opp,es,cfg['horizon'])
    crossed=evaluate(task,new,load(oldresponse),es,cfg['horizon'])
    row={'delta':delta(ar,br),'old':ar,'new':br,'candidate':j,'replicate':r,'training_seed':training_seed,'evaluation_seeds':es,'total_env_steps':am['actual_steps']+bm['actual_steps']+2*len(es)*cfg['horizon'],'frozen_delta':delta(fa,fb),'frozen_old':fa,'frozen_new':fb,'new_response_own_reward_gain':float(np.mean([x['response_return']-y['response_return'] for x,y in zip(br,fb)])),'new_against_old_response':crossed,'candidate_specific_effect':delta(crossed,br),'audit_extra_eval_steps':3*len(es)*cfg['horizon']}
    assert am['signature']['seed']==bm['signature']['seed']==training_seed
    assert am['signature']['start_hash']==bm['signature']['start_hash']==sha(op)
    assert np.isfinite(row['delta'])
    write(target,row)
    return {'task':task,'seed':seed,'candidate':j,'replicate':r,'cached':False,'seconds':round(time.monotonic()-start,3)}
def ci(values):
    x=np.array(values,float);rng=np.random.default_rng(20260915);d=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    return {'mean':float(x.mean()),'ci95':[float(v) for v in np.quantile(d,[.025,.975])],'n_training_seeds':len(x)}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=pathlib.Path,required=True);parser.add_argument('--report-only',action='store_true');args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    from crossbench.experiment import Seeds
    started=time.monotonic();specs=[];frozen={};input_hashes={}
    for task in ['melting_pd','melting_stag']:
        for seed in range(1300,1306):
            base=STUDY/task/'test'/f'seed_{seed}';manifest=read(base/'manifest.json');cfg=manifest['config'];frozen[str(base)]=sha(base/'decisions_frozen.json')
            for name,expected in manifest['source'].items():assert sha(BASE/name)==expected,(name,'source changed')
            for name in ['initial/focal.zip','initial/response.zip']+[f'candidate_{j}.zip' for j in range(cfg['candidates'])]:input_hashes[str(base/name)]=sha(base/name)
            seeds=Seeds(base/'seeds.json',seed)
            for j in range(cfg['candidates']):
                for r in range(cfg['audit_replicates']):
                    if (base/'audit'/f'candidate_{j}'/f'replicate_{r}'/'paired.json').exists():continue
                    training_seed=seeds.get('audit',j,r,'training');es=[seeds.get('audit',j,r,'evaluation',e) for e in range(cfg['query_episodes'])]
                    specs.append((task,seed,j,r,cfg,training_seed,es))
    if args.report_only and specs:raise RuntimeError('Primary audit is not complete; report-only cannot train')
    write(out/'protocol.json',{'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'completion of previously prespecified primary audit; all outcomes retained','missing_pairs':len(specs),'workers':12,'external_wall_limit_seconds':1200,'helper_sha256':sha(__file__),'decisions_before_sha256':frozen,'policy_before_sha256':input_hashes,'method_implementation':'historical independent implementation; not author PIVOT','long_response_audit_not_completed_here':True})
    progress={'state':'running','planned_missing_pairs':len(specs),'completed_pairs':0,'errors':[]};write(out/'status.json',progress)
    with ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn')) as pool:
        fs=[pool.submit(job,spec) for spec in specs]
        for f in as_completed(fs):
            try:
                result=f.result();progress['completed_pairs']+=1
                with (out/'completed.jsonl').open('a') as log:log.write(json.dumps(result)+'\n')
            except Exception as error:progress['errors'].append(type(error).__name__+': '+str(error))
            write(out/'status.json',progress)
    if progress['errors']:progress['state']='failed';write(out/'status.json',progress);raise RuntimeError('One or more primary audit tasks failed')
    for name,expected in input_hashes.items():assert sha(name)==expected
    tables={};all_records=[]
    for task in ['melting_pd','melting_stag']:
        task_records=[];mechanism=[]
        for seed in range(1300,1306):
            base=STUDY/task/'test'/f'seed_{seed}';cfg=read(base/'manifest.json')['config'];assert sha(base/'decisions_frozen.json')==frozen[str(base)]
            decisions=read(base/'decisions_frozen.json');selection_eval=set();selection_train=set()
            for f in (base/'selection').glob('candidate_*/replicate_*/paired.json'):
                q=read(f);selection_eval.update(q['evaluation_seeds']);selection_train.add(q['training_seed'])
            rows=[]
            for j in range(cfg['candidates']):
                group=[]
                for r in range(cfg['audit_replicates']):
                    q=read(base/'audit'/f'candidate_{j}'/f'replicate_{r}'/'paired.json')
                    assert not selection_eval.intersection(q['evaluation_seeds']) and q['training_seed'] not in selection_train
                    assert q['evaluation_seeds']==[a['seed'] for a in q['old']]==[a['seed'] for a in q['new']]
                    group.append(q)
                rows.append(group)
            gains=np.array([[q['delta'] for q in group] for group in rows]);fg=np.array([[q['frozen_delta'] for q in group] for group in rows])
            block=cfg['n_envs']*cfg['n_steps'];common=cfg['candidates']*math.ceil(cfg['candidate_steps']/block)*block+2*cfg['candidates']*cfg['proxy_episodes']*cfg['horizon'];footprint=(1+cfg['candidates'])*cfg['footprint_episodes']*cfg['horizon']
            for d in decisions:
                j=d['selected'];r={'task':task,'seed':seed,'method':d['method'],'budget':d['budget'],'selected':j,'audit_gain':float(gains[j].mean()),'frozen_gain':float(fg[j].mean()),'decision_cost':common+d['extra_env_steps']+(0 if d['method'] in ['proxy_only','pivot_no_footprint','paired_sample_mean'] else footprint),'query_cost':d['extra_env_steps']};task_records.append(r)
            flat=[q for group in rows for q in group]
            mechanism.append({'seed':seed,'response_effect':float(np.mean([q['delta']-q['frozen_delta'] for q in flat])),'candidate_specific_effect':float(np.mean([q['candidate_specific_effect'] for q in flat])),'response_own_reward_gain':float(np.mean([q['new_response_own_reward_gain'] for q in flat]))})
        methods=[]
        for key in sorted({(r['method'],r['budget']) for r in task_records}):
            rs=[r for r in task_records if (r['method'],r['budget'])==key];methods.append({'method':key[0],'budget':key[1],'audit_gain':ci([r['audit_gain'] for r in rs]),'mean_decision_cost':float(np.mean([r['decision_cost'] for r in rs]))})
        p={r['seed']:r for r in task_records if r['method']=='pivot_voi' and r['budget']==max(cfg['budgets'])};u={r['seed']:r for r in task_records if r['method']=='uniform_paired' and r['budget']==max(cfg['budgets'])}
        tables[task]={'n_seeds':6,'prespecified_primary_budget':max(cfg['budgets']),'methods':methods,'prespecified_primary_difference_pivot_minus_uniform':ci([p[s]['audit_gain']-u[s]['audit_gain'] for s in range(1300,1306)]),'mechanism':{key:ci([r[key] for r in mechanism]) for key in ['response_effect','candidate_specific_effect','response_own_reward_gain']}}
        all_records.extend(task_records)
        from scipy.stats import ttest_1samp
        primary=np.array([p[s]['audit_gain']-u[s]['audit_gain'] for s in range(1300,1306)])
        tables[task]['primary_approximate_two_sided_t_p']=float(ttest_1samp(primary,0).pvalue) if np.std(primary)>0 else 1.
    previous=0.
    for i,task in enumerate(sorted(tables,key=lambda t:tables[t]['primary_approximate_two_sided_t_p'])):
        previous=max(previous,min(1.,(len(tables)-i)*tables[task]['primary_approximate_two_sided_t_p']))
        tables[task]['primary_holm_p_across_two_tasks']=previous
    write(out/'scored_decisions.json',all_records);write(out/'summary.json',{'status':'complete','scope':'All 12 short-response test panels complete. Long audits remain incomplete.','tables':tables,'limitations':['Historical independent selector, not the author implementation.','Six training seeds per task; exploratory and limited precision.','Intervals descriptive; no uncorrected multiple-comparison significance claims.','Task learning and response quality must be considered.'],'seconds':round(time.monotonic()-started,3)})
    progress['state']='complete';progress['seconds']=round(time.monotonic()-started,3);write(out/'status.json',progress);print(json.dumps(progress),flush=True)
if __name__=='__main__':main()
