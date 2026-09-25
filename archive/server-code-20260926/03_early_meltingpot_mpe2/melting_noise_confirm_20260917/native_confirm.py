"""Native fixed-candidate method experiment, conditional on frozen response worlds.

This is an explicit Melting Pot adaptive extension. Candidates are prespecified
skill-mixture edits, not new end-to-end PPO policies. Each candidate and the
incumbent induce their own reward-trained response from identical initialization.
Short/long are prefixes. HF observations and held-out audit use independent
native evaluation contexts. Response worlds are fixed before evaluation.
"""
from __future__ import annotations
import argparse,copy,hashlib,inspect,json,math,time
from pathlib import Path
import numpy as np
import melting_hierarchical_e5_stratified as v2
from pivot.acquisition.pivot_voi import BayesianLinearDeltaPosterior
from experiments.v9 import e5c_efficiency as author
from sequential_pivot_extension import run as sequential, METHODS

VERSION='melting_native_method_v3'
# Keep the native seed namespace above unchanged. This is a new cohort/adapter,
# not a new environment, posterior family, candidate generator, or PIVOT-CG.
CONFIRM_VERSION='melting_native_likelihood_noise_only_confirm_v1'
LIKELIHOOD_NOISE={'4':4.255427572511991,'32':7.746457798054848}
NOISE_METHOD_NAMES={
    'pivot_sequential':'noise_only_pivot_sequential_fixed',
    'uniform_random_matched':'noise_only_uniform_random_matched',
    'global_ivr_matched':'noise_only_global_ivr_matched',
    'posterior_lucb_matched':'noise_only_posterior_lucb_matched',
}
GRID=[.125,.225,.325,.425,.575,.675,.775,.875]
def logit(p):return math.log(p/(1-p))
def features(p):return [(p-.5)/.25,abs(p-.5)/.25]

def likelihood_only(posterior,variance):
    if not math.isfinite(float(variance)) or variance<=0:raise ValueError('positive finite noise required')
    changed=copy.deepcopy(posterior);changed.noise_variance=float(variance)
    assert np.array_equal(changed.mean,posterior.mean) and np.array_equal(changed.covariance,posterior.covariance)
    assert changed.prior_precision==posterior.prior_precision and changed.n_observations==posterior.n_observations
    return changed

def noise_only_decisions(rows,posterior,query,*,budget,seed,adaptation,cap):
    """Same query callback and seed; no audit inputs; no posterior refit."""
    changed=likelihood_only(posterior,LIKELIHOOD_NOISE[str(adaptation)])
    decisions=[]
    for method,label in NOISE_METHOD_NAMES.items():
        d=sequential(rows,changed,query,method=method,budget=budget,seed=seed,stop=False)
        d['method']=label
        d.update(likelihood_noise=changed.noise_variance,original_likelihood_noise=posterior.noise_variance,
                 initial_mean_covariance_unchanged=True,base_selector_method=method)
        decisions.append(dict(d,adaptation=adaptation,budget_cap=cap,query_limit=budget,incumbent_setup_cost=adaptation if d['hf_queries'] else 0))
    # Stopping is kept as a clearly secondary control, never the primary test.
    d=sequential(rows,changed,query,method='pivot_sequential',budget=budget,seed=seed,stop=True)
    d['method']='noise_only_pivot_sequential_adaptive_stop'
    d.update(likelihood_noise=changed.noise_variance,original_likelihood_noise=posterior.noise_variance,
             initial_mean_covariance_unchanged=True,base_selector_method='pivot_sequential')
    decisions.append(dict(d,adaptation=adaptation,budget_cap=cap,query_limit=budget,incumbent_setup_cost=adaptation if d['hf_queries'] else 0))
    return decisions

def unchanged_or_write(path,value,resume):
    if path.exists():
        if not resume or v2.read(path)!=value:raise ValueError(f'Existing frozen identity changed: {path}')
    else:v2.write(path,value)

def preserve_interrupted_costs(out):
    """Retain every partial attempt, but keep completed-task ledger comparable."""
    path=out/'episode_costs.json'
    if not path.exists():return
    costs=v2.read(path);partial={name:row for name,row in costs.items() if not row['complete']}
    if not partial:return
    snapshots=out/'resume_attempts';snapshots.mkdir(exist_ok=True)
    v2.write(snapshots/f'ledger_before_resume_{len(list(snapshots.glob("ledger_before_resume_*.json"))):03d}.json',costs)
    overhead_path=out/'interrupted_episode_costs.json'
    overhead=v2.read(overhead_path) if overhead_path.exists() else {}
    for name,row in partial.items():
        if not (out/name).is_file():raise ValueError('missing partial native episode')
        if name in overhead and overhead[name]!=row:raise ValueError('partial native overhead changed')
        overhead[name]=row
    v2.write(overhead_path,overhead)
    v2.write(path,{name:row for name,row in costs.items() if row['complete']})

def fit_calibration(folder,h):
    """Frozen discovery data reused for calibration only; no test labels."""
    summary=v2.read(folder/'analysis/summary.json');assert summary['status']=='PASS'
    X=[];y=[];hashes={}
    for seed in range(43000,43012):
        path=folder/f'seed_{seed}/results.json';hashes[str(path)]=v2.sha(path)
        rows=v2.read(path);r={(x['target_probability'],x['eval_block']):x for x in rows if x['training_episodes']==h}
        for p in [.25,.75]:
            gap=np.mean([(r[p,b]['focal_return']-r[.5,b]['focal_return'])-(r[p,b]['frozen_focal_return']-r[.5,b]['frozen_focal_return']) for b in ['A','B']])
            X.append(features(p));y.append(gap)
    # Same noise rule as author E5C, with domain features vanishing at no update.
    post=BayesianLinearDeltaPosterior(noise_variance=max(float(np.var(y)),1e-4)).fit(np.array(X),np.array(y))
    return post,{'mean':post.mean.tolist(),'covariance':post.covariance.tolist(),'noise_variance':post.noise_variance,
        'observations':24,'independent_calibration_seeds':12,'input_hashes':hashes,
        'feature_rule':'[(p-p_old)/0.25, abs(p-p_old)/0.25]; no intercept enforces delta correction = 0 for no update',
        'noise_rule':'author E5C max(population variance of calibration corrections, 1e-4)',
        'posterior_caveat':'Calibration rows are paired within 12 seeds. Gaussian posterior confidence is model-based; empirical held-out regret uses root-seed inference.'}

def batch_author(rows,post,query,method,budget,seed):
    clean=[{k:r[k] for k in ['transition_id','delta_proxy','features','hf_query_cost']} for r in rows]
    selected=author._select(method,clean,post,min(budget,len(clean)),seed,{'statistics':{'voi_fantasies':64,'voi_posterior_samples':256}})
    observed={j:float(query(next(r for r in clean if r['transition_id']==j))['delta']) for j in selected}
    incumbent={'transition_id':'incumbent','delta_proxy':0.,'features':[0.,0.],'hf_query_cost':1.}
    outcome=[dict(r,delta_true=observed.get(r['transition_id'],float('nan'))) for r in clean+[incumbent]]
    pick,_=author._select_outcome(outcome,selected,method,post)
    return {'version':'author_E5C_allocation_with_explicit_incumbent_adapter','method':'author_'+method,'selected_id':pick,
        'queried_ids':selected,'hf_queries':len(selected),'charged_cost':sum(r['hf_query_cost'] for r in clean if r['transition_id'] in observed),
        'observed':observed,'real_query_posterior_update':False,'audit_used':False}

def execute(a,backend):
    out=a.output;resuming=bool(getattr(a,'resume',False));out.mkdir(parents=True,exist_ok=resuming);protocol=v2.read(a.protocol)
    assert protocol['status']=='FROZEN' and protocol['candidate_probabilities']==GRID
    assert a.seed in protocol['test_seeds'] and a.seed not in range(43000,43012)
    assert protocol['test_seeds']==list(range(47000,47030))
    assert protocol['likelihood_noise_variance']==LIKELIHOOD_NOISE
    assert protocol['adaptation_episodes']==[4,32] and protocol['hf_episode_caps']==[96,192,384]
    assert protocol['primary_hf_episode_cap']==192 and protocol['seed_extension'] is False
    c={'task':'melting_stag','seed':a.seed,'horizon':2500,'reward_scale':100.,'reward_clip':2.,'logit_clip':8.,
       'baseline':'running_mean','stratum_replicates':8,'proxy_stratum_replicates':8}
    unchanged_or_write(out/'protocol.json',{'version':VERSION,'seed':a.seed,'config':c,'batch_protocol_sha256':v2.sha(a.protocol),'backend':backend.metadata},resuming)
    import melting_method_native_v3 as original_native
    import sequential_pivot_extension as original_selector
    manifest={'schema_version':1,'confirm_version':CONFIRM_VERSION,'seed_namespace':VERSION,
        'original_worker_source':str(Path(original_native.__file__).resolve()),'original_worker_sha256':v2.sha(Path(original_native.__file__)),
        'adapter_sha256':v2.sha(Path(__file__)),'selector_sha256':v2.sha(Path(original_selector.__file__)),
        'author_acquisition_sha256':v2.sha(Path(inspect.getfile(BayesianLinearDeltaPosterior))),
        'author_e5c_sha256':v2.sha(Path(author.__file__)),'batch_protocol_sha256':v2.sha(a.protocol),
        'likelihood_noise_variance':LIKELIHOOD_NOISE,'original_branches_preserved':True,
        'scope':'Original author E5C branches retained; explicit sequential extension gains noise-only arms. Not PIVOT-CG.',
        'schema_files':{'decisions_frozen.json':'all method decisions before independent audit','posterior.json':'unchanged original fitted posterior','noise_only_posterior.json':'same fitted mean/covariance, only HF likelihood scalar replaced','scored_decisions.json':'independent-audit scoring after selection seal'}}
    unchanged_or_write(out/'implementation_manifest.json',manifest,resuming)
    unchanged_or_write(out/'candidates.json',{'probabilities':GRID,'incumbent':.5,'operator':'fixed symmetric mixture edits, no outcome-dependent proposal selection'},resuming)
    if resuming:preserve_interrupted_costs(out)
    episodes=v2.Episodes(out,backend,c,a.max_seconds,2500*704);start=time.monotonic();histories={};q={}
    for j,p in [('incumbent',.5)]+[(str(i),p) for i,p in enumerate(GRID)]:
        v2.write(out/'status.json',{'status':'response_training','target':j})
        key=('response_training',j)
        v2.train_logit(episodes,role='response',initial_logit=0.,other_logit=logit(p),key=key,template=(VERSION,'response_training'),count=32,learning_rate=.5)
        path=out/'training'/(v2.digest(key)[:24]+'.json');history=v2.read(path);assert history['complete']
        histories[str(path.relative_to(out))]=v2.sha(path)
        q[j]={h:v2.probability(history['history'][h-1]['logit_after']) for h in [4,32]}
    v2.write(out/'training_seal.json',{'history_hashes':histories,'response_probabilities':q,'all_frozen_before_proxy':True})
    seal_hash=v2.sha(out/'training_seal.json')
    def block(stage,key):
        context={'training_seal_sha256':seal_hash,'candidate_sha256':v2.sha(out/'candidates.json')}
        return v2.evaluation_block(episodes,(stage,key),(VERSION,stage,key),out/stage/(str(key)+'.json'),context)
    proxy=block('proxy','common');old=v2.weighted_estimate(proxy,.5,.5)['focal_return']
    proxies={str(i):v2.weighted_estimate(proxy,p,.5)['focal_return']-old for i,p in enumerate(GRID)}
    v2.write(out/'features.json',[{'id':str(i),'p':p,'features':features(p),'proxy_delta':proxies[str(i)]} for i,p in enumerate(GRID)])
    posteriors={};post_metadata={}
    for h in [4,32]:posteriors[h],post_metadata[str(h)]=fit_calibration(a.calibration,h)
    v2.write(out/'posterior.json',post_metadata)
    v2.write(out/'noise_only_posterior.json',{str(h):{'mean':posteriors[h].mean.tolist(),'covariance':posteriors[h].covariance.tolist(),
        'noise_variance':LIKELIHOOD_NOISE[str(h)],'original_noise_variance':posteriors[h].noise_variance,
        'n_observations':posteriors[h].n_observations,'only_changed_field':'noise_variance','calibration_source':'eight independent development measurement worlds44000–44007, no fresh cohort labels'} for h in [4,32]})
    decisions=[]
    for h in [4,32]:
        cost=h+4*c['stratum_replicates']
        rows=[{'transition_id':str(i),'delta_proxy':proxies[str(i)],'features':features(p),'hf_query_cost':cost} for i,p in enumerate(GRID)]
        def query(row):
            j=row['transition_id'];i=int(j);b=block('selection',j)
            value=v2.weighted_estimate(b,GRID[i],q[j][h])['focal_return']-v2.weighted_estimate(b,.5,q['incumbent'][h])['focal_return']
            return {'delta':value,'split':'selection','hf_query_cost':cost}
        for cap in protocol['hf_episode_caps']:
            budget=min(8,max(0,(cap-h)//cost))
            for method in METHODS:
                if method in ['proxy_only','calibrated_no_hf'] and cap!=protocol['hf_episode_caps'][0]:continue
                d=sequential(rows,posteriors[h],query,method=method,budget=budget,seed=a.seed,stop=False)
                decisions.append(dict(d,adaptation=h,budget_cap=cap,query_limit=budget,incumbent_setup_cost=h if d['hf_queries'] else 0))
            d=sequential(rows,posteriors[h],query,method='pivot_sequential',budget=budget,seed=a.seed,stop=True)
            d['method']='pivot_sequential_adaptive_stop';decisions.append(dict(d,adaptation=h,budget_cap=cap,query_limit=budget,incumbent_setup_cost=h if d['hf_queries'] else 0))
            for method in ['random_hf','paired_lucb','global_voi','pivot_voi']:
                d=batch_author(rows,posteriors[h],query,method,budget,a.seed)
                decisions.append(dict(d,adaptation=h,budget_cap=cap,query_limit=budget,incumbent_setup_cost=h if d['hf_queries'] else 0))
            decisions.extend(noise_only_decisions(rows,posteriors[h],query,budget=budget,seed=a.seed,adaptation=h,cap=cap))
        all_values={r['transition_id']:query(r)['delta'] for r in rows};all_values['incumbent']=0.
        decisions.append({'version':VERSION,'method':'all_hf_reference','adaptation':h,'budget_cap':None,'selected_id':max(all_values,key=all_values.get),
            'charged_cost':8*cost,'incumbent_setup_cost':h,'hf_queries':8,'queried_ids':list(proxies),'observed':all_values})
    for d in decisions:
        d['hf_episode_cost']=d['charged_cost']+d['incumbent_setup_cost']
        assert d['budget_cap'] is None or d['hf_episode_cost']<=d['budget_cap']
    v2.write(out/'decisions_frozen.json',decisions)
    v2.write(out/'selection_seal.json',{'decisions_sha256':v2.sha(out/'decisions_frozen.json'),'candidate_sha256':v2.sha(out/'candidates.json'),
        'features_sha256':v2.sha(out/'features.json'),'posterior_sha256':v2.sha(out/'posterior.json'),'audit_generated':False,'training_seal_sha256':seal_hash,
        'noise_only_posterior_sha256':v2.sha(out/'noise_only_posterior.json'),'implementation_manifest_sha256':v2.sha(out/'implementation_manifest.json')})
    v2.write(out/'status.json',{'status':'held_out_audit'})
    # Two 64-episode audit blocks. Frozen response worlds, fresh evaluation noise.
    episodes.config={**c,'stratum_replicates':16}
    audits=[block('audit',b) for b in ['A','B']]
    v2.verify_seal(out);assert v2.sha(out/'training_seal.json')==seal_hash
    selection_seal=v2.read(out/'selection_seal.json')
    for name in ['noise_only_posterior','implementation_manifest']:assert v2.sha(out/(name+'.json'))==selection_seal[name+'_sha256']
    gains={};mechanism=[]
    for h in [4,32]:
        gains[h]={'incumbent':0.}
        for i,p in enumerate(GRID):
            j=str(i);gains[h][j]=float(np.mean([v2.weighted_estimate(b,p,q[j][h])['focal_return']-v2.weighted_estimate(b,.5,q['incumbent'][h])['focal_return'] for b in audits]))
            for bname,b in zip(['A','B'],audits):
                new=v2.weighted_estimate(b,p,q[j][h]);old=v2.weighted_estimate(b,.5,q['incumbent'][h])
                fixed_new=v2.weighted_estimate(b,p,.5);fixed_old=v2.weighted_estimate(b,.5,.5)
                crossed=v2.weighted_estimate(b,p,q['incumbent'][h])
                proxy_delta=fixed_new['focal_return']-fixed_old['focal_return'];deployment_delta=new['focal_return']-old['focal_return']
                mechanism.append({'candidate':j,'probability':p,'adaptation':h,'block':bname,'proxy_delta_audit':proxy_delta,
                    'deployment_delta_audit':deployment_delta,'gap':deployment_delta-proxy_delta,
                    'responder_own_gain':new['response_return']-fixed_new['response_return'],
                    'candidate_specific_response_effect':new['focal_return']-crossed['focal_return'],
                    'common_opponent_adaptation_effect':crossed['focal_return']-old['focal_return']-proxy_delta})
    scored=[dict(d,audit_gain=gains[d['adaptation']][d['selected_id']],noisy_audit_isr=max(gains[d['adaptation']].values())-gains[d['adaptation']][d['selected_id']]) for d in decisions]
    v2.write(out/'scored_decisions.json',scored)
    assert len(episodes.costs)==704 and all(r['complete'] for r in episodes.costs.values())
    summary={'status':'complete','version':VERSION,'confirm_version':CONFIRM_VERSION,'seed':a.seed,'actual_native_episodes':len(episodes.costs),'actual_native_frames':episodes.frames,
        'duration_seconds':time.monotonic()-start,'source_protocol_sha256':v2.sha(a.protocol),'audit_gains':gains,'mechanism_audit':mechanism,'scored_decisions_sha256':v2.sha(out/'scored_decisions.json'),
        'scope':'Single-round conditional response-world ISR; eight fixed skill-mixture updates, frozen official network; not end-to-end PPO or multi-round CISR.',
        'cost_note':'Common incumbent adaptation charged once if any query; each queried candidate charges its adaptation plus 32 fresh native evaluation episodes. Shared short/long training prefixes, method-cache reuse, calibration and independent audit reported separately.',
        'audit_note':'Audit contexts independent of proxy/HF selection. Response policies frozen before evaluation; root-seed inference covers response-training randomness. Max-based audit ISR is noisy; paired method contrasts cancel the common max.'}
    overhead=v2.read(out/'interrupted_episode_costs.json') if (out/'interrupted_episode_costs.json').exists() else {}
    summary.update(interrupted_episode_attempts=len(overhead),interrupted_native_frames=sum(r['env_steps'] for r in overhead.values()),
        total_physical_native_frames_including_interruptions=episodes.frames+sum(r['env_steps'] for r in overhead.values()))
    v2.write(out/'summary.json',summary);v2.write(out/'status.json',{'status':'complete'});return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True);p.add_argument('--model-path',type=Path,required=True);p.add_argument('--crossbench-root',type=Path,required=True);p.add_argument('--max-seconds',type=int,default=14400);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    from melting_reference_diagnostic import OfficialSpecialistRollout
    engine=OfficialSpecialistRollout(a.model_path,horizon=2500,crossbench_root=a.crossbench_root)
    try:print(json.dumps(execute(a,engine)),flush=True)
    except v2.BudgetStop as exc:
        v2.write(a.output/'status.json',{'status':'timeout','error':str(exc),'partial_evidence_retained':True,'resume_same_frozen_seed_allowed':True})
        raise SystemExit(124)
    except Exception as exc:
        v2.write(a.output/'status.json',{'status':'failed','error':f'{type(exc).__name__}: {exc}','partial_evidence_retained':True})
        raise
    finally:engine.close()
if __name__=='__main__':main()
