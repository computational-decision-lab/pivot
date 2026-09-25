"""Two-phase author-selector port to fixed Melting Pot policy banks.
No training or test-driven tuning. Selection can finish before missing audits.
"""
import argparse,hashlib,json,math,pathlib,subprocess
import numpy as np
from pivot.acquisition.pivot_voi import BayesianLinearDeltaPosterior,select_pivot_voi
from pivot.algorithms.pivot import run_pivot_round,run_pivot_voi_round
COMMIT='9e3be723dbe895101c13dd3a6782d3e058b91558'
TASKS=['melting_pd','melting_stag']
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,x):pathlib.Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def ci(values):
    x=np.array(values,float);rng=np.random.default_rng(20260915);means=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    return {'mean':float(x.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'n_seeds':len(x)}
def select(a):
    a.output.mkdir(parents=True,exist_ok=False)
    assert subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(a.repo),'status','--porcelain','--untracked-files=no'],text=True).strip()
    protocol={'purpose':'development port of unchanged author selectors to preexisting Melting Pot policy banks','source_commit':COMMIT,'adapter_sha256':sha(__file__),'calibration_seeds':list(range(1200,1204)),'test_seeds':list(range(1300,1306)),'tasks':TASKS,'query_package_replicates':2,'budgets_packages':[1,2,4],'prior_precision':1.,'stop_delta':.05,'stop_eta':0.,'fantasies':64,'posterior_samples':256,'feature_rule':'intercept plus seven task-calibration-standardized footprints','fit_noise':'mean calibration replicate variance /4','query_noise':'mean calibration replicate variance /2; covariance retained from calibration fit','selection_reads_audit':False,'no_new_training':True,'new_confirmatory_claim':False}
    write(a.output/'protocol.json',protocol);decisions=[];hashes={};isolation={}
    def load(p):hashes[str(p.relative_to(a.base))]=sha(p);return read(p)
    for task in TASKS:
        calibration=[]
        for seed in range(1200,1204):calibration.extend(load(a.base/task/'calibration'/f'seed_{seed}'/'summary.json')['rows'])
        assert len(calibration)==16
        raw=np.array([r['features'][1:] for r in calibration]);mu=raw.mean(0);sd=raw.std(0);sd[sd<1e-8]=1
        raw_noise=max(float(np.mean([r['noise'] for r in calibration])),1e-6)
        posterior=BayesianLinearDeltaPosterior(prior_precision=1.,noise_variance=raw_noise/4).fit(np.c_[np.ones(16),(raw-mu)/sd],np.array([r['target_correction'] for r in calibration]))
        posterior.noise_variance=raw_noise/2
        write(a.output/(task+'_posterior.json'),{'mean':posterior.mean.tolist(),'covariance':posterior.covariance.tolist(),'training_mean_noise':raw_noise/4,'query_mean_noise':raw_noise/2,'feature_mean':mu.tolist(),'feature_scale':sd.tolist()})
        for seed in range(1300,1306):
            base=a.base/task/'test'/f'seed_{seed}';manifest=load(base/'manifest.json');cfg=manifest['config'];assert cfg['budgets']==[4,8] and cfg['calibration_replicates']==4
            raw=np.array(load(base/'features.json'));z=np.c_[np.ones(4),(raw[:,1:]-mu)/sd]
            block=cfg['n_envs']*cfg['n_steps'];singlecost=2*math.ceil(cfg['adapt_steps']/block)*block+2*cfg['query_episodes']*cfg['horizon'];packagecost=2*singlecost
            common=4*math.ceil(cfg['candidate_steps']/block)*block+2*4*cfg['proxy_episodes']*cfg['horizon'];fc=5*cfg['footprint_episodes']*cfg['horizon']
            candidates=[{'transition_id':str(j),'delta_proxy':float(raw[j,0]),'features':z[j].tolist(),'hf_query_cost':packagecost} for j in range(4)]
            query_seen=[];evaluation=set();training=set()
            def hf(candidate):
                j=int(candidate['transition_id']);rows=[]
                for r in range(2):
                    q=load(base/'selection'/f'candidate_{j}'/f'replicate_{r}'/'paired.json');assert q['total_env_steps']==singlecost
                    assert q['evaluation_seeds']==[x['seed'] for x in q['old']]==[x['seed'] for x in q['new']]
                    assert np.isclose(q['delta'],np.mean([n['focal_return']-o['focal_return'] for o,n in zip(q['old'],q['new'])]))
                    evaluation.update(q['evaluation_seeds']);training.add(q['training_seed']);rows.append(q)
                query_seen.append(j);return {'delta_true':float(np.mean([r['delta'] for r in rows])),'hf_query_cost':packagecost}
            def save(name,budget,result):
                assert len(query_seen)==len(set(query_seen))==result.hf_budget
                assert result.hf_cost==len(query_seen)*packagecost
                decisions.append({'task':task,'seed':seed,'method':name,'budget_packages':budget,'selected':int(result.selected_candidate_id),'query_packages_used':result.hf_budget,'query_cost':result.hf_cost,'decision_cost':common+fc+result.hf_cost,'queried_ids':list(result.queried_ids),'stop_reason':result.stop_reason});query_seen.clear()
            decisions.append({'task':task,'seed':seed,'method':'proxy_only','budget_packages':0,'selected':int(np.argmax(raw[:,0])),'query_packages_used':0,'query_cost':0,'decision_cost':common,'queried_ids':[],'stop_reason':'proxy_only'})
            save('correction_only',0,run_pivot_round(None,candidates,None,hf,lambda rows,model,budget:[],0,model=posterior))
            rngseed=int(hashlib.sha256((task+str(seed)).encode()).hexdigest()[:8],16);order=np.random.default_rng(rngseed).permutation(4).tolist()
            for budget in [1,2,4]:
                save('random_batch',budget,run_pivot_round(None,candidates,None,hf,lambda rows,model,budget:[str(j) for j in order[:budget]],budget,model=posterior))
                save('top_proxy_batch',budget,run_pivot_round(None,candidates,None,hf,lambda rows,model,budget:[str(j) for j in np.argsort(-raw[:,0],kind='stable')[:budget]],budget,model=posterior))
                save('author_pivot_no_stop',budget,run_pivot_round(None,candidates,None,hf,select_pivot_voi,budget,model=posterior,acquisition_kwargs={'seed':seed,'fantasies':64,'posterior_samples':256}))
                save('author_pivot_voi',budget,run_pivot_voi_round(None,candidates,hf,posterior,budget,seed=seed,delta=.05,eta=0.,fantasies=64,posterior_samples=256))
            isolation[task+'/'+str(seed)]={'selection_evaluation_seeds':sorted(evaluation),'selection_training_seeds':sorted(training)}
    write(a.output/'decisions_frozen.json',decisions);write(a.output/'selection_input_hashes.json',hashes);write(a.output/'isolation.json',isolation)
    write(a.output/'selection_seal.json',{'decisions_sha256':sha(a.output/'decisions_frozen.json'),'audit_opened':False,'n_decisions':len(decisions),'new_training':False});print(json.dumps({'phase':'selection_complete','n_decisions':len(decisions)}),flush=True)
def score(a):
    assert not (a.output/'summary.json').exists()
    seal=read(a.output/'selection_seal.json');assert sha(a.output/'decisions_frozen.json')==seal['decisions_sha256'];records=read(a.output/'decisions_frozen.json');isolation=read(a.output/'isolation.json');audit_hashes={};summary={}
    for task in TASKS:
        for seed in range(1300,1306):
            base=a.base/task/'test'/f'seed_{seed}';iso=isolation[task+'/'+str(seed)];gains=[]
            for j in range(4):
                group=[]
                for r in range(4):
                    p=base/'audit'/f'candidate_{j}'/f'replicate_{r}'/'paired.json';q=read(p);audit_hashes[str(p.relative_to(a.base))]=sha(p)
                    assert not set(q['evaluation_seeds']).intersection(iso['selection_evaluation_seeds']) and q['training_seed'] not in iso['selection_training_seeds'];group.append(q['delta'])
                gains.append(group)
            gains=np.array(gains)
            for row in records:
                if row['task']==task and row['seed']==seed:row['audit_gain']=float(gains[row['selected']].mean())
        taskrows=[r for r in records if r['task']==task];methods=[]
        for key in sorted({(r['method'],r['budget_packages']) for r in taskrows}):
            rs=[r for r in taskrows if (r['method'],r['budget_packages'])==key];methods.append({'method':key[0],'budget_packages':key[1],'audit_gain':ci([r['audit_gain'] for r in rs]),'mean_query_packages':float(np.mean([r['query_packages_used'] for r in rs])),'mean_decision_cost':float(np.mean([r['decision_cost'] for r in rs]))})
        contrasts=[]
        for budget in [1,2,4]:
            diff=[];matched=[]
            for seed in range(1300,1306):
                get=lambda name:next(r for r in taskrows if r['seed']==seed and r['method']==name and r['budget_packages']==(0 if name=='correction_only' else budget))
                p=get('author_pivot_voi');r=get('random_batch');zero=get('correction_only');same=r if p['query_packages_used'] else zero
                assert p['query_cost']==same['query_cost'];diff.append(p['audit_gain']-r['audit_gain']);matched.append(p['audit_gain']-same['audit_gain'])
            contrasts.append({'budget_packages':budget,'same_cap_pivot_minus_random':ci(diff),'same_stop_actual_cost_matched_pivot_minus_random':ci(matched)})
        summary[task]={'n_training_seeds':6,'methods':methods,'contrasts':contrasts}
    assert sha(a.output/'decisions_frozen.json')==seal['decisions_sha256'];write(a.output/'scored_decisions.json',records);write(a.output/'audit_input_hashes.json',audit_hashes)
    write(a.output/'summary.json',{'status':'complete','purpose':'development port of unchanged author selectors','n_decisions':len(records),'audit_isolated':True,'tasks':summary,'limitations':['Six preexisting trained seeds per substrate; no new confirmatory success claim.','MPE-style footprints with only four calibration seeds per substrate.','Short-response results only; native environment with documented seeding patch and 1000-frame cap.','Raw rewards are not pooled across tasks.','Displayed intervals exploratory and unadjusted.']});print(json.dumps({'phase':'scoring_complete','n_decisions':len(records)}),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['select','score'],required=True);p.add_argument('--base',type=pathlib.Path,required=True);p.add_argument('--repo',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();(select if a.phase=='select' else score)(a)
if __name__=='__main__':main()
