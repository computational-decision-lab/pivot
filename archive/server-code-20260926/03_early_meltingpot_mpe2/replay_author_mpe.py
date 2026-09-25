"""Development replay of unchanged upstream selectors on an existing MPE panel.

Selection uses only the historical selection split. An explicit second phase
opens audit labels after all choices have been written and hashed. This is an
adapter experiment, not an exact replication of an upstream environment.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,math,os,pathlib,subprocess,time
import numpy as np
from pivot.acquisition.pivot_voi import BayesianLinearDeltaPosterior,select_pivot_voi
from pivot.algorithms.pivot import run_pivot_round,run_pivot_voi_round
COMMIT='9e3be723dbe895101c13dd3a6782d3e058b91558'

def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,x):pathlib.Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def ci(x,seed=20260915):
    x=np.asarray(x,float);rng=np.random.default_rng(seed)
    d=np.mean(x[rng.integers(len(x),size=(10000,len(x)))],axis=1)
    return {'mean':float(x.mean()),'ci95':[float(z) for z in np.quantile(d,[.025,.975])],'n_seeds':len(x)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--repo',type=pathlib.Path,required=True);a=p.parse_args()
    started=time.monotonic();out=a.output;out.mkdir(parents=True,exist_ok=False)
    assert subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(a.repo),'status','--porcelain','--untracked-files=no'],text=True).strip()
    config={'purpose':'development_replay_of_previously_seen_data','source_commit':COMMIT,'adapter_sha256':sha(__file__),'calibration_seeds':list(range(100,124)),'test_seeds':list(range(500,530)),'query_package_replicates':4,'budgets_in_unique_candidate_packages':[1,2,4],'prior_precision':1.,'noise_variance_rule':'mean calibration per-replicate variance / 4','feature_rule':'intercept + seven calibration-standardized MPE footprint features; proxy added separately','fantasies':64,'posterior_samples':256,'delta':.05,'eta':0.,'include_incumbent':False,'no_sequential_posterior_update_claim':True,'no_new_training':True,'audit_opened':False,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(out/'protocol.json',config)
    hashes={}
    def load(path):
        hashes[str(path.relative_to(a.base))]=sha(path)
        return read(path)
    calibration=[]
    for seed in config['calibration_seeds']:
        for j in range(4):
            row=load(a.base/f'pivot_calibration_128/seed_{seed}/candidate_{j}.json')
            assert row['audit_labels_used'] is False and row['query_episodes']==128
            assert len(row['samples'])==4
            calibration.append(row)
    raw=np.array([r['features'][1:] for r in calibration]);mean=raw.mean(0);scale=raw.std(0);scale[scale<1e-8]=1
    x=np.c_[np.ones(len(raw)),(raw-mean)/scale];y=np.array([r['target_correction'] for r in calibration])
    noise=max(float(np.mean([r['query_noise_variance'] for r in calibration]))/4,1e-6)
    posterior=BayesianLinearDeltaPosterior(prior_precision=1.,noise_variance=noise).fit(x,y)
    write(out/'posterior.json',{'feature_mean':mean.tolist(),'feature_scale':scale.tolist(),'coefficient_mean':posterior.mean.tolist(),'covariance':posterior.covariance.tolist(),'noise_variance':noise,'n_observations':posterior.n_observations})
    # Only check that all required files exist here, before opening any audit.
    for seed in config['test_seeds']:
        base=a.base/f'pivot_panel30/seed_{seed}'
        assert (base/'features.json').is_file()
        for j in range(4):
            for r in range(4):assert (base/f'selection/candidate_{j}/replicate_{r}/paired.json').is_file()
            for r in range(8):assert (base/f'audit/candidate_{j}/replicate_{r}/paired.json').is_file()
    records=[];panels={};selection_eval={};selection_adaptation={}
    for seed in config['test_seeds']:
        selection_eval[seed]=set();selection_adaptation[seed]=set()
        base=a.base/f'pivot_panel30/seed_{seed}';features=load(base/'features.json');manifest=load(base/'manifest.json');cfg=manifest['config']
        assert cfg['candidates']==4 and cfg['query_episodes']==128 and cfg['audit_replicates']==8
        arr=np.array(features['rows']);z=np.c_[np.ones(4),(arr[:,1:]-mean)/scale]
        block=cfg['n_envs']*cfg['n_steps']
        singlecost=2*math.ceil(cfg['adapt_steps']/block)*block+2*cfg['query_episodes']*cfg['horizon'];packagecost=4*singlecost
        commoncost=4*math.ceil(cfg['candidate_steps']/block)*block+4*2*cfg['proxy_episodes']*cfg['horizon']
        candidates=[{'transition_id':str(j),'delta_proxy':float(arr[j,0]),'features':z[j].tolist(),'hf_query_cost':packagecost} for j in range(4)]
        queried=[]
        def hf(candidate):
            j=int(candidate['transition_id']);items=[]
            for r in range(4):
                row=load(base/f'selection/candidate_{j}/replicate_{r}/paired.json')
                assert row['split']=='selection' and row['total_env_steps']==singlecost
                assert len(row['old'])==128 and len(row['new'])==128
                assert [v['seed'] for v in row['old']]==[v['seed'] for v in row['new']]
                assert np.isclose(row['delta'],np.mean([n['prey_return']-o['prey_return'] for o,n in zip(row['old'],row['new'])]))
                selection_eval[seed].update(v['seed'] for v in row['old']);selection_adaptation[seed].add(row['adaptation_seed']);items.append(row)
            queried.append(j)
            return {'delta_true':float(np.mean([r['delta'] for r in items])),'hf_query_cost':sum(r['total_env_steps'] for r in items)}
        def record(name,budget,result):
            assert len(queried)==result.hf_budget and len(set(queried))==len(queried)
            assert set(map(str,queried))==set(result.queried_ids)
            assert result.hf_cost==len(queried)*packagecost
            records.append({'seed':seed,'method':name,'budget_packages':budget,'selected':int(result.selected_candidate_id),'queried_ids':list(result.queried_ids),'query_packages_used':result.hf_budget,'query_env_steps':result.hf_cost,'decision_env_steps':commoncost+int(features['env_steps'])+result.hf_cost,'selected_estimate':result.selected_delta_estimate,'stop_reason':result.stop_reason})
            queried.clear()
        records.append({'seed':seed,'method':'proxy_only','budget_packages':0,'selected':int(np.argmax(arr[:,0])),'queried_ids':[],'query_packages_used':0,'query_env_steps':0,'decision_env_steps':commoncost,'selected_estimate':float(arr[:,0].max()),'stop_reason':'proxy_only'})
        record('correction_only',0,run_pivot_round(None,candidates,None,hf,lambda rows,model,budget:[],0,model=posterior))
        permutation=np.random.default_rng(seed+20260915).permutation(4).tolist()
        for budget in [1,2,4]:
            record('random_batch',budget,run_pivot_round(None,candidates,None,hf,lambda rows,model,budget:[str(j) for j in permutation[:budget]],budget,model=posterior))
            record('top_proxy_batch',budget,run_pivot_round(None,candidates,None,hf,lambda rows,model,budget:[str(j) for j in np.argsort(-arr[:,0],kind='stable')[:budget]],budget,model=posterior))
            record('author_pivot_voi_no_stop',budget,run_pivot_round(None,candidates,None,hf,select_pivot_voi,budget,model=posterior,acquisition_kwargs={'seed':seed,'fantasies':64,'posterior_samples':256}))
            record('author_pivot_voi',budget,run_pivot_voi_round(None,candidates,hf,posterior,budget,seed=seed,delta=.05,eta=0.,fantasies=64,posterior_samples=256))
        panels[seed]={'features':arr.tolist(),'predicted_gain':(posterior.predict(z)+arr[:,0]).tolist()}
    write(out/'decisions_frozen.json',records);frozen_hash=sha(out/'decisions_frozen.json')
    write(out/'selection_input_hashes.json',hashes)
    write(out/'selection_seal.json',{'decisions_sha256':frozen_hash,'audit_opened':False,'n_decisions':len(records)})
    # First audit-label access starts here, after every decision is frozen.
    audit_hashes={};audit_by_seed={}
    for seed in config['test_seeds']:
        base=a.base/f'pivot_panel30/seed_{seed}';gains=[];frozen_gains=[]
        for j in range(4):
            vals=[];ctrl=[]
            for r in range(8):
                path=base/f'audit/candidate_{j}/replicate_{r}/paired.json';row=read(path);audit_hashes[str(path.relative_to(a.base))]=sha(path)
                assert row['split']=='audit'
                assert not selection_eval[seed].intersection(v['seed'] for v in row['old'])
                assert row['adaptation_seed'] not in selection_adaptation[seed]
                vals.append(row['delta'])
                path=path.with_name('frozen_control.json');control=read(path);audit_hashes[str(path.relative_to(a.base))]=sha(path);ctrl.append(control['delta'])
            gains.append(vals);frozen_gains.append(ctrl)
        g=np.array(gains);means=g.mean(1);left=g[:,:4].mean(1);right=g[:,4:].mean(1)
        audit_by_seed[seed]={'audit_gain':means.tolist(),'independent_frozen_gain':np.array(frozen_gains).mean(1).tolist(),'audit_replicates':g.tolist()}
        for record in records:
            if record['seed']!=seed:continue
            j=record['selected'];record.update(audit_gain=float(means[j]),harmful_sample_mean=bool(means[j]<0),crossfit_reference_gap=float(.5*(right[np.argmax(left)]-right[j]+left[np.argmax(right)]-left[j])))
    for seed in config['test_seeds']:
        full=[r for r in records if r['seed']==seed and r['budget_packages']==4 and r['method'] in ['random_batch','top_proxy_batch','author_pivot_voi_no_stop']]
        assert len({r['selected'] for r in full})==1, 'All-query methods must agree'
    assert sha(out/'decisions_frozen.json')==frozen_hash
    write(out/'audit_input_hashes.json',audit_hashes);write(out/'scored_decisions.json',records);write(out/'candidate_audit.json',audit_by_seed)
    table=[]
    for key in sorted({(r['method'],r['budget_packages']) for r in records}):
        group=[r for r in records if (r['method'],r['budget_packages'])==key]
        table.append({'method':key[0],'budget_packages':key[1],'audit_gain':ci([r['audit_gain'] for r in group]),'mean_queries':float(np.mean([r['query_packages_used'] for r in group])),'mean_query_env_steps':float(np.mean([r['query_env_steps'] for r in group])),'mean_decision_env_steps':float(np.mean([r['decision_env_steps'] for r in group])),'negative_audit_mean_fraction':float(np.mean([r['harmful_sample_mean'] for r in group]))})
    differences=[]
    for budget in [1,2,4]:
        for ref in ['random_batch','author_pivot_voi_no_stop']:
            get=lambda name:{r['seed']:r for r in records if r['method']==name and r['budget_packages']==budget}
            first=get('author_pivot_voi');other=get(ref)
            differences.append({'contrast':'author_pivot_voi - '+ref,'budget_packages':budget,'audit_gain_difference':ci([first[s]['audit_gain']-other[s]['audit_gain'] for s in config['test_seeds']]),'selection_changed_seeds':sum(first[s]['selected']!=other[s]['selected'] for s in config['test_seeds'])})
    truth=np.array([audit_by_seed[s]['audit_gain'] for s in config['test_seeds']]);proxy=np.array([panels[s]['features'] for s in config['test_seeds']])[:,:,0];pred=np.array([panels[s]['predicted_gain'] for s in config['test_seeds']])
    summary={'status':'complete','purpose':config['purpose'],'source_commit':COMMIT,'n_calibration_seeds':24,'n_panel_seeds':30,'n_candidates':120,'n_decisions':len(records),'decisions_sha256':frozen_hash,'audit_isolated':True,'isolation_check_scope':'within each independent trained candidate panel','source_worktree_unchanged':not subprocess.check_output(['git','-C',str(a.repo),'status','--porcelain','--untracked-files=no'],text=True).strip(),'new_environment_steps':0,'new_api_calls':0,'query_package_definition':'4 historical paired response replicates, 128 evaluation episodes each; paid independently for each method','results':table,'exploratory_paired_differences':differences,'predictor_mse_vs_noisy_audit':{'proxy':float(np.mean((proxy-truth)**2)),'shared_bayesian_correction':float(np.mean((pred-truth)**2))},'limitations':['Previously inspected historical test panel; development evidence only.','MPE-specific features and calibrated noise differ from original controlled worlds.','Upstream generic entry uses one pre-query stop decision and a batch; no sequential posterior updating.','Shared linear posterior models coefficient uncertainty; misspecification can cause overconfidence.','Negative audit means are noisy estimates, not confirmed harmful upgrades.','Cost excludes shared pretraining and calibration; query costs retain full standalone cost.','Bootstrap intervals are exploratory and not adjusted for the multiple comparisons here.'],'seconds':round(time.monotonic()-started,3)}
    write(out/'summary.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
