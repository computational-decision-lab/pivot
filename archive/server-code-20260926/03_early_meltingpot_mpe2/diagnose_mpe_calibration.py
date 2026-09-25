"""Post-hoc diagnostic using calibration-only discrepancy estimation.
Does not change any existing decisions or rerun selector hyperparameters.
"""
import argparse,json,pathlib,hashlib
import numpy as np
from pivot.acquisition.pivot_voi import BayesianLinearDeltaPosterior

def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,x):pathlib.Path(p).write_text(json.dumps(x,indent=2)+'\n')
def fit(rows):
    x=np.array([r['features'][1:] for r in rows]);y=np.array([r['target_correction'] for r in rows]);mu=x.mean(0);sd=x.std(0);sd[sd<1e-8]=1
    noise=max(float(np.mean([r['query_noise_variance'] for r in rows]))/4,1e-6)
    model=BayesianLinearDeltaPosterior(prior_precision=1.,noise_variance=noise).fit(np.c_[np.ones(len(x)),(x-mu)/sd],y)
    return model,mu,sd

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=pathlib.Path,required=True);p.add_argument('--replay',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=[read(a.base/f'pivot_calibration_128/seed_{s}/candidate_{j}.json') for s in range(100,124) for j in range(4)]
    assert all(r['audit_labels_used'] is False for r in rows)
    calibration=[]
    for seed in range(100,124):
        train=[r for r in rows if r['seed']!=seed];test=[r for r in rows if r['seed']==seed];m,mu,sd=fit(train)
        for row in test:
            x=np.r_[1.,(np.array(row['features'][1:])-mu)/sd][None,:];error=float(row['target_correction']-m.predict(x)[0]);latent=float(m.predictive_variance(x)[0]);noise=row['query_noise_variance']/4
            calibration.append({'seed':seed,'candidate':row['candidate'],'error':error,'latent_variance':latent,'measurement_variance':noise})
    # Freeze discrepancy estimate before accessing this diagnostic's audit data.
    discrepancy=max(0.,float(np.mean([r['error']**2-r['latent_variance']-r['measurement_variance'] for r in calibration])))
    write(a.output/'calibration_only_estimate.json',{'leave_one_training_seed_out':True,'n_seeds':24,'mean_squared_error':float(np.mean([r['error']**2 for r in calibration])),'discrepancy_variance_estimate':discrepancy,'formula':'max(0, mean(residual^2 - coefficient_variance - observed_calibration_mean_noise_variance))','not_tuned_on_test':True})
    write(a.output/'calibration_residuals.json',calibration)
    m,mu,sd=fit(rows);audit=read(a.replay/'candidate_audit.json');decisions=read(a.replay/'decisions_frozen.json');records=[];selectionchecks=[]
    for seed in range(500,530):
        feats=np.array(read(a.base/f'pivot_panel30/seed_{seed}/features.json')['rows']);x=np.c_[np.ones(4),(feats[:,1:]-mu)/sd];pred=m.predict(x)+feats[:,0];latent=m.predictive_variance(x);rep=np.array(audit[str(seed)]['audit_replicates']);truth=rep.mean(1);measurement=rep.var(1,ddof=1)/rep.shape[1]
        for j in range(4):
            error=pred[j]-truth[j]
            records.append({'seed':seed,'candidate':j,'prediction_error':float(error),'coefficient_variance':float(latent[j]),'audit_mean_measurement_variance':float(measurement[j]),'covered_coefficient_only':bool(abs(error)<=1.95996398454*np.sqrt(latent[j])),'covered_with_measurement_noise':bool(abs(error)<=1.95996398454*np.sqrt(latent[j]+measurement[j])),'covered_with_calibration_discrepancy':bool(abs(error)<=1.95996398454*np.sqrt(latent[j]+measurement[j]+discrepancy))})
        draws=m.sample_predictions(x,32768,np.random.default_rng(seed+20260915))+feats[:,0];best=int(np.argmax(pred));prob=float(np.mean(np.argmax(draws,axis=1)==best))
        source=next(r for r in decisions if r['seed']==seed and r['method']=='author_pivot_voi' and r['budget_packages']==4)
        selectionchecks.append({'seed':seed,'selection_probability_32768_draws':prob,'predicted_best':best,'audit_sample_best':int(np.argmax(truth)),'audit_sample_best_matches':bool(best==int(np.argmax(truth))),'query_packages_used':source['query_packages_used'],'stopped_prequery':source['query_packages_used']==0})
    matching=[]
    scored=read(a.replay/'scored_decisions.json')
    for budget in [1,2,4]:
        differences=[]
        for seed in range(500,530):
            get=lambda name:next(r for r in scored if r['seed']==seed and r['method']==name and r['budget_packages']==(0 if name=='correction_only' else budget))
            pv=get('author_pivot_voi');rnd=get('random_batch');zero=get('correction_only')
            # Same pre-query stop rule in both arms: actual queries match per seed.
            matched=rnd if pv['query_packages_used'] else zero
            assert matched['query_env_steps']==pv['query_env_steps']
            differences.append(pv['audit_gain']-matched['audit_gain'])
        xs=np.array(differences);rng=np.random.default_rng(20260915);boots=xs[rng.integers(30,size=(10000,30))].mean(1)
        matching.append({'budget_packages':budget,'contrast':'author VOI allocation minus random allocation with same author stop rule','mean_gain_difference':float(xs.mean()),'ci95':np.quantile(boots,[.025,.975]).tolist(),'actual_query_cost_matches_per_seed':True,'classification':'post-hoc matched-cost development diagnostic'})
    summary={'purpose':'post-hoc calibration and matched-cost diagnostic; no new confirmatory claim','n_seeds':30,'n_candidates':120,'discrepancy_variance_from_calibration':discrepancy,'mean_width_parameter_sd':float(np.mean(np.sqrt([r['coefficient_variance'] for r in records]))),'coverage_vs_noisy_audit':{key:float(np.mean([r[key] for r in records])) for key in ['covered_coefficient_only','covered_with_measurement_noise','covered_with_calibration_discrepancy']},'stopped_panels':sum(r['stopped_prequery'] for r in selectionchecks),'stopped_panels_matching_noisy_audit_best':sum(r['stopped_prequery'] and r['audit_sample_best_matches'] for r in selectionchecks),'matched_cost_contrasts':matching,'limitations':['Audit argmax is noisy; mismatch is not proof of a wrong true winner.','Coverage refers to noisy audit means, not known exact deployment values.','Calibration discrepancy estimator is a diagnostic approximation, not a calibrated 95% guarantee.','No selector was retuned or rerun after looking at these test outcomes.','These old test outcomes have already been inspected; this is development evidence.']}
    write(a.output/'coverage_rows.json',records);write(a.output/'selection_confidence.json',selectionchecks);write(a.output/'summary.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
