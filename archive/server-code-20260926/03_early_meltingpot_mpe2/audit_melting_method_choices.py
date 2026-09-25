"""Post-result diagnostics of frozen query traces, without new simulation or tuning."""
import argparse,collections,csv,io,json,tarfile
from pathlib import Path
import numpy as np

def ci(x):
    x=np.array(x,dtype=float);b=np.random.default_rng(20260917).integers(len(x),size=(10000,len(x)))
    return {'mean':float(x.mean()),'ci95':np.quantile(x[b].mean(axis=1),[.025,.975]).tolist(),'n_roots':len(x)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);a=p.parse_args()
    with (a.analysis/'method_seed_results.csv').open(encoding='utf-8-sig') as f:scored=list(csv.DictReader(f))
    methods=['pivot_sequential','pivot_sequential_adaptive_stop','uniform_random_matched','global_ivr_matched'];records=[];residuals=[]
    with (a.analysis/'uncertainty_audit.csv').open(encoding='utf-8-sig') as f:predictions={(int(r['seed']),int(r['adaptation']),r['candidate']):float(r['predicted_delta']) for r in csv.DictReader(f)}
    with tarfile.open(a.archive,'r:gz') as t:
        members={m.name:m for m in t.getmembers()};root='melting_method_confirm_20260917'
        for seed in range(44000,44030):
            m=members[f'{root}/seed_{seed}/decisions_frozen.json'];assert m.isfile();data=json.load(t.extractfile(m))
            for d in data:
                if d['method'] not in methods or d['budget_cap']!=192:continue
                queries=d['queried_ids'];seen=set();zero=0;total=0;all_zero=0;exact_zero_ties=0
                for step in d.get('steps',[]):
                    if 'query_id' not in step:continue
                    if 'scores' in step:
                        eligible=[r for r in step['scores'] if r['transition_id']!='incumbent' and r['transition_id'] not in seen]
                        chosen=next(r for r in eligible if r['transition_id']==step['query_id'])
                        zero+=chosen['evsi']<=1e-12;all_zero+=all(r['evsi']<=1e-12 for r in eligible);total+=1
                        if all(r['evsi']==0 for r in eligible):
                            assert step['query_id']==min(r['transition_id'] for r in eligible)
                            exact_zero_ties+=1
                    seen.add(step['query_id'])
                both=any(int(j)<4 for j in queries) and any(int(j)>=4 for j in queries)
                records.append({'seed':seed,'adaptation':d['adaptation'],'method':d['method'],'queries':queries,'both_edit_directions_queried':bool(both),
                                'both_endpoints_queried':all(j in queries for j in ['0','7']),'hf_queries':len(queries),
                                'scored_query_steps':total,'chosen_zero_evsi_steps':int(zero),'all_available_zero_evsi_steps':int(all_zero),
                                'exact_zero_lexical_tie_steps':exact_zero_ties,'stop_reason':d.get('stop_reason')})
            post=json.load(t.extractfile(members[f'{root}/seed_{seed}/posterior.json']))
            summary=json.load(t.extractfile(members[f'{root}/seed_{seed}/summary.json']))
            for h in [4,32]:
                q={(r['candidate'],r['block']):r['deployment_delta_audit'] for r in summary['mechanism_audit'] if r['adaptation']==h}
                cross=[];noise=[];variances=[];C=np.array(post[str(h)]['covariance'])
                for j,prob in enumerate([.125,.225,.325,.425,.575,.675,.775,.875]):
                    first=q[str(j),'A'];second=q[str(j),'B'];pred=predictions[seed,h,str(j)]
                    cross.append((first-pred)*(second-pred));noise.append((first-second)**2/4)
                    feature=np.array([(prob-.5)/.25,abs(prob-.5)/.25]);variances.append(float(feature@C@feature))
                residuals.append({'seed':seed,'adaptation':h,'cross_block_prediction_squared_error':float(np.mean(cross)),
                    'audit_mean_noise_variance_estimate':float(np.mean(noise)),'average_latent_posterior_variance':float(np.mean(variances))})
    summaries=[]
    for h in [4,32]:
        for method in methods:
            rows=[r for r in records if r['adaptation']==h and r['method']==method];assert len(rows)==30
            summaries.append({'adaptation':h,'method':method,'query_count_histogram':dict(collections.Counter(r['hf_queries'] for r in rows)),
                'both_edit_directions_queried_seeds':sum(r['both_edit_directions_queried'] for r in rows),
                'both_endpoints_queried_seeds':sum(r['both_endpoints_queried'] for r in rows),
                'scored_query_steps':sum(r['scored_query_steps'] for r in rows),
                'chosen_zero_evsi_steps':sum(r['chosen_zero_evsi_steps'] for r in rows),
                'all_available_zero_evsi_steps':sum(r['all_available_zero_evsi_steps'] for r in rows),
                'exact_zero_lexical_tie_steps':sum(r['exact_zero_lexical_tie_steps'] for r in rows)})
    contrasts=[]
    def gain(seed,h,m):
        r=[r for r in scored if int(r['seed'])==seed and int(r['adaptation'])==h and r['method']==m and r['budget_cap']=='192'];assert len(r)==1
        return float(r[0]['audit_gain'])
    for h in [4,32]:
        for right in ['pivot_sequential','pivot_sequential_adaptive_stop','uniform_random_matched']:
            contrasts.append({'adaptation':h,'left':'global_ivr_matched','right':right,**ci([gain(seed,h,'global_ivr_matched')-gain(seed,h,right) for seed in range(44000,44030)])})
    out={'scope':'Post-result descriptive method audit, not a new confirmation test. All methods and data were frozen before outcomes; these extra contrasts/trace summaries were chosen after the primary null. No new simulation, seeds, candidates or algorithm changes.',
         'summaries':summaries,'exploratory_paired_contrasts':contrasts,'per_seed':records,
         'noise_corrected_prediction_diagnostics':{str(h):{k:ci([r[k] for r in residuals if r['adaptation']==h]) for k in ['cross_block_prediction_squared_error','audit_mean_noise_variance_estimate','average_latent_posterior_variance']} for h in [4,32]},
         'prediction_diagnostic_scope':'Post-result cross-block squared prediction error removes independent audit-mean noise in expectation; prediction is held fixed across A/B, and still includes proxy-estimation uncertainty and response-world/model error. It does not isolate one error source.',
         'interpretation_limits':['Trace association is not proof of a single causal failure mechanism.','Finite-Monte-Carlo zero EVSI need not be mathematically zero EVSI.','Global-IVR is the prespecified matched integrated-variance rule, distinct from author Global-VOI.']}
    (a.analysis/'post_result_query_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['summaries','exploratory_paired_contrasts','noise_corrected_prediction_diagnostics']},indent=2))

if __name__=='__main__':main()
