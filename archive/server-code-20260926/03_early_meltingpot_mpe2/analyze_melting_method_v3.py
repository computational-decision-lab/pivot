"""Audit all native panels, then test the predeclared mechanism-method interaction."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from audit_melting_short_long import ci
from verify_melting_method_v3 import inspect_panel
from verify_melting_mechanism import sha,csv_rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.batch/'protocol.json').read_text());status=json.loads((a.batch/'status.json').read_text())
    assert status['status']=='complete' and status['frozen_inputs_unchanged']
    assert protocol['test_seeds']==list(range(44000,44030))
    panels=[];contexts=set();verified={};rows=[]
    for seed in protocol['test_seeds']:
        x=inspect_panel(a.batch/f'seed_{seed}',sha(a.batch/'protocol.json'))
        assert not contexts&x['contexts'],'Root seed context collision';contexts.update(x['contexts']);panels.append(x)
        for name,h in x['verified_input_hashes'].items():verified[f'seed_{seed}/{name}']=h
        for d in x['scored']:
            rows.append({k:d.get(k) for k in ['method','adaptation','budget_cap','hf_queries','hf_episode_cost','selected_id','audit_gain','noisy_audit_isr']}|{'seed':seed})
    cap=protocol['primary_hf_episode_cap'];primary=protocol['primary_method'];baseline=protocol['primary_comparator']
    def get(seed,h,method,budget):
        found=[r for r in rows if r['seed']==seed and r['adaptation']==h and r['method']==method and (r['budget_cap']==budget or method in ['proxy_only','calibrated_no_hf','all_hf_reference'])]
        assert len(found)==1;return found[0]
    contrast=[]
    for seed in protocol['test_seeds']:
        short=get(seed,4,primary,cap)['audit_gain']-get(seed,4,baseline,cap)['audit_gain']
        long=get(seed,32,primary,cap)['audit_gain']-get(seed,32,baseline,cap)['audit_gain']
        contrast.append({'seed':seed,'short_regret_reduction':short,'long_regret_reduction':long,'interaction':long-short})
    effects={k:ci([r[k] for r in contrast]) for k in ['short_regret_reduction','long_regret_reduction','interaction']}
    all_conditions=sorted({(r['adaptation'],r['method'],r['budget_cap']) for r in rows},key=str);frontier=[]
    for h,method,b in all_conditions:
        subset=[r for r in rows if (r['adaptation'],r['method'],r['budget_cap'])==(h,method,b)];assert len(subset)==30
        stat=ci([r['noisy_audit_isr'] for r in subset]);gain=ci([r['audit_gain'] for r in subset])
        frontier.append({'adaptation':h,'method':method,'budget_cap':b,'mean_hf_episode_cost':float(np.mean([r['hf_episode_cost'] for r in subset])),
                         'mean_audit_isr':stat['mean'],'isr_ci_low':stat['ci95'][0],'isr_ci_high':stat['ci95'][1],
                         'mean_audit_gain':gain['mean'],'gain_ci_low':gain['ci95'][0],'gain_ci_high':gain['ci95'][1]})
    diagnostics=[]
    for h in [4,32]:
        for left,right,label in [('calibrated_no_hf','proxy_only','calibration_value'),(primary,'calibrated_no_hf','beyond_calibration'),
            ('pivot_sequential','uniform_random_matched','fixed_budget_acquisition'),(primary,'pivot_sequential','stopping_value'),('author_pivot_voi','author_random_hf','author_batch_mixed_estimator')]:
            effect=ci([get(s,h,left,cap)['audit_gain']-get(s,h,right,cap)['audit_gain'] for s in protocol['test_seeds']])
            diagnostics.append({'adaptation':h,'contrast':label,'left':left,'right':right,**effect})
    mechanism_rows=[]
    for panel in panels:
        r={'seed':panel['seed']}
        for h in [4,32]:
            data=[x for x in panel['mechanism'] if x['adaptation']==h]
            table={(x['candidate'],x['block']):x for x in data}
            r[f'own_gain_{h}']=float(np.mean([x['responder_own_gain'] for x in data]))
            r[f'gap_squared_{h}']=float(np.mean([table[str(j),'A']['gap']*table[str(j),'B']['gap'] for j in range(8)]))
            r[f'candidate_response_abs_{h}']=float(np.mean([abs((table[str(j),'A']['candidate_specific_response_effect']+table[str(j),'B']['candidate_specific_response_effect'])/2) for j in range(8)]))
        r['own_gain_long_minus_short']=r['own_gain_32']-r['own_gain_4'];r['gap_squared_long_minus_short']=r['gap_squared_32']-r['gap_squared_4'];mechanism_rows.append(r)
    confirmation_mechanism={k:ci([r[k] for r in mechanism_rows]) for k in mechanism_rows[0] if k!='seed'}
    method_supported=effects['long_regret_reduction']['ci95'][0]>0 and effects['interaction']['ci95'][0]>0
    supported=method_supported and confirmation_mechanism['own_gain_long_minus_short']['ci95'][0]>0 and confirmation_mechanism['gap_squared_long_minus_short']['ci95'][0]>1
    coverage=[r for p in panels for r in p['coverage']]
    uncertainty_diagnostics={}
    for h in [4,32]:
        uncertainty_diagnostics[str(h)]={k:ci([float(np.mean([r[k] for r in p['coverage'] if r['adaptation']==h])) for p in panels]) for k in ['absolute_model_error','absolute_proxy_error','latent_95_contains_noisy_audit','observation_predictive_95_contains_noisy_audit']}
        uncertainty_diagnostics[str(h)]['pivot_zero_query_fraction']=ci([float(get(seed,h,primary,cap)['hf_queries']==0) for seed in protocol['test_seeds']])
    result={'status':'CHAIN_SUPPORTED_IN_THIS_SCOPE' if supported else 'METHOD_CHAIN_NOT_SUPPORTED','seed_count':30,'primary_method':primary,'primary_comparator':baseline,'primary_hf_episode_cap':cap,
        'primary_effects':effects,'method_contrasts_supported':method_supported,'confirmation_mechanism':confirmation_mechanism,'registered_diagnostics':diagnostics,'accounting':{'native_episodes':sum(p['episodes'] for p in panels),'native_frames':sum(p['frames'] for p in panels),'calibration_native_episodes':2688},
        'protocol_sha256':sha(a.batch/'protocol.json'),'verified_inputs':len(verified),'registered_uncertainty_diagnostics':uncertainty_diagnostics,
        'scope':protocol['audit_target'],'limitations':protocol['claims_excluded'],
        'interpretation':'Conditional fresh-seed native adaptive-extension result; primary paired contrasts cancel the noisy audit maximum. No additional seeds or altered settings to chase a win.',
        'next_action':'Report bounded mechanism and method evidence' if supported else 'Preserve null/negative and inspect registered method/calibration/stopping diagnostics before any new method version.'}
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'verified_input_hashes.json').write_text(json.dumps(verified,indent=2)+'\n')
    csv_rows(a.output/'method_seed_results.csv',rows);csv_rows(a.output/'frontier.csv',frontier);csv_rows(a.output/'primary_seed_contrasts.csv',contrast);csv_rows(a.output/'confirmation_mechanism_by_seed.csv',mechanism_rows)
    csv_rows(a.output/'uncertainty_audit.csv',coverage)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
