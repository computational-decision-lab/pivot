"""Apply the independently registered second-cohort meaningful-effect criterion.
The predecessor analysis file remains unchanged; this explicit additional report
separates its CI-only statistical flag from this cohort's combined success rule.
"""
import argparse,json,pathlib,hashlib,datetime
p=argparse.ArgumentParser();p.add_argument('--base',required=True,type=pathlib.Path);a=p.parse_args();b=a.base
protocol=json.loads((b/'formal_protocols/confirmation_protocol.json').read_text());summary_path=b/'confirmation/analysis/summary.json';s=json.loads(summary_path.read_text())
assert protocol['cohort_id']=='second_independent_precision_cohort_20260923'
assert s['confirmation_roots']==list(range(501,531)) and s['n_independent_roots']==30
assert s['primary_adaptation']==12 and s['primary_query_budget']==2
v=s['hypotheses']['primary_PIVOT_KG_minus_Uniform_expected_100_selected_gain'];threshold=float(protocol['minimum_meaningful_selected_gain'])
assert threshold==2.0
report={'cohort_id':protocol['cohort_id'],'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'registered_rule':'primary gain mean>=2 AND paired root-bootstrap95%lower bound>0','primary_gain_contrast':v,'minimum_meaningful_gain':threshold,'statistical_CI_only_success_from_unchanged_analyzer':s['prespecified_success'],'combined_prespecified_meaningful_success':bool(v['mean']>=threshold and v['lo']>0),'all_other_comparisons_secondary':True,'confirmation_summary_file_sha256':hashlib.sha256(summary_path.read_bytes()).hexdigest(),'next_action':'Deliver all results. Do not add roots, retune, repeat, or launch a third cohort.'}
out=b/'confirmation/analysis/precision_decision.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
