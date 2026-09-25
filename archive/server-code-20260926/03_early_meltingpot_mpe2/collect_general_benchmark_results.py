"""Verify complete batch outputs and export small evidence, never policy weights."""
import argparse,csv,io,json,tarfile
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();b=a.base
    mpe=b/'fresh_mpe_general';melting=b/'melting_general_long'
    assert read(mpe/'status.json')['state']=='complete'
    assert read(melting/'status.json')['state']=='complete'
    rows=read(mpe/'scored_decisions.json');failures=[]
    for r in rows:
        assert r['query_count']==len(r['queries'])<=r['budget']
        assert r['query_cost']==sum(q['cost'] for q in r['queries'])
        assert len({(q['candidate'],q['replicate']) for q in r['queries']})==len(r['queries'])
    for task,seed in {(r['task'],r['seed']) for r in rows}:
        full=[r for r in rows if r['task']==task and r['seed']==seed and r['budget']==4 and r['method'] in ['author_batch_no_stop','random_batch','uniform_sample_mean']]
        assert len(full)==3 and len({r['selected'] for r in full})==1,'All-candidate sample-mean baselines must agree'
        assert all(r['query_count']==4 for r in full)
    melting_rows=read(melting/'scored_decisions.json');paired={}
    for r in melting_rows:
        assert r['query_packages_used']==len(r['queried_ids'])<=r['budget_packages']
        assert len(set(r['queried_ids']))==len(r['queried_ids'])
        assert r['query_cost']==r['query_packages_used']*195072
        key=(r['task'],r['seed'],r['method'],r['budget_packages'])
        paired.setdefault(key,[]).append(r)
    for group in paired.values():
        assert len(group)==2 and {r['response_steps'] for r in group}=={32768,131072}
        assert len({(r['selected'],r['query_cost'],tuple(r['queried_ids'])) for r in group})==1
    assert len(paired)==168 and len(melting_rows)==336
    report={'status':'verified','mpe_decisions':len(rows),'mpe_all_query_agreement':True,'query_cost_checks':True,
            'melting_horizon_scores':len(melting_rows),'melting_frozen_choices':len(paired),
            'melting_same_choices_and_cost_across_horizons':True,'melting_query_cost_checks':True,
            'scope':'Both selected-task benchmark extensions; not full official suites'}
    (b/'general_benchmark_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    fields=['task','seed','method','budget','selected','query_count','query_cost','audit_gain','frozen_gain','decision_cost']
    with (mpe/'method_seed_results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    rows=read(melting/'scored_decisions.json');fields=['task','seed','response_steps','method','budget_packages','selected','query_packages_used','query_cost','audit_gain','frozen_gain','audit_response_replicates']
    with (melting/'method_seed_results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    target=b/'general_benchmark_small.tar.gz'
    paths=[b/'general_benchmark_verification.json']
    paths.extend(b/name for name in ['general_paper_metrics.json','general_paper_metric_panels.json'] if (b/name).exists())
    for root in [mpe,melting]:
        paths.extend(root/name for name in ['protocol.json','status.json','summary.json','scored_decisions.json','mechanism_by_seed.json','method_seed_results.csv','completed_jobs.jsonl'] if (root/name).exists())
    paths.extend(mpe/name for name in ['source_hashes.json','environment.json','global_selection_seal.json'])
    for task in ['push','adversary']:paths.extend([mpe/task/'candidate_audit.json',mpe/task/'calibration.json'])
    paths.extend([melting/'candidate_audit.json',melting/'audit_hashes.json'])
    # Selected per-panel immutable seals are tiny; traces and raw rollout lists stay remote.
    paths.extend(mpe.glob('*/seed_*/selection_seal.json'))
    with tarfile.open(target,'w:gz') as tf:
        for path in paths:tf.add(path,arcname=path.relative_to(b),recursive=False)
    print(json.dumps({'archive':str(target),'bytes':target.stat().st_size,'files':len(paths),'verification':report}))
if __name__=='__main__':main()
