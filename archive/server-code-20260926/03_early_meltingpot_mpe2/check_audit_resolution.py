"""Noisy audit-rank stability; descriptive seed-cluster diagnostic only."""
import argparse,itertools,json,pathlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--audit',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
data=json.loads(a.audit.read_text());partitions=[(0,)+c for c in itertools.combinations(range(1,8),3)];rows=[]
for seed,info in sorted(data.items()):
    vals=np.array(info['audit_replicates']);assert vals.shape==(4,8)
    agree=[];crossgaps=[]
    for part in partitions:
        other=sorted(set(range(8))-set(part));left=vals[:,part].mean(1);right=vals[:,other].mean(1);la=np.argsort(-left,kind='stable');ra=np.argsort(-right,kind='stable')
        agree.append(la[0]==ra[0]);crossgaps.append(.5*((right[la[0]]-right[la[1]])+(left[ra[0]]-left[ra[1]])))
    full=vals.mean(1);order=np.argsort(-full,kind='stable');se=float(np.sqrt(vals.var(1,ddof=1)[order[:2]].sum()/8));gap=float(full[order[0]]-full[order[1]])
    rows.append({'seed':int(seed),'half_split_winner_agreement':float(np.mean(agree)),'crossfit_top_vs_runner_gap':float(np.mean(crossgaps)),'full_sample_top_two_gap':gap,'estimated_top_two_difference_se':se,'gap_over_se':gap/se if se>0 else None,'top_gap_within_one_se':bool(gap<=se)})
x=np.array([r['half_split_winner_agreement'] for r in rows]);rng=np.random.default_rng(20260915);boot=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
result={'purpose':'post-hoc audit resolution diagnostic','n_training_seeds':len(rows),'partitions_per_seed':35,'mean_half_split_winner_agreement':float(x.mean()),'seed_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),'n_full_sample_top_gap_within_one_se':sum(r['top_gap_within_one_se'] for r in rows),'mean_crossfit_top_vs_runner_gap':float(np.mean([r['crossfit_top_vs_runner_gap'] for r in rows])),'rows':rows,'limitations':['The 35 partitions reuse data; only trained seeds are bootstrap units.','Half-split disagreement does not prove true best candidates differ.','Top-two gap is selected using the same data; it is not a confidence interval or unbiased separation estimate.','Noisy per-panel ranks do not imply that aggregate paired method effects cannot be estimated.','No thresholds are used here to drop tasks or cherry-pick seeds.']}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
