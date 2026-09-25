import json,csv,itertools,sys
from pathlib import Path
import numpy as np
B=Path('/Users/yuanzhiyi/Desktop/MetaDrive_PIVOT_20260923'); O=Path('/private/tmp/metadrive_diagnostics_20260923')
roots=[json.loads(p.read_text()) for p in sorted((B/'confirmation/roots').glob('*/root.json'))]
frozen=json.loads((B/'calibration/analysis/frozen.json').read_text()); loro=json.loads((B/'calibration/analysis/loro_coverage.json').read_text())
rows=list(csv.DictReader((B/'confirmation/analysis/seed_results.csv').open()))
D={}
def ci(a):
 a=np.asarray(a,float);rng=np.random.default_rng(78131);z=a[rng.integers(0,len(a),(10000,len(a)))].mean(1)
 return dict(mean=float(a.mean()),lo=float(np.quantile(z,.025)),hi=float(np.quantile(z,.975)),n=len(a))
def summ(a):
 a=np.array(a,float);return dict(mean=float(a.mean()),median=float(np.median(a)),q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75)))
def corr(a,b):return float(np.corrcoef(np.array(a).ravel(),np.array(b).ravel())[0,1])
for idx,h in enumerate([4,12]):
 A=np.array([r['audit_a'][idx] for r in roots]);BB=np.array([r['audit_b'][idx] for r in roots]); S=np.array([r['selection'][idx] for r in roots]);P=np.array([r['proxy'] for r in roots]); Am=A.mean(-1);Bm=BB.mean(-1);Y=(Am+Bm)/2;SA=S.mean(-1)
 fit=frozen['joint_fits'][str(h)]['spec'];M=P+np.array(fit['mu']);allM=np.c_[np.zeros(30),M];order=np.argsort(-allM,axis=1);marg=allM[np.arange(30),order[:,0]]-allM[np.arange(30),order[:,1]]
 a=np.c_[np.zeros(30),Am];b=np.c_[np.zeros(30),Bm];y=(a+b)/2; ar=np.argmax(a,axis=1);br=np.argmax(b,axis=1)
 cross=(b[np.arange(30),ar]+a[np.arange(30),br])/2
 lr=loro['by_adaptation'][str(h)]['per_root'];res=np.array([np.array(x['heldout_audit_mean'])-x['predicted_means'] for x in lr]); vv=np.array([np.array(x['prior_marginal_variance'])+x['audit_measurement_variance'] for x in lr]);z=res/np.sqrt(vv)
 data=dict(audit_AB_sign_agree=float(np.mean((Am>0)==(Bm>0))),audit_AB_topchoice_agree=float(np.mean(ar==br)),audit_AB_pearson_all=corr(Am,Bm),audit_AB_mean_withinroot_rank_correlation=float(np.mean([corr(np.argsort(np.argsort(x)),np.argsort(np.argsort(y))) for x,y in zip(Am,Bm)])),selection_vs_combined_audit_corr=corr(SA,Y),query4_paired_mean_SE=summ(S.std(-1,ddof=1)/2),calibration_estimated_query_SE=np.sqrt(fit['R']).tolist(),prior_top_two_margin=summ(marg),fraction_root_median_query_SE_exceeds_margin=float(np.mean(np.median(S.std(-1,ddof=1)/2,axis=1)>marg)),cross_audit_information_diagnostic=ci(cross),empirical_same_audit_max_optimistically_biased=ci(y.max(1)),loro_candidate_mean_residual=res.mean(0).tolist(),loro_candidate_residual_RMSE=np.sqrt((res**2).mean(0)).tolist(),loro_candidate_standardized_residual_mean=z.mean(0).tolist(),loro_candidate_standardized_residual_rms=np.sqrt((z*z).mean(0)).tolist(),loro_candidate_coverage=np.mean(np.abs(z)<=1.96,0).tolist(),confirmation_candidate_mean_gain=Y.mean(0).tolist(),confirmation_candidate_residual_mean=(Y-M).mean(0).tolist())
 for q in [1,2,4]:
  rr=[r for r in rows if int(r['adaptation'])==h and int(r['budget_queries'])==q];methods=sorted(set(r['method'] for r in rr));scores={m:np.array([float(r['selected_audit_gain']) for r in rr if r['method']==m]) for m in methods}
  gains={m:ci(v) for m,v in scores.items()}; contrasts={f'kg_minus_{m}':ci(scores['pivot_kg']-scores[m]) for m in methods if m!='pivot_kg'}
  comp={};outcomes={}
  for r in roots:
   d=json.loads((B/f"confirmation/analysis/decisions/seed_{r['seed']}_h{h}.json").read_text());ds=[x for x in d['joint_decisions'] if x['budget_queries']==q];dd={x['method']:x for x in ds};kg=dd['pivot_kg']
   for m,v in dd.items():
    if m=='pivot_kg':continue
    vchoices=v.get('selected_distribution',[v['selected_idx']]);vqs=v.get('queries_by_draw',[v.get('queries',[])])
    metrics=comp.setdefault(m,{'choice_agree':[],'query_set_agree':[],'first_query_agree':[]})
    metrics['choice_agree'].append(float(np.mean([j==kg['selected_idx'] for j in vchoices])))
    if vqs[0]:
     metrics['query_set_agree'].append(float(np.mean([set(j)==set(kg['queries']) for j in vqs])))
     metrics['first_query_agree'].append(float(np.mean([j[0]==kg['queries'][0] for j in vqs])))
   for m,v in dd.items():
    outcomes.setdefault(m,[]).extend(v.get('selected_distribution',[v['selected_idx']]))
  data[str(q)]=dict(method_gain=gains,paired_contrasts=contrasts,cross_audit_minus_KG=ci(cross-scores['pivot_kg']),cross_audit_minus_noHF=ci(cross-scores['no_hf_v2']),cross_audit_minus_proxy=ci(cross-scores['proxy_only']),agreement={m:{k:float(np.mean(v)) if v else None for k,v in mm.items()} for m,mm in comp.items()},selected_hist={m:{str(i):v.count(i) for i in range(7)} for m,v in outcomes.items()})
 D[str(h)]=data
D['notes']=['Post hoc diagnostic on already completed first cohort; does not alter any sealed outcome or primary endpoint.','Cross-audit A-select-B-evaluate and reverse use 8 rollout gains per candidate. They estimate a particular noisy full-information policy, not an exact oracle upper bound.','Intervals are descriptive root bootstrap; no multiplicity-controlled confirmatory inference for these new diagnostics.']
(O/'diagnostics.json').write_text(json.dumps(D,indent=2)+'\n')
for h,d in D.items():
 if h=='notes':continue
 print('RESPONSE',h)
 for k,v in d.items():
  if k not in ['1','2','4']: print(k,v)
 print('PRIMARY Q2',json.dumps(d['2'],indent=2))
