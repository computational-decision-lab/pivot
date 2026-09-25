import json,numpy as np,itertools
from pathlib import Path
B=Path('/Users/yuanzhiyi/Desktop/MetaDrive_PIVOT_20260923');O=Path('/private/tmp/metadrive_diagnostics_20260923'); roots=[json.loads(p.read_text()) for p in sorted((B/'confirmation/roots').glob('*/root.json'))];f=json.loads((B/'calibration/analysis/frozen.json').read_text())
def ci(a):
 a=np.array(a,float); z=a[np.random.default_rng(78131).integers(0,len(a),(10000,len(a)))].mean(1);return dict(mean=float(a.mean()),lo=float(np.quantile(z,.025)),hi=float(np.quantile(z,.975)),n=len(a))
bg=np.array([np.array(r['response_validation']).mean((0,2)) for r in roots]); gaps=[]; flips=[]
for r in roots:
 aa=np.array(r['audit_a']);bb=np.array(r['audit_b']);fa=np.array(r['audit_fixed_a']);fb=np.array(r['audit_fixed_b']);a=aa.mean(-1);b=bb.mean(-1);fixeda=fa.mean(-1);fixedb=fb.mean(-1)
 gap=((aa-fa).mean(-1)+(bb-fb).mean(-1))/2;gaps.append(np.abs(gap).mean(-1));flip=(fixeda>0)&(fixedb>0)&(a[1]<0)&(b[1]<0)
 for i in np.where(flip)[0]: flips.append(dict(root=r['seed'],candidate=r['candidate_names'][i],proxy=r['proxy'][i],fixed_A=float(fixeda[i]),fixed_B=float(fixedb[i]),long_A=float(a[1,i]),long_B=float(b[1,i])))
gaps=np.array(gaps);m=dict(roots=30,background_fixed=ci(bg[:,0]),background_short=ci(bg[:,1]),background_long=ci(bg[:,2]),background_long_minus_fixed=ci(bg[:,2]-bg[:,0]),background_long_minus_short=ci(bg[:,2]-bg[:,1]),absolute_paired_gap_short=ci(gaps[:,0]),absolute_paired_gap_long=ci(gaps[:,1]),gap_long_minus_short=ci(gaps[:,1]-gaps[:,0]),stable_fixed_positive_long_negative_count=len(flips),stable_flip_roots=len(set(x['root'] for x in flips)),stable_flips=flips)
(O/'mechanism_30roots.json').write_text(json.dumps(m,indent=2)+'\n');print('MECHANISM',json.dumps(m,indent=2))
# All 15 query subsets, fixed before reading observations; no choice of subsets or n by outcome.
# This is a uniform-allocation precision diagnostic, not a new formal method comparison.
# Calibration posterior stays frozen. Known independent-rollout R scales R4*4/n.
res={}
for hi,h in enumerate([4,12]):
 spec=f['joint_fits'][str(h)]['spec'];subsets=list(itertools.combinations(range(6),2)); rows=[]; choice_store={}
 for n in [1,2,4]:
  vals=[]; mse=[];within=[];chroot=[]
  for r in roots:
   y=(np.array(r['audit_a'][hi]).mean(-1)+np.array(r['audit_b'][hi]).mean(-1))/2;ya=np.r_[0.,y];obs=np.array(r['selection'][hi])[:,:n].mean(-1);gains=[];choices=[]
   for ss in subsets:
    mu=np.array(r['proxy'])+spec['mu']; K=np.array(spec['K_prior']);noise=np.array(spec['R'])*4/n
    for i in ss:
     g=K[:,i]/(K[i,i]+noise[i]);mu=mu+g*(obs[i]-mu[i]);K=K-np.outer(g,K[i,:])
    ch=int(np.argmax(np.r_[0.,mu]));choices.append(ch);gains.append(ya[ch])
   vals.append(np.mean(gains));mse.append(np.mean((obs-y)**2));chroot.append(choices)
  choice_store[n]=np.array(chroot)
  rows.append(dict(n=n,uniform_all_15_subsets_gain=ci(vals),query_mean_rmse_to_noisy_independent_audit=float(np.sqrt(np.mean(mse)))))
 for row in rows:
  row['choice_agreement_with_n4']=float(np.mean(choice_store[row['n']]==choice_store[4]))
 res[str(h)]={'replication_curve':rows,'notes':'Fixed exhaustive set of all15 candidate query pairs; averaged across subsets within root. Frozen calibrated posterior; R_n=R4*4/n. This diagnostics-only curve uses the first n selection episodes and independent combined audit. No extra simulator runs. n4 exact Uniform expectation differs from formal 100-draw Monte Carlo by integration error.'}
(O/'uniform_fixed_subset_precision.json').write_text(json.dumps(res,indent=2)+'\n');print('PRECISION',json.dumps(res,indent=2))
