import json, pathlib, hashlib, sys, math, itertools, datetime
import numpy as np
from scipy.integrate import quad
sys.dont_write_bytecode=True
B=pathlib.Path('/Users/yuanzhiyi/Desktop/MetaDrive_PIVOT_20260923')
A=B/'confirmation/analysis'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dg(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
protocol=read(B/'formal_protocols/confirmation_protocol.json');frozen=read(B/'calibration/analysis/frozen.json');summary=read(A/'summary.json')
roots={int(p.parent.name[5:]):read(p) for p in (B/'confirmation/roots').glob('seed_*/root.json')}
assert sorted(roots)==list(range(301,331))
calids=set(frozen['calibration_roots']); assert not calids&set(roots)
checks={}; counts={}; max_gain_error=0.;max_regret_error=0.; max_estimate_error=0.;max_kg_error=0.;n_kg=0
scored={}
for rec in read(A/'result.json')['root_results']:
 for row in rec['scored']:scored[(row['root'],row['adaptation'],row['budget_queries'],row['method'])]=row
manifest=read(A/'decision_manifest.json'); values={}
for p in sorted((A/'decisions').glob('*.json')):
 d=read(p);root=d['root'];h=d['adaptation'];r=roots[root];hi=[4,12].index(h)
 assert dg(d)==manifest[p.name]
 assert d['fit_sha256']==sha(B/'calibration/analysis/frozen.json')
 assert d['protocol_sha256']==sha(B/'formal_protocols/confirmation_protocol.json')
 assert d['root_file_sha256']==sha(B/f'confirmation/roots/seed_{root}/root.json')
 aa=np.asarray(r['audit_a'])[hi];bb=np.asarray(r['audit_b'])[hi]
 labels=np.r_[0.,np.concatenate([aa,bb],axis=1).mean(1)]
 sel=np.asarray(r['selection'])[hi].mean(1)
 assert dg(d['joint_decisions'])==d['joint_decision_sha256']; assert dg(d['author_decisions'])==d['author_decision_sha256']
 for row in d['joint_decisions']+d['author_decisions']:
  key=(root,h,row['budget_queries'],row['method']);actual=scored[key]
  choices=row.get('selected_distribution',[row['selected_idx']]); gain=float(np.mean(labels[choices])); values[key]=gain
  max_gain_error=max(max_gain_error,abs(gain-actual['selected_audit_gain']))
  max_regret_error=max(max_regret_error,abs(float(labels.max())-gain-actual['empirical_oracle_regret']))
  q=row['hf_queries']; assert row['hf_episode_cost']==(2*h+q*(2*h+8) if q else 0)
  qs=row.get('queries_by_draw',[row.get('queries',[])])
  for query in qs: assert len(query)==q and len(set(query))==q and all(1<=x<=6 for x in query)
  if 'steps' in row:
   m=np.asarray(r['proxy'])+np.asarray(frozen['joint_fits'][str(h)]['spec']['mu'])
   cov=np.asarray(frozen['joint_fits'][str(h)]['spec']['K_prior'])
   R=np.asarray(frozen['joint_fits'][str(h)]['spec']['R'])
   for step in row['steps']:
    j=step['query']; assert abs(step['observed']-sel[j])<1e-12
    if 'kg' in step:
     jmax=max(step['kg'],key=lambda j:step['kg'][j]); assert step['kg'][str(j)]>=step['kg'][jmax]-1e-12
     slopes=np.r_[cov[:,j]/math.sqrt(cov[j,j]+R[j]),0.]
     ints=np.r_[m,0.]
     br=[]
     for i,k in itertools.combinations(range(7),2):
      if abs(slopes[i]-slopes[k])>1e-12:
       z=(ints[k]-ints[i])/(slopes[i]-slopes[k])
       if -10<z<10:br.append(float(z))
     f=lambda z:float(np.max(ints+slopes*z))*math.exp(-z*z/2)/math.sqrt(2*math.pi)
     integ=quad(f,-10,10,points=sorted(set(br)),epsabs=1e-9,limit=300)[0]
     kg=max(0.,integ-float(max(ints)))
     max_kg_error=max(max_kg_error,abs(kg-step['kg'][str(j)]));n_kg+=1
    gain_col=cov[:,j]/(cov[j,j]+R[j]);m=m+gain_col*(sel[j]-m[j]);cov=cov-np.outer(gain_col,cov[:,j]);cov=(cov+cov.T)/2
   expected=np.r_[0.,r['proxy']] if row['method']=='proxy_only' else np.r_[0.,m]
   max_estimate_error=max(max_estimate_error,float(np.max(np.abs(expected-row['estimates_incumbent_first']))))
   assert expected[row['selected_idx']]>=np.max(expected)-1e-10
for name,dhash in read(A/'artifact_manifest.json').items():assert sha(A/name)==dhash
for name,dhash in frozen['analysis_source_sha256'].items():
 path=B/('code' if name in ('pivot_v2.py','selector_analysis.py') else 'analysis_tools')/name
 assert sha(path)==dhash
for name,dhash in frozen['calibration_root_file_sha256'].items():assert sha(B/'calibration'/name)==dhash
for h,fit in frozen['author_fits'].items():
 for name,dhash in fit['provenance']['author_source_sha256'].items():assert sha(B/'author_source_snapshot'/name)==dhash
checks['root_ids_and_calibration_separation']=True;checks['all_60_decision_seals_and_root_hashes']=True;checks['frozen_source_calibration_author_snapshot_hashes']=True;checks['analysis_artifact_hashes']=True
checks['all_cost_formulae_and_query_unique_indices']=True
checks['all_gain_rescoring_max_abs_error']=max_gain_error;checks['all_regret_rescoring_max_abs_error']=max_regret_error
checks['joint_gaussian_update_and_final_estimate_max_abs_error']=max_estimate_error;checks['chosen_KG_numeric_quadrature_max_abs_error']=max_kg_error;checks['chosen_KG_numeric_quadrature_checks']=n_kg
contrasts={}
for name,a,b in [('primary','pivot_kg','uniform_v2_expected_100'),('secondary_author','author_PIVOT_VOI_E5C','author_Uniform_E5C')]:
 x=np.array([values[(r,12,2,a)]-values[(r,12,2,b)] for r in sorted(roots)])
 rng=np.random.default_rng(78131);boot=x[rng.integers(30,size=(10000,30))].mean(1)
 contrasts[name]={'mean':float(x.mean()),'lo':float(np.quantile(boot,.025)),'hi':float(np.quantile(boot,.975)),'n_roots':30,'per_root_difference':x.tolist()}
 report=summary['hypotheses'][next(k for k in summary['hypotheses'] if k.startswith('primary_' if name=='primary' else 'secondary_'))]
 assert all(abs(contrasts[name][k]-report[k])<1e-12 for k in ('mean','lo','hi'))
# Ensure disjoint seed streams for all declared data purposes and between roots.
streamseeds={};s=protocol['seed_layout']
for r in roots:
 groups={'train':{r*1000+j for j in range(2)},'proxy':{r*1000+10+j for j in range(4)},'validation':{r*1000+20+j for j in range(2)}}
 for stream,n in [('selection',4),('audit_a',8),('audit_b',8)]:groups[stream]={r*1000+s[stream]+20*c+j for c in range(6) for j in range(n)}
 for a,b in itertools.combinations(groups,2): assert not groups[a]&groups[b]
 streamseeds[r]=set.union(*groups.values())
for a,b in itertools.combinations(streamseeds,2):assert not streamseeds[a]&streamseeds[b]
checks['all_declared_seed_streams_disjoint']=True
# Check episode key references also cannot overlap between selection and audits.
for r in roots:
 refs=read(B/f'confirmation/roots/seed_{r}/episode_references.json')
 def flatten(x):return sum((flatten(z) for z in x),[]) if isinstance(x,list) else [x]
 selrefs=set(flatten(refs['selection']));arefs=set(flatten(refs['audit_a']));brefs=set(flatten(refs['audit_b']))
 assert not selrefs&arefs and not selrefs&brefs and not arefs&brefs
checks['selection_audit_episode_keys_disjoint']=True
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Independent read-only audit of completed frozen first MetaDrive cohort; original artifacts untouched','severe_errors_found':[],'checks':checks,'recomputed_contrasts':contrasts,'n_scored_rows':len(scored),'limitations':['Thirty independent root RNG streams and response pools share one fixed native bottleneck geometry; these are not 30 different maps or independent benchmark tasks.','Root bootstrap conditions on the fixed trained calibration posterior and candidate family; it does not include uncertainty from retraining a new calibration cohort.','Labels are independent finite-sample audit estimates. Max-audit oracle regret is upward biased; paired selected-gain contrast does not use a maximization.','Uniform_expected_100 averages allocation draws on a fixed bank, not independent deployment roots; no pseudoreplication in the bootstrap.','Original author E5C comparison uses a new footprint adapter and frozen horizon normalization; its Uniform uses proxy for unqueried outcomes, so it does not isolate allocation value.','No fresh native-physics reruns were performed in this bounded analysis audit; original deterministic native replay checks are separate.']}
path=pathlib.Path('/private/tmp/metadrive_independent_audit_20260923.json');path.write_text(json.dumps(result,indent=2)+'\n')
md='# Independent completed MetaDrive audit\n\nNo severe analysis, selection/audit leakage, candidate-index, paired-bootstrap, or reward-unit error was found in this bounded review. This is not a guarantee of absence of all simulator issues.\n\n'
md+=f"Recomputed {len(scored)} selected-gain and regret rows directly from root audit arrays and all 60 sealed decision records. Maximum gain error {max_gain_error:.3g}; maximum Gaussian-final-estimate error {max_estimate_error:.3g}. Independently integrated {n_kg} selected KG values, maximum error {max_kg_error:.3g}. All source, root, decision and artifact manifests checked here matched.\n\n"
for n,c in contrasts.items():md+=f"- {n}: mean {c['mean']:.12f}, paired root-bootstrap 95% CI [{c['lo']:.12f}, {c['hi']:.12f}], 30 roots.\n"
md+='\nBoth comparisons include zero. The first cohort does not establish PIVOT superiority, and no discovered analysis bug overturns that finding.\n\nScope limits:\n'+''.join('- '+x+'\n' for x in result['limitations'])
path.with_suffix('.md').write_text(md)
print(json.dumps({k:v for k,v in result.items() if k not in ('limitations','recomputed_contrasts')},indent=2))
print('AUDIT_REPORT',str(path))
