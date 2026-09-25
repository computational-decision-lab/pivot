from pathlib import Path
import json,hashlib,itertools,datetime
import numpy as np
B=Path('/root/autodl-tmp/metadrive_precision_20260923')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def canon(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(p.read_text())
def flatten(o):
 if isinstance(o,list):
  for v in o:yield from flatten(v)
 else:yield o
def check_manifest(p):
 n=0;errs=[]
 for line in p.read_text().splitlines():
  h,name=line.split('  ',1);n+=1
  if sha(B/name)!=h:errs.append(name)
 return {'entries':n,'mismatches':errs,'pass':not errs}
report={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Independent read-only integrity, pairing, partition, cost and metric recomputation; not a rerun or new experiment.','source_manifest':check_manifest(B/'SOURCE_PROTOCOL_SHA256SUMS.txt'),'confirmation_freeze':check_manifest(B/'checks/CONFIRMATION_FREEZE.sha256'),'exit_code':int((B/'checks/LAUNCH_ONCE/exit_code.txt').read_text()),'phases':{}}
errs=[];totalsteps=totalagent=totalepisodes=0;totalseconds=0.;allsplits={};rootmap={};paircount=0;maxgain=0.;pairbad=0;taskbad=0;missingrefs=0
for phase,seeds in [('calibration',range(401,413)),('confirmation',range(501,531))]:
 pdir=B/phase;prot=read(pdir/'protocol.json');status=read(pdir/'status.json');roots=sorted(pdir.glob('roots/seed_*/root.json'),key=lambda p:int(p.parent.name[5:]));expected=list(seeds)
 if [read(p)['seed'] for p in roots]!=expected:errs.append(phase+': root IDs mismatch')
 data={};steps=agents=0;secs=0.;scenesets={k:set() for k in ['training','proxy','validation','selection','audit_a','audit_b']};used=set();collisions=[];rooterrors=[]
 for ep in sorted((pdir/'episodes').glob('*.json')):
  e=read(ep);data[ep.stem]=e
  if canon(e['task'])!=ep.stem:taskbad+=1
  steps+=int(e['steps']);agents+=int(e['actual_agent_steps']);secs+=float(e['seconds'])
 for rp in roots:
  root=read(rp);rootmap[root['seed']]=root;ref=read(rp.parent/'episode_references.json');ss={}
  for stream in scenesets:
   keys=set(flatten(ref[stream]));used|=keys
   missing=[k for k in keys if k not in data];missingrefs+=len(missing)
   if missing:continue
   ss[stream]={data[k]['task']['seed'] for k in keys};scenesets[stream]|=ss[stream]
  for a,b in itertools.combinations(ss,2):
   if ss[a]&ss[b]:collisions.append([root['seed'],a,b,len(ss[a]&ss[b])])
  def paired_array(n):
   global paircount,pairbad
   out=[]
   for cond in ref[n]:
    cs=[]
    for cand in cond:
     gs=[]
     for ka,kb in cand:
      a,b=data[ka],data[kb];paircount+=1
      if a['initial_positions']!=b['initial_positions'] or a['task']['seed']!=b['task']['seed']:pairbad+=1
      gs.append(a['ego_return']-b['ego_return'])
     cs.append(gs)
    out.append(cs)
   return np.array(out)
  aa=paired_array('audit_a');bb=paired_array('audit_b');sel=paired_array('selection');proxy=np.array([[data[k]['ego_return'] for k in ks] for ks in ref['proxy']]);val=np.array([[[data[k]['background_mean_return'] for k in ks] for ks in cs] for cs in ref['validation']]);train=np.array([[[data[k]['background_mean_return'] for k in ks] for ks in pool] for pool in ref['training']]);response=[[int(np.argmax(train[i,:h].mean(axis=1))) for h in [4,12]] for i in range(7)]
  if response!=root['response_indices']:rooterrors.append([root['seed'],'response indices'])
  values={'proxy':(proxy[1:]-proxy[0]).mean(axis=1),'proxy_paired_episodes':proxy[1:]-proxy[0],'selection':sel,'audit_a':aa[1:],'audit_b':bb[1:],'audit_fixed_a':aa[0],'audit_fixed_b':bb[0],'response_validation':val}
  for k,computed in values.items():
   d=float(np.max(np.abs(computed-np.array(root[k]))));maxgain=max(maxgain,d)
   if d>1e-10:rooterrors.append([root['seed'],k,d])
 unmatched=set(data)-used
 if collisions:errs.append(phase+': stream overlap')
 if rooterrors:errs.append(phase+': raw-to-root mismatch')
 if unmatched:errs.append(phase+': unreferenced episode files')
 if len(data)!=status['unique_native_episodes']:errs.append(phase+': episode count mismatch')
 report['phases'][phase]={'root_count':len(roots),'root_ids':expected,'unique_native_episodes':len(data),'all_episode_files_referenced':not unmatched,'unreferenced_count':len(unmatched),'native_environment_steps':steps,'native_agent_steps':agents,'sum_episode_compute_seconds':secs,'wallclock_seconds':status['elapsed_seconds'],'stream_seed_counts':{k:len(v) for k,v in scenesets.items()},'stream_overlaps':collisions,'root_recomputation_mismatches':rooterrors}
 allsplits[phase]=set.union(*scenesets.values());totalepisodes+=len(data);totalsteps+=steps;totalagent+=agents;totalseconds+=secs
 if phase=='confirmation':
  ledger=read(pdir/'analysis/summary.json')['native_experiment_cost']
  if ledger['unique_generated_native_episodes']!=len(data) or ledger['native_environment_steps']!=steps or ledger['native_agent_steps']!=agents or abs(ledger['sum_episode_compute_seconds']-secs)>1e-5:errs.append('confirmation cost ledger mismatch')
 del data
report['cross_phase_scene_seed_overlap']=len(allsplits['calibration']&allsplits['confirmation'])
if report['cross_phase_scene_seed_overlap']:errs.append('calibration/confirmation scene overlap')
analysis=B/'confirmation/analysis';dm=read(analysis/'decision_manifest.json');am=read(analysis/'artifact_manifest.json');decisions=list((analysis/'decisions').glob('*.json'));badseal=[]
fitsha=sha(B/'calibration/analysis/frozen.json');protsha=sha(B/'formal_protocols/confirmation_protocol.json')
for dp in decisions:
 d=read(dp)
 for k,l in [('joint_decisions','joint_decision_sha256'),('author_decisions','author_decision_sha256')]:
  if canon(d[k])!=d[l]:badseal.append([dp.name,l])
 if canon(d)!=dm[dp.name]:badseal.append([dp.name,'manifest'])
 if d['fit_sha256']!=fitsha or d['protocol_sha256']!=protsha or d['root_file_sha256']!=sha(B/f"confirmation/roots/seed_{d['root']}/root.json"):badseal.append([dp.name,'input binding'])
artifactbad=[k for k,v in am.items() if sha(analysis/k)!=v]
res=read(analysis/'result.json');scored=0;maxscore=0.;costbad=[];primary={};author={}
for rr in res['root_results']:
 root=rootmap[rr['root']];h=rr['adaptation'];hi=[4,12].index(h);labels=np.r_[0.,(np.mean(root['audit_a'][hi],axis=1)+np.mean(root['audit_b'][hi],axis=1))/2.]
 for row in rr['scored']:
  idx=row.get('selected_distribution',[row['selected_idx']]);g=float(np.mean(labels[np.asarray(idx,dtype=int)]));err=max(abs(g-row['selected_audit_gain']),abs(float(labels.max())-g-row['empirical_oracle_regret']));maxscore=max(maxscore,err);scored+=1
  q=row['hf_queries'];charge=0 if q==0 else 2*h+q*(2*h+32)
  if row['hf_episode_cost']!=charge or row['hf_episode_cost']>row['hf_episode_budget']:costbad.append([row['root'],h,row['method'],row['budget_queries']])
  if h==12 and row['budget_queries']==2:
   if row['method'] in ['pivot_kg','uniform_v2_expected_100']:primary.setdefault(row['root'],{})[row['method']]=g
   if row['method'] in ['author_PIVOT_VOI_E5C','author_Uniform_E5C']:author.setdefault(row['root'],{})[row['method']]=g

def contrast(rows,a,b):
 x=np.array([rows[k][a]-rows[k][b] for k in sorted(rows)]);rng=np.random.default_rng(78131);means=x[rng.integers(len(x),size=(10000,len(x)))].mean(axis=1);lo,hi=np.quantile(means,[.025,.975]);return {'mean':float(x.mean()),'lo':float(lo),'hi':float(hi),'roots':len(x)}
pc=contrast(primary,'pivot_kg','uniform_v2_expected_100');ac=contrast(author,'author_PIVOT_VOI_E5C','author_Uniform_E5C');pd=read(analysis/'precision_decision.json')
for k in ['mean','lo','hi']:
 if abs(pc[k]-pd['primary_gain_contrast'][k])>1e-10:errs.append('primary contrast mismatch '+k)
if taskbad or missingrefs or pairbad or badseal or artifactbad or costbad or maxscore>1e-10:errs.append('one or more integrity recomputation failures')
report.update({'task_key_mismatches':taskbad,'missing_episode_references':missingrefs,'paired_comparisons_checked':paircount,'paired_initial_condition_or_seed_mismatches':pairbad,'max_raw_to_root_abs_error':maxgain,'decision_files':len(decisions),'expected_decision_files':60,'decision_seal_mismatches':badseal,'artifact_manifest_files':len(am),'artifact_manifest_mismatches':artifactbad,'scored_rows_checked':scored,'max_recomputed_gain_regret_abs_error':maxscore,'cost_rule_mismatches':costbad,'primary_recomputed':pc,'author_secondary_recomputed':ac,'combined_prespecified_meaningful_success':bool(pc['mean']>=2 and pc['lo']>0),'total_unique_native_episodes':totalepisodes,'total_native_environment_steps':totalsteps,'total_native_agent_steps':totalagent,'total_episode_compute_seconds':totalseconds,'pipeline_wallclock_seconds':(datetime.datetime.fromisoformat((B/'checks/LAUNCH_ONCE/ended_utc.txt').read_text().strip().replace('Z','+00:00'))-datetime.datetime.fromisoformat((B/'checks/LAUNCH_ONCE/started_utc.txt').read_text().strip().replace('Z','+00:00'))).total_seconds(),'cloud_currency_cost':'Not asserted: provider billing rate/ledger not inspected by this integrity audit. Runtime and native simulation costs are reported.','issues':errs,'pass':not errs and len(decisions)==60 and report['source_manifest']['pass'] and report['confirmation_freeze']['pass'] and report['exit_code']==0})
print(json.dumps(report,indent=2,allow_nan=False))
