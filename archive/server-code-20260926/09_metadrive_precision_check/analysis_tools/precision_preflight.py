"""Operational checks for the frozen second cohort. No native episodes generated."""
import argparse, ast, hashlib, itertools, json, pathlib, sys
import numpy as np
import selector_analysis as selectors
import author_adapter as author
import native_adapter, run_benchmark, analyze_formal


def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=pathlib.Path,required=True);a=p.parse_args();base=a.base
 protocols=[json.loads((base/f'formal_protocols/{x}_protocol.json').read_text()) for x in ('calibration','confirmation')]
 cal,con=protocols
 assert cal['root_seeds']==list(range(401,413)) and con['root_seeds']==list(range(501,531))
 assert {k:v for k,v in cal.items() if k not in ('phase','root_seeds')}=={k:v for k,v in con.items() if k not in ('phase','root_seeds')}
 assert cal['proxy_episodes']==16 and cal['selection_episodes']==16 and cal['audit_episodes_per_split']==32
 assert cal['response']['training_episodes_per_profile']==2 and cal['response_validation_episodes']==2
 assert cal['response']['adaptation_steps']==[4,12] and cal['response']['pool_size']==12
 upper=7*12*2+7*16+7*3*2+2*6*16*2+2*3*6*32*2
 assert upper==3010 and upper*42==126420 and upper*42<=cal['runtime_limits']['cohort_native_episode_cap']
 allseeds={}
 for root in cal['root_seeds']+con['root_seeds']:
  groups={}
  for stream,n,k in [('train',2,1),('proxy',16,1),('response_validation',2,1),('selection',16,6),('audit_a',32,6),('audit_b',32,6)]:
   groups[stream]={run_benchmark.scene(cal,root,stream,c,j) for c in range(k) for j in range(n)}
  for x,y in itertools.combinations(groups,2):assert not groups[x]&groups[y]
  allseeds[root]=set.union(*groups.values())
  assert min(allseeds[root])>=root*10000 and max(allseeds[root])<root*10000+10000
 for x,y in itertools.combinations(allseeds,2):assert not allseeds[x]&allseeds[y]
 # Synthetic arrays test imports, adaptation dimensions and every selector's cost.
 # They are explicitly synthetic and never stored under calibration/confirmation.
 rng=np.random.default_rng(64921)
 def world(r):return {'seed':r,'adaptation_steps':[4,12],'proxy':rng.normal(size=6).tolist(),'selection':rng.normal(size=(2,6,16)).tolist(),'audit_a':rng.normal(size=(2,6,32)).tolist(),'audit_b':rng.normal(size=(2,6,32)).tolist()}
 train=[world(x) for x in range(401,413)];root=world(501)
 class NoAudit(dict):
  def __getitem__(self,k):
   if k in ('audit_a','audit_b'):raise AssertionError('audit accessed during decision')
   return super().__getitem__(k)
  def get(self,k,d=None):
   if k in ('audit_a','audit_b'):raise AssertionError('audit accessed during decision')
   return super().get(k,d)
 decision_count=0
 for h in (4,12):
  jf=selectors.fit_model(train,h);af=author.fit_author(train,h,cal)
  j=selectors.decide_root(NoAudit(root),jf);u=author.decide_author(NoAudit(root),af)
  for d in j+u:
   q=d['hf_queries'];assert d['hf_episode_cost']==(2*h+q*(2*h+32) if q else 0)
  assert [d for d in j if d['method']=='pivot_kg' and d['budget_queries']==2][0]['hf_episode_cost']==(88 if h==4 else 136)
  selectors.score_root(root,jf,j,selectors.seal_decisions(j));author.score_author(root,af,u,author._digest(u));decision_count+=len(j+u)
 # Syntax-only compilation without touching source bytecode files.
 for d in (base/'code',base/'analysis_tools'):
  for f in d.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
 # The future precision protocol is frozen by this physical file hash.
 plan_sha=hashlib.sha256((base/'formal_protocols/formal_analysis_plan.json').read_bytes()).hexdigest()
 assert cal['formal_analysis_plan_sha256']==plan_sha
 print(json.dumps({'status':'PREFLIGHT_OK','data_kind':'synthetic_checks_only_no_native_episodes','cohort_raw_episode_upper_bound':upper*42,'calibration_raw_upper_bound':upper*12,'confirmation_raw_upper_bound':upper*30,'all_seed_streams_disjoint':True,'candidate_count':6,'response_training_episodes_unchanged':2,'synthetic_decisions_checked':decision_count,'primary_h12_q2_cost':136,'new_protocol_sha256':plan_sha}))

if __name__=='__main__':main()
