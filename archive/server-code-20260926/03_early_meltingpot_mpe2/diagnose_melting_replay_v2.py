"""Instrument existing official rollout without changing actions or rewards.

Run sequences in separate processes to locate the first differing observation,
recurrent state, action, or return. Diagnostic episodes are not effect data.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,time
from pathlib import Path

def digest(value):
    h=hashlib.sha256()
    def visit(x):
        if hasattr(x,'numpy'): x=x.numpy()
        if hasattr(x,'tobytes'):
            h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x): h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(str(type(x).__name__).encode())
            for a in x:visit(a)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()

class Traced:
    def __init__(self,policy,label,trace): self.policy=policy;self.label=label;self.trace=trace
    def initial_state(self):
        s=self.policy.initial_state();self.trace.append({'kind':'init','skill':self.label,'state':digest(s)});return s
    def step(self,timestep,state):
        row={'kind':'step','skill':self.label,'observation':digest(timestep),'state_in':digest(state)}
        a,s=self.policy.step(timestep,state)
        row.update(action=int(a),state_out=digest(s));self.trace.append(row)
        return a,s

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',type=Path,required=True);p.add_argument('--crossbench-root',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True);p.add_argument('--sequence',default='A,A,B,A')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from melting_reference_diagnostic import OfficialSpecialistRollout
    cases=json.loads(a.cases.read_text());a.output.mkdir(parents=True,exist_ok=False)
    backend=OfficialSpecialistRollout(a.model_path,task='melting_stag',horizon=2500,crossbench_root=a.crossbench_root)
    trace=[];backend.policies={k:Traced(v,k,trace) for k,v in backend.policies.items()}
    records=[];previous={}
    try:
        for i,name in enumerate(a.sequence.split(',')):
            trace.clear();case=cases[name];sig=case['signature']
            r,events=backend.episode(sig['matchup'],sig['environment_seed'],sig['policy_seeds'],time.monotonic()+180,2500)
            target=a.output/f'{i}_{name}.trace.json.gz'
            with gzip.open(target,'wt') as f: json.dump(trace,f)
            row={'order':i,'case':name,'original_trajectory_match':r['trajectory_sha256']==case['trajectory_sha256'],
                 'rollout':r,'trace_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'trace_file':target.name}
            if name in previous:
                old=previous[name];idx=next((j for j,(x,y) in enumerate(zip(old,trace)) if x!=y),None)
                row['repeat_matches']=old==trace
                row['first_difference']=None if idx is None else {'index':idx,'previous':old[idx],'current':trace[idx]}
                row['trace_lengths']=[len(old),len(trace)]
            previous[name]=list(trace);records.append(row)
            (a.output/'summary.json').write_text(json.dumps({'status':'running','model':backend.metadata,'records':records},indent=2))
            print(json.dumps({k:v for k,v in row.items() if k!='rollout'}),flush=True)
    finally: backend.close()
    (a.output/'summary.json').write_text(json.dumps({'status':'complete','native_episodes':len(records),'used_for_effect_estimation':False,'model':backend.metadata,'records':records},indent=2))
if __name__=='__main__': main()
