import argparse,json,pathlib,hashlib,sys
import native_adapter
from native_adapter import NativeEvaluator
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--checks',required=True);a=p.parse_args()
files=sorted((pathlib.Path(a.out)/'episodes').glob('*.json'))
# Deterministic content-name selection, not based on returns or methods.
sel=[files[i] for i in [0,len(files)//2,len(files)-1]];results=[]
for f in sel:
 d=json.loads(f.read_text());t=d['task']
 assert hashlib.sha256(pathlib.Path(native_adapter.__file__).read_bytes()).hexdigest()==t['native_adapter_sha256']
 with NativeEvaluator(t['environment']) as e:r=e.run_episode(t['seed'],t['ego'],t['background'])
 diff=max(abs(r['per_agent_returns'][k]-d['per_agent_returns'][k]) for k in r['per_agent_returns'])
 results.append({'episode_file':f.name,'seed':t['seed'],'max_return_difference':diff,'trajectory_hash_matches':r['trajectory_sha256']==d['trajectory_sha256'],'initial_positions_match':r['initial_positions']==d['initial_positions']})
passed=all(x['trajectory_hash_matches'] and x['initial_positions_match'] and x['max_return_difference']==0 for x in results)
c=json.loads(pathlib.Path(a.checks).read_text());c['checks']['saved_bank_replay_three_episodes']=passed;c['all_pass']=all(c['checks'].values());c['bank_replay']=results
pathlib.Path(a.checks).write_text(json.dumps(c,indent=2)+'\n');print(json.dumps(c));sys.exit(0 if passed else 2)
