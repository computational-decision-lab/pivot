"""Read-only initialization diagnostic; no policy learning or effect estimates."""
import argparse, hashlib, json, random, sys
from pathlib import Path
import numpy as np

def h(x):
    return hashlib.sha256(x.tobytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--crossbench-root',required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,a.crossbench_root)
    from crossbench.melting import Parallel
    from melting_reference_diagnostic import reference_parallel_class
    C=reference_parallel_class(Parallel);cases=json.loads(a.cases.read_text());a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for mode in ['unaltered','seed_python_numpy']:
        for i in range(16):
            seed=cases['A']['signature']['environment_seed']
            if mode=='seed_python_numpy':random.seed(seed);np.random.seed(seed)
            e=C('melting_stag',2500)
            settings=e.cfg.lab2d_settings_builder(roles=e.cfg.default_player_roles,config=e.cfg)
            settings_hash=hashlib.sha256(json.dumps(settings,sort_keys=True,default=str).encode()).hexdigest()
            e.reset(seed=seed)
            try:
                obs=e.local_policy_timesteps
                values={f'player_{k}_{name}':np.array(v) for k,o in enumerate(obs) for name,v in o.observation.items()}
                np.savez_compressed(a.output/f'{mode}_{i}.npz',**values)
                rows.append({'mode':mode,'index':i,'environment_seed':seed,'settings_sha256':settings_hash,
                    'initial_observations':{k:{'hash':h(v),'shape':list(v.shape),'dtype':str(v.dtype),'sum':float(v.sum())} for k,v in values.items()}})
            finally:e.close()
    (a.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({m: {'settings_hashes':len({r['settings_sha256'] for r in rows if r['mode']==m}),
        'initial_rgb_hashes':len({r['initial_observations']['player_0_RGB']['hash'] for r in rows if r['mode']==m})} for m in ['unaltered','seed_python_numpy']}))
if __name__=='__main__':main()
