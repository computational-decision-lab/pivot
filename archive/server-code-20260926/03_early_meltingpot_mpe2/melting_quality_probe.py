"""Bounded diagnostic of existing Melting Pot checkpoints; no training."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse,hashlib,json,pathlib,subprocess,sys,time
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def run(item):
    task,seed,out=item
    import torch
    torch.set_num_threads(1)
    from crossbench.experiment import load,evaluate
    from benchmark import seed_for
    folder=pathlib.Path('/root/autodl-tmp/colin_melting/runs/melting_cross_main')/task/'test'/f'seed_{seed}'
    cfg=json.loads((folder/'manifest.json').read_text())['config'];h=cfg['horizon']
    paths={k:folder/'initial'/n for k,n in [('trained','focal.zip'),('untrained','focal0_untrained.zip'),('opponent','response.zip')]}
    hashes={k:sha(p) for k,p in paths.items()}
    seeds=[seed_for(20260915,'melting_quality_probe_v1',task,seed,i) for i in range(16)]
    existing=set(json.loads((folder/'seeds.json').read_text()).values());assert not existing.intersection(seeds)
    before=time.monotonic();opp=load(paths['opponent']);trained=load(paths['trained']);untrained=load(paths['untrained'])
    rows={}
    for label,model in [('untrained',untrained),('trained',trained)]:rows[label]=evaluate(task,model,opp,seeds,h)
    gains=np.array([b['focal_return']-a['focal_return'] for a,b in zip(rows['untrained'],rows['trained'])])
    result={'task':task,'seed':seed,'episodes_per_policy':16,'horizon':h,'paired_gain':float(gains.mean()),'mean_trained_return':float(np.mean([r['focal_return'] for r in rows['trained']])),'mean_untrained_return':float(np.mean([r['focal_return'] for r in rows['untrained']])),'trained_nonzero_episode_fraction':float(np.mean([r['focal_return']!=0 for r in rows['trained']])),'checkpoints_sha256':hashes,'rows':rows,'seconds':round(time.monotonic()-before,3),'new_training_steps':0,'evaluation_steps':32*h}
    assert hashes=={k:sha(p) for k,p in paths.items()}
    path=pathlib.Path(out)/(task+'_'+str(seed)+'.json');path.write_text(json.dumps(result,indent=2)+'\n');return {k:v for k,v in result.items() if k!='rows'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    plan={'purpose':'development_quality_diagnostic_on_old_checkpoints','tasks':['melting_pd','melting_stag'],'seeds':[1300,1301,1302],'episodes':16,'new_training_steps':0,'limits':'Only three trained seeds per task. Nonzero returns are not interaction counts. Not a PIVOT comparison.','source_sha256':sha(__file__),'workers':6}
    (a.output/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    results=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        fs=[pool.submit(run,(task,seed,str(a.output))) for task in plan['tasks'] for seed in plan['seeds']]
        for f in as_completed(fs):
            result=f.result();results.append(result);print(json.dumps(result),flush=True)
    summary={'status':'complete','purpose':plan['purpose'],'results':sorted(results,key=lambda r:(r['task'],r['seed'])),'new_training_steps':0,'total_evaluation_steps':sum(r['evaluation_steps'] for r in results),'limitations':plan['limits']}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
