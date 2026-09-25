"""Bounded native Melting Pot conditional measurement audit; no selector tuning."""
import argparse, concurrent.futures, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temp.replace(path)

def weights(p,q):
    return np.array([(1-p)*(1-q),(1-p)*q,p*(1-q),p*q])

def noise_metrics(matrix, p, q, qold, n=8):
    """Rows are independent replicate seeds; columns retain paired strata."""
    x=matrix@weights(.5,qold); y=matrix@weights(p,q)
    vx=float(np.var(x,ddof=1)); vy=float(np.var(y,ddof=1))
    cov=float(np.cov(x,y,ddof=1)[0,1]); paired=float(np.var(y-x,ddof=1))/n
    independent=(vx+vy)/n
    assert np.isclose(paired,(vx+vy-2*cov)/n,atol=1e-9)
    return dict(paired_variance=paired,independent_variance=independent,independent_equal_cost_variance=2*independent,covariance=cov,
                mean_delta=float(np.mean(y-x)), covariance_identity_error=abs(paired-(vx+vy-2*cov)/n))

def worker(a):
    import melting_hierarchical_e5_stratified as native
    from melting_reference_diagnostic import OfficialSpecialistRollout
    protocol=json.loads(a.protocol.read_text()); world=next(w for w in protocol['worlds'] if w['root']==a.root)
    out=a.output/f'root_{a.root}'; out.mkdir(parents=True,exist_ok=False)
    config=dict(task='melting_stag',seed=world['evaluation_seed'],horizon=2500,stratum_replicates=16,proxy_stratum_replicates=16)
    backend=OfficialSpecialistRollout(a.model,crossbench_root=a.crossbench)
    runtime_paths=[Path(__file__),a.protocol,Path(native.__file__)]
    write(out/'inputs.json',dict(world=world,backend=backend.metadata,sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime_paths}))
    episodes=native.Episodes(out,backend,config,3300,128*2500)
    try:
        matrices=[]
        for label in ['A','B']:
            write(out/'status.json',dict(status='native_evaluation',block=label))
            block=native.evaluation_block(episodes,('measurement',label),('conditional_noise_v1',label),out/f'block_{label}.json',{'response_world':world['training_seal']})
            columns=[]
            for pair in [(0,0),(0,1),(1,0),(1,1)]:
                rows=[r for r in block['rows'] if tuple(r['sampled_specialists'])==pair]
                assert len(rows)==16
                columns.append([r['focal_return'] for r in rows])
            matrices.append(np.array(columns).T)
        matrix=np.concatenate(matrices); records=[]; covariances={}
        for h in [4,32]:
            q=world['training_seal']['response_probabilities']; qold=q['incumbent'][str(h)]
            deltas=[]
            for f in world['features']:
                j=f['id']; result=noise_metrics(matrix,f['p'],q[j][str(h)],qold)
                result.update(candidate=j,adaptation=h,root=a.root,author_observation_variance=world['posterior'][str(h)]['noise_variance'])
                result['block_means']=[float(np.mean(m@(weights(f['p'],q[j][str(h)])-weights(.5,qold)))) for m in matrices]
                records.append(result); deltas.append(matrix@(weights(f['p'],q[j][str(h)])-weights(.5,qold)))
            covariances[str(h)]=(np.cov(np.array(deltas),ddof=1)/8).tolist()
        assert len(episodes.costs)==128 and all(r['complete'] for r in episodes.costs.values())
        write(out/'summary.json',dict(status='complete',root=a.root,records=records,covariance_at_eight_replicates=covariances,native_episodes=len(episodes.costs),native_frames=episodes.frames))
        write(out/'status.json',dict(status='complete'))
    except Exception as exc:
        write(out/'status.json',dict(status='failed',error=repr(exc))); raise
    finally: backend.close()

def analyze(out):
    import csv
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summaries=[json.loads(p.read_text()) for p in sorted(out.glob('root_*/summary.json'))]
    assert len(summaries)==8
    rows=[r for s in summaries for r in s['records']]
    with (out/'noise_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    report={'scope':'Conditional measurement diagnostic on existing worlds; not confirmation of PIVOT advantage','native_episodes':sum(s['native_episodes'] for s in summaries),'native_frames':sum(s['native_frames'] for s in summaries),'by_adaptation':{}}
    for h in [4,32]:
        rs=[r for r in rows if r['adaptation']==h]
        report['by_adaptation'][str(h)]={k:float(np.mean([r[k] for r in rs])) for k in ['paired_variance','independent_equal_cost_variance','author_observation_variance']}
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for axis,h in zip(axes,[4,32]):
        rs=[r for r in rows if r['adaptation']==h]
        for x,key in enumerate(['paired_variance','independent_equal_cost_variance','author_observation_variance']):
            points=[np.mean([r[key] for r in rs if r['root']==root]) for root in range(44000,44008)]
            axis.scatter(np.full(8,x),points,alpha=.65);axis.plot([x-.15,x+.15],[np.mean(points)]*2,color='black')
        axis.set_xticks(range(3),['Paired HF','Independent HF\n(equal cost)','Author noise']);axis.set_title(f'Adaptation {h}');axis.set_ylabel('Variance of 8-replicate mean');axis.set_ylim(bottom=0)
    fig.suptitle('Melting Pot: conditional measurement audit (8 existing response worlds)');fig.tight_layout();fig.savefig(out/'measurement_noise.png',dpi=180);fig.savefig(out/'measurement_noise.pdf');plt.close(fig)
    write(out/'summary.json',report)

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--crossbench',type=Path,required=True);p.add_argument('--root',type=int);a=p.parse_args()
    if a.root is not None:return worker(a)
    a.output.mkdir(parents=True,exist_ok=False);protocol=json.loads(a.protocol.read_text());write(a.output/'protocol.json',protocol)
    source=Path(__file__);write(a.output/'source_seal.json',{'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256(a.protocol.read_bytes()).hexdigest()})
    def launch(world):
        cmd=[sys.executable,str(source),'--protocol',str(a.protocol),'--output',str(a.output),'--model',str(a.model),'--crossbench',str(a.crossbench),'--root',str(world['root'])]
        with (a.output/f"root_{world['root']}.log").open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=3450)
        return {'root':world['root'],'returncode':result.returncode}
    results=[];write(a.output/'status.json',dict(status='running',jobs=results))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for future in concurrent.futures.as_completed([pool.submit(launch,w) for w in protocol['worlds']]):
            results.append(future.result());write(a.output/'status.json',dict(status='running',jobs=results))
    if any(r['returncode'] for r in results):
        write(a.output/'status.json',dict(status='failed',jobs=results));raise SystemExit(2)
    analyze(a.output);write(a.output/'status.json',dict(status='complete',jobs=results))

if __name__=='__main__':main()
