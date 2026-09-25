"""Real-environment engineering smoke, disjoint from the frozen main study."""
import json
import os
from pathlib import Path
import subprocess
from run_melting_hierarchical_batch import ROOT,PY,MODEL


def main():
    out=ROOT/'melting_recovery_20260915/hierarchical_smoke'
    out.mkdir(exist_ok=False)
    (out/'calibration').mkdir()
    env=os.environ.copy()
    env.update(PYTHONPATH=':'.join(['/root/autodl-tmp/pivot_author_20260915/colin_pivot/src',
              '/root/autodl-tmp/pivot_author_20260915/colin_pivot',str(ROOT),
              '/root/autodl-tmp/colin_melting/vendor/meltingpot_source','/root/autodl-tmp/colin_melting']),
               CUDA_VISIBLE_DEVICES='',TF_CPP_MIN_LOG_LEVEL='3',TF_ENABLE_ONEDNN_OPTS='0')
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS']:
        env[k]='1'
    states=[]
    for phase,seed,target in [('calibration',39000,out/'calibration/seed_39000'),('test',39001,out/'test_39001')]:
        cmd=[PY,str(ROOT/'melting_hierarchical_e5.py'),'--phase',phase,'--seed',str(seed),'--output',str(target),
             '--model-path',str(MODEL),'--crossbench-root','/root/autodl-tmp/colin_melting',
             '--k','2','--budgets','0,1,2','--horizon','1000','--candidate-train-episodes','1',
             '--response-train-episodes','1','--eval-episodes','1','--proxy-episodes','1',
             '--learning-rate','0.5','--reward-scale','100','--reward-clip','2',
             '--max-seconds','500','--max-env-steps','30000']
        if phase=='test':cmd+=['--calibration-dir',str(out/'calibration')]
        with (out/f'{phase}.log').open('w') as f:
            p=subprocess.run(cmd,env=env,stdout=f,stderr=f,timeout=540)
        states.append({'phase':phase,'returncode':p.returncode,'output':str(target)})
        (out/'status.json').write_text(json.dumps(states,indent=2))
        print(json.dumps(states[-1]),flush=True)
        if p.returncode:raise SystemExit(p.returncode)


if __name__=='__main__':main()
