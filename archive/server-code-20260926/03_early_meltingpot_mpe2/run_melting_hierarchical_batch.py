"""Execute the frozen, time-bounded hierarchical Melting Pot E5 extension."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/root/autodl-tmp/pivot_author_20260915/colin_pivot_cloud')
PY = '/root/autodl-tmp/benchmark_extensions/melting_env/bin/python'
MODEL = ROOT/'melting_recovery_20260915/reference_smoke/reference_assets/assets/saved_models/stag_hunt_in_the_matrix__repeated/puppet_1'
DEADLINE = dt.datetime.fromisoformat('2026-09-15T16:37:25+00:00').timestamp()


def write(path, data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def command(phase, seed, output, calibration):
    return [PY, str(ROOT/'melting_hierarchical_e5_stratified.py'), '--phase',phase,
            '--task','melting_stag','--seed',str(seed),'--output',str(output),
            '--model-path',str(MODEL),'--crossbench-root','/root/autodl-tmp/colin_melting',
            '--k','4','--budgets','0,1,2,4','--candidate-train-episodes','4',
            '--response-train-episodes','4',
            '--reward-scale','100','--reward-clip','2','--learning-rate','0.5',
            '--old-logit','0','--opponent-logit','0',
            '--stratified-replicates','2','--proxy-stratified-replicates','2',
            '--max-seconds',str(max(1,int(DEADLINE-time.time()))),
            '--max-env-steps',str((88 if phase=='calibration' else 152)*2500)] + (
                ['--calibration-dir',str(calibration)] if phase=='test' else [])


def execute(phase,seed,output,calibration):
    env=os.environ.copy()
    env.update(PYTHONPATH=':'.join(['/root/autodl-tmp/pivot_author_20260915/colin_pivot/src',
             '/root/autodl-tmp/pivot_author_20260915/colin_pivot',str(ROOT),
             '/root/autodl-tmp/colin_melting/vendor/meltingpot_source','/root/autodl-tmp/colin_melting']),
             CUDA_VISIBLE_DEVICES='',TF_CPP_MIN_LOG_LEVEL='3',TF_ENABLE_ONEDNN_OPTS='0')
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                'TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','NUMEXPR_NUM_THREADS']:
        env[key]='1'
    cmd=command(phase,seed,output,calibration)
    started=dt.datetime.now(dt.timezone.utc).isoformat(); begin=time.monotonic()
    with output.with_suffix('.log').open('w') as f:
        p=subprocess.Popen(cmd,env=env,cwd=str(ROOT),stdout=f,stderr=f,stdin=subprocess.DEVNULL,start_new_session=True)
        try:
            code=p.wait(timeout=max(1,DEADLINE-time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            code=-1
    return {'phase':phase,'seed':seed,'returncode':code,'seconds':time.monotonic()-begin,
            'started_at':started,'finished_at':dt.datetime.now(dt.timezone.utc).isoformat(),
            'output':str(output),'command':cmd}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    out.mkdir(parents=True,exist_ok=False)
    for name in ['calibration','test']:(out/name).mkdir()
    script_names=['melting_hierarchical_e5_stratified.py','melting_hierarchical_e5.py','melting_reference_diagnostic.py',
                  'melting_behavior_diagnostic.py','melting_e5_adapter.py',Path(__file__).name]
    protocol={'study':'Melting Pot repeated stag hunt: learned episodic skill mixtures, author E5C selection',
              'registered_at':dt.datetime.now(dt.timezone.utc).isoformat(),
              'task':'melting_stag','candidate_count':4,'budgets':[0,1,2,4],
              'calibration_seeds':list(range(31000,31008)),'test_seeds':list(range(32000,32020)),
              'candidate_train_episodes':4,'response_train_episodes':4,'proxy_evaluation_total_episodes':8,'pair_evaluation_total_episodes':8,
              'learning_rate':0.5,'reward_scale':100,'reward_clip':2,'old_logit':0,'opponent_logit':0,
              'calibration_episodes_per_panel':88,'test_episodes_per_panel':152,'total_planned_episodes':3744,
              'calibration_workers':8,'test_workers':20,'native_max_frames':2500,
              'compute_deadline_utc':'2026-09-15T16:37:25+00:00','shutdown_deadline_utc':'2026-09-15T16:47:25+00:00',
              'query_unit':'16 native-terminated episodes: 4 response-training episodes per branch plus 2 fresh evaluation episodes for each of four fixed skill pairs',
              'evaluation_design':'stratified Monte Carlo over four skill pairs, analytically weighted by frozen episode-mixture probabilities',
              'stratified_replicates_per_skill_pair':2,'proxy_stratified_replicates_per_skill_pair':2,
              'evaluation_note':'conditional returns come from newly executed real Melting Pot episodes; training never reads evaluation blocks; audit independently reruns response learning and all four evaluation strata',
              'mechanism_note':'frozen/adapted policy expectations are reweighted on the same independent audit block; estimates correlated by design; no extra physical episodes counted',
              'actual_environment_frames_reported_separately':True,
              'selection_freeze_scope':'all methods and budgets within each test panel, before that panel audit',
              'independence_unit':'fresh high-level learning/environment seed, conditional on one frozen official low-level skill network',
              'comparison_scope':'single task, one-dimensional episodic stag/hare mixture, single update round',
              'primary_contrasts':['PIVOT versus Proxy Only at budget 2 (extra validation cost)',
                                   'PIVOT versus Random-HF at budget 2 (same episode query budget)'],
              'secondary':'all budget frontiers; paired frozen/adaptive gains; responder own-return change; audit-block reweighting at the five saved response-learning checkpoints (correlated descriptive curve, no additional experiments)',
              'statistics':'paired training-seed means and bootstrap 95% intervals; main two tests Holm adjusted; other analyses descriptive',
              'outcome_independent_stop_rule':'fixed samples; failures and timeouts retained; no seed replacement or outcome-based extra training',
              'sources_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in script_names},
              'hardware_scope':'existing 25-vCPU,92-GiB instance; no GPU/LLM API inference used in this protocol'}
    write(out/'protocol.json',protocol)
    state={'status':'running','jobs':[]}
    write(out/'status.json',state)
    for phase,seeds,workers in [('calibration',protocol['calibration_seeds'],8),('test',protocol['test_seeds'],20)]:
        state['phase']=phase;write(out/'status.json',state)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(execute,phase,s,out/phase/f'seed_{s}',out/'calibration') for s in seeds]
            for f in as_completed(futures):
                row=f.result();state['jobs'].append(row);write(out/'status.json',state)
                print(json.dumps(row),flush=True)
        failed=[j for j in state['jobs'] if j['phase']==phase and j['returncode']!=0]
        if failed:
            state['status']='failed_or_budget_stopped';write(out/'status.json',state)
            raise SystemExit(1)
        expected_episodes = 88 if phase == 'calibration' else 152
        for seed in seeds:
            panel = out/phase/f'seed_{seed}'
            summary = json.loads((panel/'summary.json').read_text())
            if summary['status'] != 'complete' or summary['seed'] != seed:
                raise RuntimeError('Panel summary does not confirm planned completion')
            if summary['completed_physical_episodes'] != expected_episodes or summary['incomplete_physical_attempts']:
                raise RuntimeError('Panel physical episode count differs from its frozen budget')
            if phase == 'test':
                info=json.loads((panel/'protocol.json').read_text())
                if sorted(info['calibration_seeds']) != protocol['calibration_seeds']:
                    raise RuntimeError('Test did not use all eight planned calibration seeds')
                if not summary['audit_isolated'] or summary['n_decisions'] != 17:
                    raise RuntimeError('Test lacks its complete isolated E5 decisions')
    state['status']='complete';state['finished_at']=dt.datetime.now(dt.timezone.utc).isoformat()
    write(out/'status.json',state)


if __name__=='__main__':main()
