"""Reanalyse frozen native evidence; no simulator or training is executed.

The new short/long contrast is retrospective. It never grants a prospective
mechanism pass. Independent A/B blocks provide a cross-product diagnostic
for squared gaps without the positive noise bias of squaring one estimate.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,re,tarfile
from pathlib import Path
import numpy as np

def sha(b): return hashlib.sha256(b).hexdigest()
def ci(x):
    x=np.asarray(x,dtype=float)
    assert x.ndim==1 and len(x)>1 and np.isfinite(x).all()
    draws=np.random.default_rng(20260917).integers(len(x),size=(10000,len(x)))
    return {'mean':float(x.mean()),'ci95':np.quantile(x[draws].mean(axis=1),[.025,.975]).tolist(),'n_seeds':len(x)}
def calc(rows):
    lookup={(r['target_probability'],r['training_episodes'],r['eval_block']):r for r in rows}
    result={}
    for h in (0,4,32):
        gaps=[];prod=[];own=[]
        for p in (.25,.75):
            gb=[]
            for b in ('A','B'):
                r=lookup[p,h,b];old=lookup[.5,h,b]
                dv=r['frozen_focal_return']-old['frozen_focal_return']
                ds=r['focal_return']-old['focal_return']
                gb.append(ds-dv);own.append(r['response_own_gain'])
                result[f'gap_{p}_{h}_{b}']=ds-dv
            gaps.append(abs(np.mean(gb)));prod.append(gb[0]*gb[1])
            result[f'signed_gap_{p}_{h}']=float(np.mean(gb))
        result[f'absolute_gap_{h}']=float(np.mean(gaps))
        result[f'cross_block_gap_squared_{h}']=float(np.mean(prod))
        result[f'responder_gain_{h}']=float(np.mean(own))
    for name in ('absolute_gap','cross_block_gap_squared','responder_gain','signed_gap_0.25','signed_gap_0.75'):
        result[f'{name}_long_minus_short']=result[f'{name}_32']-result[f'{name}_4']
    assert result['absolute_gap_0']==0 and result['cross_block_gap_squared_0']==0
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--published-csv',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with a.published_csv.open(encoding='utf-8-sig') as f:
        published={(int(r['seed']),float(r['target_probability']),int(r['training_episodes'])):float(r['response_effect']) for r in csv.DictReader(f)}
    allrows=[];checks=[]
    with tarfile.open(a.archive,'r:gz') as t:
        members={m.name:m for m in t.getmembers()}
        def read(name):
            m=members[name];assert m.isfile() and m.size<1000000
            return t.extractfile(m).read()
        for seed in range(42000,42012):
            root=f'melting_response_sprint_20260916/seed_{seed}'
            raw=read(root+'/results.json');summary=json.loads(read(root+'/summary.json'))
            assert summary['complete'] and summary['results_sha256']==sha(raw)
            rows=json.loads(raw);assert len(rows)==24
            r=calc(rows)
            for prob in (.25,.75):
                for h in (0,4,32):
                    assert math.isclose(r[f'signed_gap_{prob}_{h}'],published[seed,prob,h],abs_tol=1e-9)
            allrows.append({'seed':seed,**r})
            checks.append({'seed':seed,'results_sha256':sha(raw),'summary_complete':True,'published_csv_matches':True})
    stats={k:ci([r[k] for r in allrows]) for k in allrows[0] if k!='seed' and not k.startswith('gap_')}
    report={'status':'retrospective_analysis_complete','native_episodes_run':0,'source_archive':str(a.archive.resolve()),
            'source_archive_sha256':sha(a.archive.read_bytes()),'short_adaptation_episodes':4,'long_adaptation_episodes':32,
            'statistics':stats,'input_checks':checks,
            'interpretation':'Development evidence only; protocol chosen after these data were collected. Not a new seed cohort and not a PIVOT comparison.',
            'gate':'NOT_GRANTED: native replay discrepancy unresolved; prospective mechanism gate not run.',
            'noise_note':'Absolute gap of noisy means is upward biased. A/B cross-product estimates squared conditional gap only when blocks are independent and unbiased given trained policies; negative estimates are retained.',
            'statistical_unit':'12 independent root seeds; average probes/blocks within seed, paired bootstrap 10000 draws; pointwise descriptive intervals'}
    (a.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (a.output/'seed_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    for key in ('absolute_gap_4','absolute_gap_32','absolute_gap_long_minus_short','cross_block_gap_squared_long_minus_short','responder_gain_long_minus_short'):
        print(key,json.dumps(stats[key]))
    print('Independent source and CSV checks: 12/12 PASS; new native episodes: 0')
if __name__=='__main__': main()
