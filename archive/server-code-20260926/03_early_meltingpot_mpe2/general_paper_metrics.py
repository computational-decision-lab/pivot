"""Descriptive paper-aligned metrics; no change to choices or registered contrasts.
Audit sample means estimate deployment deltas, so metrics are noisy plug-ins,
not claims that latent true signs or the best candidate are known.
"""
import argparse,json
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def bootstrap_mean(x):
    x=np.array(x,float);rng=np.random.default_rng(20260915);b=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    return {'mean':float(x.mean()),'ci95':np.quantile(b,[.025,.975]).tolist(),'n_training_seeds':len(x)}
def cluster_ratio(num,den):
    num=np.array(num,float);den=np.array(den,float)
    if den.sum()==0:return {'rate':None,'ci95':None,'numerator':0,'denominator':0}
    rng=np.random.default_rng(20260915);i=rng.integers(len(num),size=(10000,len(num)));a=num[i].sum(1);b=den[i].sum(1);valid=b>0
    return {'rate':float(num.sum()/den.sum()),'ci95':np.quantile(a[valid]/b[valid],[.025,.975]).tolist(),'numerator':int(num.sum()),'denominator':int(den.sum()),'bootstrap_nonempty_draws':int(valid.sum()),'n_training_seed_clusters':len(num)}
def metrics(rows):
    result={}
    for name in ['proxy','corrected']:
        errors=[];signs=[];comparable=[];bad=[];positive=[]
        for row in rows:
            p=np.array(row[name]);true=np.array(row['estimated_deployment_gain']);errors.append(float(np.mean(np.abs(p-true))))
            mask=(np.abs(p)>1e-9)&(np.abs(true)>1e-9);signs.append(int(((p>0)==(true>0))[mask].sum()));comparable.append(int(mask.sum()))
            m=p>1e-9;positive.append(int(m.sum()));bad.append(int((m&(true < -1e-9)).sum()))
        result[name]={'estimated_IDE':bootstrap_mean(errors),'observed_ISC':cluster_ratio(signs,comparable),'observed_IRR':cluster_ratio(bad,positive)}
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();b=a.base;tables={};panels=[]
    assert read(b/'fresh_mpe_general/status.json')['state']=='complete'
    assert read(b/'melting_general_long/status.json')['state']=='complete'
    for task in ['push','adversary']:
        root=b/'fresh_mpe_general';cal=read(root/task/'calibration.json');rows=[]
        for row in read(root/task/'candidate_audit.json'):
            f=np.array(read(root/task/f"seed_{row['seed']}"/'features.json'));z=np.c_[np.ones(4),(f[:,1:]-cal['feature_mean'])/cal['feature_scale']]
            r={'platform':'MPE2','task':task,'seed':row['seed'],'response_steps':8192,'proxy':row['proxy'],'corrected':(f[:,0]+z@np.array(cal['mean'])).tolist(),'estimated_deployment_gain':np.array(row['audit_replicates']).mean(1).tolist()}
            rows.append(r)
        tables['MPE2/'+task]=metrics(rows);panels.extend(rows)
    root=Path('/root/autodl-tmp/colin_melting/runs/melting_cross_main')
    for task in ['melting_pd','melting_stag']:
        cal=read(b/'author_melting'/(task+'_posterior.json'))
        for steps in [32768,131072]:
            rows=[]
            for row in read(b/'melting_general_long/candidate_audit.json'):
                if row['task']!=task or row['response_steps']!=steps:continue
                f=np.array(read(root/task/'test'/f"seed_{row['seed']}"/'features.json'));z=np.c_[np.ones(4),(f[:,1:]-cal['feature_mean'])/cal['feature_scale']]
                r={'platform':'Melting Pot','task':task,'seed':row['seed'],'response_steps':steps,'proxy':row['proxy'],'corrected':(f[:,0]+z@np.array(cal['mean'])).tolist(),'estimated_deployment_gain':np.array(row['audit_replicates']).mean(1).tolist()};rows.append(r)
            tables[f'MeltingPot/{task}/{steps}']=metrics(rows);panels.extend(rows)
    result={'status':'complete','purpose':'Additional descriptive metrics aligned with paper definitions; not additional prespecified hypothesis tests','tables':tables,'limitations':['Metrics use noisy independent audit means, not known latent deployment values.','Ratio intervals resample whole training-seed clusters; zero-denominator rates are undefined.','Correction model is fitted for the short response horizon; long Melting Pot evaluation is an OOD response-strength stress test.','No full-trajectory CISR claim from these single-transition panels.']}
    (b/'general_paper_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(b/'general_paper_metric_panels.json').write_text(json.dumps(panels,indent=2,allow_nan=False)+'\n');print('Paper-aligned descriptive metrics saved')
if __name__=='__main__':main()
