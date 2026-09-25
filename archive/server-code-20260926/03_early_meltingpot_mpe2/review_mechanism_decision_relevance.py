"""Discovery-only cross-fit choice diagnostic; not a PIVOT method comparison."""
import argparse,json
from pathlib import Path
import numpy as np
from audit_melting_short_long import ci
from verify_melting_mechanism import csv_rows,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.batch/'analysis/summary.json').read_text());assert report['seed_count']==12
    rows=[]
    for seed in range(43000,43012):
        path=a.batch/f'seed_{seed}/results.json';summary=json.loads((path.parent/'summary.json').read_text());assert sha(path)==summary['results_sha256']
        data=json.loads(path.read_text())
        for h in [4,32]:
            table={(r['target_probability'],r['eval_block']):r for r in data if r['training_episodes']==h}
            effects=[];switched=[]
            for selection,evaluation in [('A','B'),('B','A')]:
                ps=[.25,.5,.75]
                proxy=max(ps,key=lambda p:table[p,selection]['frozen_focal_return'])
                response_aware=max(ps,key=lambda p:table[p,selection]['focal_return'])
                effects.append(table[response_aware,evaluation]['focal_return']-table[proxy,evaluation]['focal_return'])
                switched.append(proxy!=response_aware)
            rows.append({'seed':seed,'adaptation':h,'crossfit_choice_gain':float(np.mean(effects)),'different_choices_fraction':float(np.mean(switched))})
    result={'scope':'Discovery-only choice opportunity, not PIVOT or exact oracle regret. Select on A and evaluate on B, then swap; average within root seed.',
            'created_after_primary_mechanism_protocol':True,'used_to_redefine_primary_mechanism_gate':False,
            'effects':{str(h):ci([r['crossfit_choice_gain'] for r in rows if r['adaptation']==h]) for h in [4,32]}}
    (a.batch/'analysis/decision_relevance.json').write_text(json.dumps(result,indent=2)+'\n');csv_rows(a.batch/'analysis/decision_relevance_by_seed.csv',rows);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
