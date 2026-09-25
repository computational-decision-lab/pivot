"""Read-only live progress; deliberately does not inspect partial method outcomes."""
import argparse
import collections
import datetime
import json
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--interval',type=float,default=55.);p.add_argument('--once',action='store_true');a=p.parse_args()
    assert a.interval>=30
    begin=time.monotonic()
    while time.monotonic()-begin<15000:
        try:
            status=json.loads((a.batch/'status.json').read_text())
            costs=[list(json.loads(x.read_text()).values()) for x in a.batch.glob('seed_*/episode_costs.json')]
            states=collections.Counter()
            for folder in sorted(a.batch.glob('seed_*')):
                if not folder.is_dir():continue
                declared=json.loads((folder/'status.json').read_text())['status']
                ledger=json.loads((folder/'episode_costs.json').read_text())
                present={r['stage'] for r in ledger.values()}
                phase=declared if declared in ['complete','failed'] else next((name for name in ['audit','selection','proxy','response_training'] if name in present),'initializing')
                states[phase]+=1
            snapshot={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cohort_status':status['status'],
                      'completed_native_episodes':sum(sum(r['complete'] for r in rows) for rows in costs),
                      'native_frames':sum(sum(r['env_steps'] for r in rows) for rows in costs),'stages':dict(states),
                      'analysis_exists':(a.batch/'analysis/summary.json').exists()}
            print(json.dumps(snapshot),flush=True)
            if a.once:return
            if status['status'] in ['incomplete','stopping_after_failure'] or (snapshot['analysis_exists'] and (a.batch/'analysis/primary_contrasts.pdf').exists()):return
        except (json.JSONDecodeError,FileNotFoundError):
            print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'snapshot_read_race_retry_next_interval'}),flush=True)
        time.sleep(a.interval)


if __name__=='__main__':main()
