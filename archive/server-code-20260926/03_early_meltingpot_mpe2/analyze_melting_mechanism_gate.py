"""Verify every episode/learning update, then evaluate the frozen response gate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from verify_melting_mechanism import Evidence,inspect_panel,sha,csv_rows
from audit_melting_short_long import calc,ci

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=Path,required=True);p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.batch/'protocol.json').read_text());seeds=protocol['mechanism']['seeds']
    assert len(seeds)==len(set(seeds))==12
    gate_ref=json.loads((a.batch/'replay_gate.json').read_text())
    assert gate_ref['status']=='PASS' and gate_ref['native_episodes']>=8
    assert json.loads((a.batch/'status.json').read_text())['runtime_unchanged']
    assert gate_ref['runtime_file_hashes']==protocol['runtime_file_hashes']
    evidence=Evidence(a.batch);panels=[];allcontexts=set()
    for seed in seeds:
        panel=inspect_panel(evidence,a.batch/f'seed_{seed}',seed,protocol,a.sources)
        assert not panel['contexts'] & allcontexts,'Root seeds share evaluation/training random contexts'
        allcontexts.update(panel['contexts']);panels.append(panel)
    values=[{'seed':x['seed'],**calc(x['rows'])} for x in panels]
    stats={k:ci([r[k] for r in values]) for k in values[0] if k!='seed' and not k.startswith('gap_')}
    response_pass=stats['responder_gain_long_minus_short']['ci95'][0]>0
    gap_pass=stats['cross_block_gap_squared_long_minus_short']['ci95'][0]>1
    choice_rows=[]
    for panel in panels:
        for h in (4,32):
            r={(x['target_probability'],x['eval_block']):x for x in panel['rows'] if x['training_episodes']==h}
            probs=[.25,.5,.75]
            proxy=[np.mean([r[prob,b]['frozen_focal_return'] for b in ['A','B']]) for prob in probs]
            deploy=[np.mean([r[prob,b]['focal_return'] for b in ['A','B']]) for prob in probs]
            proxy_pick=int(np.argmax(proxy));deployment_pick=int(np.argmax(deploy))
            choice_rows.append({'seed':panel['seed'],'adaptation_episodes':h,'proxy_best_probe':probs[proxy_pick],
                               'deployment_best_probe':probs[deployment_pick],'different_point_estimate_winners':proxy_pick!=deployment_pick,
                               'noisy_in_sample_probe_regret':max(deploy)-deploy[proxy_pick],
                               'interpretation':'probe diagnostic only, neither exact regret nor formal candidate selection result'})
    report={'status':'PASS' if response_pass and gap_pass else 'NO_GO',
        'next_stage':'FREEZE_METHOD_PROTOCOL' if response_pass and gap_pass else 'OPENSPIEL_NO_EXTRA_MELTING_SEEDS',
        'response_quality_pass':response_pass,'gap_pass':gap_pass,'protocol_sha256':sha(a.batch/'protocol.json'),
        'source_hashes':protocol['source_hashes_before_replay_fix'],'verified_inputs':len(evidence.hashes),
        'seed_count':len(seeds),'statistics':stats,'accounting':{'native_episodes':sum(sum(x['counts'].values()) for x in panels),
        'training_episodes':sum(x['counts']['training'] for x in panels),'evaluation_episodes':sum(x['counts']['evaluation'] for x in panels),
        'native_frames':sum(sum(x['frames'].values()) for x in panels)},
        'scope':'One fixed official low-level network, reward-trained skill mixture, fixed probes; mechanism screen only, no PIVOT methods run.',
        'rule':'Both predeclared tests must pass; no seed extension. Failure is insufficient evidence at this fixed budget, not proof of no population effect.',
        'gap_caveat':'Absolute gap has measurement-noise bias. Cross-block squared gaps require independent unbiased A/B estimates conditional on frozen learned policies.'}
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'verified_input_hashes.json').write_text(json.dumps(evidence.hashes,indent=2)+'\n')
    csv_rows(a.output/'seed_results.csv',values);csv_rows(a.output/'probe_choice_diagnostics.csv',choice_rows)
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
