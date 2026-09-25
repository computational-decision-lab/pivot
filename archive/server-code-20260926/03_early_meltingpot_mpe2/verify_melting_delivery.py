"""Check downloaded evidence hashes and independently recompute primary contrasts."""
import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with tarfile.open(a.archive,'r:gz') as t:
        members={m.name:m for m in t.getmembers()}
        matches=[m for m in members.values() if m.name.endswith('/delivery_manifest.json')]
        assert len(matches)==1
        root=Path(matches[0].name).parent.as_posix()
        def read(name):
            member=members[root+'/'+name];assert member.isfile()
            return t.extractfile(member).read()
        manifest=json.loads(read('delivery_manifest.json'))
        for item in manifest['files']:
            value=read(item['path'])
            assert len(value)==item['bytes'] and hashlib.sha256(value).hexdigest()==item['sha256'],item['path']
        protocol_bytes=read('protocol.json');protocol=json.loads(protocol_bytes);summary=json.loads(read('analysis/summary.json'));status=json.loads(read('status.json'))
        digest=hashlib.sha256(protocol_bytes).hexdigest()
        assert digest==summary['protocol_sha256']==manifest['protocol_sha256']
        assert protocol['test_seeds']==list(range(44000,44030))
        assert status['status']=='complete' and status['frozen_inputs_unchanged']
        assert len(status['jobs'])==30 and all(x['status']=='complete' for x in status['jobs'])
        assert summary['accounting']['native_episodes']==manifest['native_episodes']==21120
        rows=list(csv.DictReader(io.StringIO(read('analysis/method_seed_results.csv').decode('utf-8-sig'))))
        full_query_checks=0
        for seed in protocol['test_seeds']:
            for h in [4,32]:
                matching=[r for r in rows if int(r['seed'])==seed and int(r['adaptation'])==h]
                reference=next(r for r in matching if r['method']=='all_hf_reference')
                for r in matching:
                    if int(r['hf_queries'])==8:
                        assert abs(float(r['audit_gain'])-float(reference['audit_gain']))<1e-10
                        full_query_checks+=1
        def value(seed,h,method):
            found=[r for r in rows if int(r['seed'])==seed and int(r['adaptation'])==h and r['method']==method and r['budget_cap']=='192']
            assert len(found)==1;return float(found[0]['audit_gain'])
        contrasts={'short_regret_reduction':[],'long_regret_reduction':[],'interaction':[]}
        for seed in protocol['test_seeds']:
            short=value(seed,4,protocol['primary_method'])-value(seed,4,protocol['primary_comparator'])
            long=value(seed,32,protocol['primary_method'])-value(seed,32,protocol['primary_comparator'])
            for k,v in [('short_regret_reduction',short),('long_regret_reduction',long),('interaction',long-short)]:contrasts[k].append(v)
        resampled=np.random.default_rng(20260917).integers(30,size=(10000,30));recomputed={}
        for key,values in contrasts.items():
            x=np.array(values);mean=float(x.mean());bounds=np.quantile(x[resampled].mean(axis=1),[.025,.975])
            assert np.allclose([mean,*bounds],[summary['primary_effects'][key]['mean'],*summary['primary_effects'][key]['ci95']],rtol=1e-12,atol=1e-12)
            recomputed[key]={'mean':mean,'ci95':bounds.tolist()}
    result={'status':'PASS','archive_sha256':hashlib.sha256(a.archive.read_bytes()).hexdigest(),'protocol_sha256':digest,
            'artifact_hashes_verified':len(manifest['files']),'native_episodes':21120,'independent_roots':30,'full_query_reference_matches':full_query_checks,
            'primary_contrasts_recomputed_from_csv':recomputed,
            'scope':'Local transport/hash verification plus independent aggregate recomputation. Native episode/reward/learning/seed reconstruction is recorded by the server verifier; this does not claim re-running the environment locally.'}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
