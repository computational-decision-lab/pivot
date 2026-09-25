"""Synthetic orchestration/isolation test, not environment evidence."""
import contextlib,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import melting_method_native_v3 as m

class FakeNative:
    metadata={'backend':'synthetic_contract_test_NOT_BENCHMARK'}
    def episode(self,matchup,env_seed,policy_seeds,deadline,allowance):
        a,b=[int(v=='stag') for v in matchup]
        return {'complete':True,'env_steps':1,'focal_return':3.+2*a+b,'response_return':2.+a+4*b,'event_parse_errors':[]},[]

class NativeContract(unittest.TestCase):
    def test_seal_cost_and_fixed_conditional_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);protocol=root/'protocol.json'
            protocol.write_text(json.dumps({'status':'FROZEN','candidate_probabilities':m.GRID,'test_seeds':[44000],'hf_episode_caps':[96,192,384]}))
            a=SimpleNamespace(output=root/'test',protocol=protocol,seed=44000,max_seconds=120,calibration=root)
            def calibration(*_):
                p=m.BayesianLinearDeltaPosterior(noise_variance=1.).fit(np.eye(2),np.zeros(2));return p,{'synthetic':True,'mean':p.mean.tolist(),'covariance':p.covariance.tolist(),'noise_variance':p.noise_variance}
            original=m.sequential
            def smaller(*args,**kw):return original(*args,**kw,fantasies=4,posterior_samples=16)
            with patch.object(m,'fit_calibration',calibration),patch.object(m,'sequential',smaller),contextlib.redirect_stdout(io.StringIO()):
                result=m.execute(a,FakeNative())
            self.assertEqual(result['actual_native_episodes'],704)
            rows=json.loads((a.output/'scored_decisions.json').read_text())
            self.assertTrue(rows)
            for r in rows:
                if r['budget_cap'] is not None:self.assertLessEqual(r['hf_episode_cost'],r['budget_cap'])
                self.assertGreaterEqual(r['noisy_audit_isr'],0)
                self.assertIn(r['selected_id'],[str(i) for i in range(8)]+['incumbent'])
            seeds=json.loads((a.output/'seeds.json').read_text())
            stages={name:{v for k,v in seeds.items() if json.loads(k)[:2]==[m.VERSION,name]} for name in ['response_training','proxy','selection','audit']}
            for a1 in stages:
                for b1 in stages:
                    if a1!=b1:self.assertFalse(stages[a1]&stages[b1])
            self.assertTrue(all(stages.values()))
            decisions=json.loads((a.output/'decisions_frozen.json').read_text());self.assertFalse(any('audit_gain' in r for r in decisions))
            m.v2.verify_seal(a.output)
            from verify_melting_method_v3 import inspect_panel
            verified=inspect_panel(a.output,m.v2.sha(protocol))
            self.assertEqual(verified['episodes'],704)
            path=a.output/'scored_decisions.json';tampered=json.loads(path.read_text());tampered[0]['audit_gain']+=1;path.write_text(json.dumps(tampered))
            with self.assertRaises(ValueError):inspect_panel(a.output,m.v2.sha(protocol))
if __name__=='__main__':unittest.main()
