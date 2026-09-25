"""Synthetic correctness checks, never benchmark evidence."""
import copy, unittest
import numpy as np
from pivot.acquisition.pivot_voi import BayesianLinearDeltaPosterior
from sequential_pivot_extension import run,METHODS

class NoAudit(dict):
    def __getitem__(self,k):
        if k in ('audit','delta_true'):raise AssertionError('Hidden audit accessed')
        return super().__getitem__(k)

class SelectionContract(unittest.TestCase):
    def setUp(self):
        self.posterior=BayesianLinearDeltaPosterior(noise_variance=.2).fit(np.eye(2),np.zeros(2))
        self.rows=[NoAudit(transition_id=str(i),delta_proxy=.1*(i+1),features=f,hf_query_cost=3.,audit=object(),delta_true=object())
                   for i,f in enumerate([[1,0],[0,1],[1,1]])]
    def test_full_budget_queries_once_and_updates_real_posterior(self):
        for method in METHODS[:4]:
            calls=[]
            def query(row):calls.append(row['transition_id']);return {'delta':[-3,2,.5][int(row['transition_id'])],'split':'selection','hf_query_cost':3}
            result=run(self.rows,self.posterior,query,method=method,budget=3,seed=9,stop=False,fantasies=8,posterior_samples=32)
            self.assertEqual(result['selected_id'],'1');self.assertEqual(set(calls),{'0','1','2'})
            self.assertEqual(len(calls),3);self.assertEqual(result['charged_cost'],9)
            self.assertEqual([r['posterior_observations_after'] for r in result['steps']],[3,4,5])
            self.assertEqual(self.posterior.n_observations,2)
    def test_zero_budget_and_incumbent(self):
        rows=[dict(r,delta_proxy=-1.) for r in self.rows]
        def forbidden(_):raise AssertionError('Unpaid query')
        for method in METHODS:
            r=run(rows,self.posterior,forbidden,method=method,budget=0,seed=1)
            self.assertEqual(r['selected_id'],'incumbent');self.assertEqual(r['charged_cost'],0)
    def test_real_query_changes_unqueried_estimates(self):
        def query(_):return -100.
        r=run(self.rows,self.posterior,query,method='uniform_random_matched',budget=1,seed=9)
        queried=set(r['queried_ids']);baseline={x['transition_id']:x['delta_proxy'] for x in self.rows}
        self.assertTrue(any(abs(v-baseline[k])>1 for k,v in r['estimates'].items() if k not in queried and k!='incumbent'))
    def test_audit_response_rejected(self):
        with self.assertRaises(ValueError):run(self.rows,self.posterior,lambda _:{'delta':1,'split':'audit'},method='uniform_random_matched',budget=1,seed=9)
    def test_wrong_cost_rejected(self):
        with self.assertRaises(ValueError):run(self.rows,self.posterior,lambda _:{'delta':1,'hf_query_cost':0},method='uniform_random_matched',budget=1,seed=9)
    def test_confident_posterior_stops_before_spending(self):
        posterior=BayesianLinearDeltaPosterior(noise_variance=.2,mean=np.zeros(2),covariance=np.eye(2)*1e-12)
        rows=[dict(r,delta_proxy=float(i*10)) for i,r in enumerate(self.rows)]
        def forbidden(_):raise AssertionError('Confident posterior should stop')
        r=run(rows,posterior,forbidden,method='pivot_sequential',budget=3,seed=7,fantasies=8,posterior_samples=32)
        self.assertEqual(r['selected_id'],'2');self.assertEqual(r['charged_cost'],0)

if __name__=='__main__':unittest.main()
