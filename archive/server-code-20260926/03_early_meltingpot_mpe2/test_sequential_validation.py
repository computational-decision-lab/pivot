import numpy as np
from sequential_validation import condition,evsi,run,exchangeable_discrepancy


def test_condition_matches_joint_normal_and_preserves_input():
    m=np.array([1.,2.]);c=np.array([[2.,.6],[.6,1.]])
    a,b=condition(m,c,0,4.,.5)
    np.testing.assert_allclose(a,[3.4,2.72])
    np.testing.assert_allclose(b,[[.4,.12],[.12,.856]])
    np.testing.assert_allclose(c,[[2.,.6],[.6,1.]])
    assert np.linalg.eigvalsh(b).min()>0


def test_evsi_matches_analytic_two_action_case():
    # A query shifts only its own posterior mean by N(0, 1/2).
    actual=evsi([0.,0.],np.eye(2),1.,nodes=128)
    expected=1/(2*np.sqrt(np.pi))
    np.testing.assert_allclose(actual,[expected,expected],rtol=.004)


def test_common_uncertainty_cannot_change_ranking():
    np.testing.assert_allclose(evsi([1.,0.],np.ones((2,2))*5,1.),[0.,0.],atol=1e-12)


def test_actual_observation_updates_posterior_before_stopping():
    called=[]
    def q(j,r):called.append((j,r));return {'delta':10.,'total_env_steps':7}
    result=run([0.,0.],np.eye(2),.1,q,[7,7],4,173)
    assert len(called)==1 and result['query_cost']==7
    assert result['stop_reason']=='posterior_confidence'
    assert 0<result['estimates'][called[0][0]]<10 # posterior, not raw-label overwrite
    assert len(result['trace'])==2


def test_repeat_queries_use_distinct_replicates_and_pay_cost():
    called=[]
    def q(j,r):called.append((j,r));return {'delta':0.,'total_env_steps':3}
    result=run([0.,0.],np.eye(2),100.,q,[3,3],5,4,stop=False,strategy='uniform')
    assert called==[(0,0),(1,0),(0,1),(1,1),(0,2)]
    assert result['query_cost']==15


def test_discrepancy_projection_is_psd():
    error=np.array([[1.,1.,1.,1.],[-1.,-1.,-1.,-1.]])
    c,info=exchangeable_discrepancy(error,np.zeros((2,4,4)))
    np.testing.assert_allclose(c,np.ones((4,4)))
    assert info['contrast_direction_variance']==0


if __name__=='__main__':
    tests=[v for k,v in list(globals().items()) if k.startswith('test_')]
    for test in tests:test()
    print(f'{len(tests)} analytical and query-contract checks passed')
