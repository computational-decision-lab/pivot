"""Numerical implementation audit only: no native benchmark outcomes are used.

For the author's linear Gaussian latent-value model, one scalar query produces
posterior means mu + a*Z. The expected reduction in Bayes simple regret equals
E[max(mu+a*Z)]-max(mu). We integrate the upper envelope of affine functions,
providing a noise-free reference for its nested Monte Carlo acquisition.

This does not change the frozen method or claim a new acquisition algorithm.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from pivot.acquisition.pivot_voi import BayesianLinearDeltaPosterior, score_pivot_voi


def phi(x):
    return math.exp(-x*x/2)/math.sqrt(2*math.pi)


def cdf(x):
    return (1+math.erf(x/math.sqrt(2)))/2


def affine_gaussian_max(means, slopes):
    """Integrate max_i(means[i] + slopes[i]*Z), Z standard normal."""
    m=np.asarray(means,dtype=float);a=np.asarray(slopes,dtype=float)
    boundaries={-math.inf, math.inf}
    for i in range(len(m)):
        for j in range(i):
            if a[i]!=a[j]:
                boundaries.add(float((m[j]-m[i])/(a[i]-a[j])))
    boundaries=sorted(boundaries);answer=0.
    for left,right in zip(boundaries,boundaries[1:]):
        if left==-math.inf and right==math.inf:mid=0.
        elif left==-math.inf:mid=right-max(1.,abs(right))
        elif right==math.inf:mid=left+max(1.,abs(left))
        else:mid=(left+right)/2
        j=int(np.argmax(m+a*mid))
        answer+=m[j]*(cdf(right)-cdf(left))+a[j]*(phi(left)-phi(right))
    return float(answer)


def exact_evsi(rows,post):
    X=np.array([r['features'] for r in rows]);mu=post.predict(X)+[r['delta_proxy'] for r in rows]
    K=X@post.covariance@X.T
    return {r['transition_id']:max(0.,affine_gaussian_max(mu,K[:,j]/math.sqrt(K[j,j]+post.noise_variance))-max(mu))
            for j,r in enumerate(rows)}


def checks():
    # Closed-form two-arm E[max(0, mu+sd*Z)], including no-information limits.
    for mu in [-2.,-.1,0.,.5,2.]:
        for sd in [.1,1.,3.]:
            expected=sd*phi(mu/sd)+mu*cdf(mu/sd)
            assert abs(affine_gaussian_max([0.,mu],[0.,sd])-expected)<1e-11
    assert affine_gaussian_max([1.,2.],[0.,0.])==2.


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    checks()
    rows=[{'transition_id':str(i),'features':[x,abs(x)],'delta_proxy':.2*x,'hf_query_cost':1.}
          for i,x in enumerate([-1.5,-1.1,-.7,-.3,.3,.7,1.1,1.5])]
    rows.append({'transition_id':'zero_information','features':[0.,0.],'delta_proxy':0.,'hf_query_cost':1.})
    post=BayesianLinearDeltaPosterior(mean=np.array([-.15,-.05]),covariance=np.array([[.5,.1],[.1,.3]]),noise_variance=2.)
    exact=exact_evsi(rows,post);records=[]
    for seed in range(32):
        scores=score_pivot_voi(rows,post,seed=seed,fantasies=64,posterior_samples=256)
        records.append({'seed':seed,'monte_carlo_evsi':{r['transition_id']:r['evsi'] for r in scores},'top_query':scores[0]['transition_id']})
    summary={}
    for r in rows:
        j=r['transition_id'];v=np.array([x['monte_carlo_evsi'][j] for x in records]);truth=exact[j]
        summary[j]={'exact_latent_model_evsi':truth,'mc_mean':float(v.mean()),'mc_sd':float(v.std(ddof=1)),
                    'rmse':float(np.sqrt(np.mean((v-truth)**2))),'positive_fraction':float(np.mean(v>0))}
    # Counterexample to identifying raw-query replacement with the posterior
    # decision rule used by acquisition fantasies. Both are legal estimators,
    # but they induce different decisions under noisy observations.
    example=BayesianLinearDeltaPosterior(mean=np.array([0.]),covariance=np.array([[.01]]),noise_variance=1.)
    corrected=.1+example.condition([1.],-.2).predict(np.array([[1.]]))[0]
    result={'scope':'Synthetic numerical and code-path audit. Zero new native episodes; no formal test outcomes used; frozen experiment unchanged.',
            'identity':'EVSI_j = E_Z[max_i(mu_i + K_ij/sqrt(K_jj+noise_variance)*Z)] - max_i(mu_i), for the latent Gaussian posterior-mean decision rule.',
            'reference_checks':'15 two-arm closed-form checks and a no-information limit passed',
            'fixture':{'rows':rows,'posterior_mean':post.mean.tolist(),'posterior_covariance':post.covariance.tolist(),'noise_variance':post.noise_variance},
            'configuration':{'repetitions':32,'fantasies':64,'posterior_samples':256},'per_query':summary,'raw_records':records,
            'estimator_mismatch_counterexample':{'prior_latent_variance':.01,'observation_variance':1.,'proxy_delta':.1,'observed_correction':-.2,'observed_delta':-.1,
                'posterior_mean_delta':float(corrected),'posterior_mean_choice':'candidate','raw_query_replacement_choice':'incumbent',
                'implication':'The frozen extension uses author posterior-mean fantasies but retains author-style raw measured values after queries. Thus its numerical EVSI is not exactly the value of information for its executed estimator. This is an implementation boundary, not a native performance finding.'}}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'zero_information_query':summary['zero_information'],'estimator_mismatch_counterexample':result['estimator_mismatch_counterexample']},indent=2))


if __name__=='__main__':main()
