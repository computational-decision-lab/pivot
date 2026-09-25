"""Explicit paper-loop extension; never silently substitutes for author E5C.

Uses the author's Bayesian differential posterior, EVSI computation and stop
criterion. Unlike E5C's batch allocation, real queries condition the posterior
and acquisition/stopping are recomputed. Uniform/Random HF, global integrated
variance reduction and a posterior-LUCB heuristic share the same estimator.
The LUCB heuristic is NOT claimed to be classical frequentist LUCB.
No audit labels are accepted or read. Native query accounting belongs to caller.
"""
from __future__ import annotations
import copy, math
from collections.abc import Mapping
import numpy as np
from pivot.acquisition.pivot_voi import score_pivot_voi, should_stop

VERSION='paper_loop_extension_v1_not_author_e5c'
METHODS=('pivot_sequential','uniform_random_matched','global_ivr_matched','posterior_lucb_matched','calibrated_no_hf','proxy_only')

def run(candidates,posterior,query,*,method,budget,seed,stop=True,delta=.05,eta=0.,fantasies=64,posterior_samples=256):
    if method not in METHODS:raise ValueError(method)
    if type(budget) is not int or budget<0:raise ValueError('budget')
    # Input whitelist prevents hidden query-bank or audit values entering policy.
    rows=[{'transition_id':str(r['transition_id']),'delta_proxy':float(r['delta_proxy']),
           'features':list(map(float,r['features'])),'hf_query_cost':float(r['hf_query_cost'])} for r in candidates]
    ids=[r['transition_id'] for r in rows]
    if not rows or len(set(ids))!=len(rows) or 'incumbent' in ids:raise ValueError('candidate identities')
    d=posterior.feature_dim
    for r in rows:
        if len(r['features'])!=d or not all(map(math.isfinite,r['features']+[r['delta_proxy'],r['hf_query_cost']])) or r['hf_query_cost']<=0:raise ValueError('candidate inputs')
    rows.append({'transition_id':'incumbent','delta_proxy':0.,'features':[0.]*d,'hf_query_cost':1.})
    rng=np.random.default_rng(seed);post=copy.deepcopy(posterior);observed={};steps=[];reason='budget_exhausted'
    def transformed():
        return [dict(r,delta_proxy=observed[r['transition_id']],features=[0.]*d) if r['transition_id'] in observed else dict(r) for r in rows]
    if method in ('calibrated_no_hf','proxy_only'):budget=0;reason='no_hf_method'
    for iteration in range(min(budget,len(candidates))):
        current=transformed();available=[i for i,r in enumerate(rows[:-1]) if r['transition_id'] not in observed]
        X=np.array([r['features'] for r in current]);proxies=np.array([r['delta_proxy'] for r in current]);means=post.predict(X)+proxies
        score_seed=int(rng.integers(0,2**31-1));details={}
        if method=='pivot_sequential':
            scored=score_pivot_voi(current,post,seed=score_seed,fantasies=fantasies,posterior_samples=posterior_samples)
            eligible=[r for r in scored if r['transition_id'] in {ids[i] for i in available}]
            top=eligible[0];do_stop,why=should_stop(selection_probability=float(top['selection_probability']),max_acquisition=float(top['acquisition']),delta=delta,eta=eta)
            if stop and do_stop:reason=why;steps.append({'iteration':iteration,'stop_reason':why,'scores':scored});break
            j=ids.index(top['transition_id']);details={'scores':scored}
        elif method=='uniform_random_matched':j=int(rng.choice(available))
        elif method=='global_ivr_matched':
            C=post.covariance; scores=[]
            for i in available:
                v=C@X[i];reduction=float(np.sum((X@v)**2)/(post.noise_variance+X[i]@v))
                scores.append((reduction/rows[i]['hf_query_cost'],i))
            j=max(scores,key=lambda x:(x[0],-x[1]))[1];details={'integrated_variance_reduction_per_cost':scores}
        elif method=='posterior_lucb_matched':
            sd=np.sqrt(post.predictive_variance(X));best=int(np.argmax(means))
            challenger=max((i for i in range(len(rows)) if i!=best),key=lambda i:(means[i]+1.96*sd[i],-i))
            pair=[i for i in [best,challenger] if i in available]
            if not pair:pair=available
            j=max(pair,key=lambda i:(sd[i]/rows[i]['hf_query_cost'],-i))
            details={'best':rows[best]['transition_id'],'challenger':rows[challenger]['transition_id'],'heuristic':'posterior best/challenger; sample the queryable member with larger sd per cost'}
        else:raise AssertionError(method)
        chosen=rows[j];result=query(dict(chosen,features=list(chosen['features'])))
        if isinstance(result,Mapping):
            if result.get('split',result.get('stream','selection'))!='selection':raise ValueError('query must be selection split')
            value=float(result['delta'])
            if not math.isclose(float(result.get('hf_query_cost',chosen['hf_query_cost'])),chosen['hf_query_cost']):raise ValueError('query cost differs from frozen cost')
        else:value=float(result)
        if not math.isfinite(value):raise ValueError('nonfinite query')
        before=post.n_observations
        post=post.condition(chosen['features'],value-chosen['delta_proxy'])
        observed[chosen['transition_id']]=value
        steps.append({'iteration':iteration,'query_id':chosen['transition_id'],'observed_delta':value,'charged_cost':chosen['hf_query_cost'],
                      'posterior_observations_before':before,'posterior_observations_after':post.n_observations,**details})
    current=transformed();X=np.array([r['features'] for r in current]);proxies=np.array([r['delta_proxy'] for r in current])
    estimates=proxies if method=='proxy_only' else post.predict(X)+proxies
    selected=int(np.argmax(estimates))
    return {'version':VERSION,'method':method,'selected_id':rows[selected]['transition_id'],'selected_estimate':float(estimates[selected]),
            'queried_ids':list(observed),'hf_queries':len(observed),'charged_cost':sum(r['hf_query_cost'] for r in rows if r['transition_id'] in observed),
            'stop_reason':reason,'steps':steps,'estimates':{r['transition_id']:float(v) for r,v in zip(rows,estimates)},
            'real_query_posterior_update':True,'audit_used':False,'incumbent_available':True}
