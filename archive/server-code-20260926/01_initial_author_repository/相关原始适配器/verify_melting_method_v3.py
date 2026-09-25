"""Independent accounting, isolation and learning-update audit of a method panel."""
import json,math
from pathlib import Path
import numpy as np
from verify_melting_mechanism import Evidence,sha,digest,close,require,sigmoid,verify_block,weighted

def inspect_panel(folder,batch_protocol):
    folder=Path(folder);e=Evidence(folder);s=e.load(folder/'summary.json');p=e.load(folder/'protocol.json');c=p['config']
    require(s['status']=='complete' and e.load(folder/'status.json')['status']=='complete','unfinished panel')
    require(s['source_protocol_sha256']==p['batch_protocol_sha256']==batch_protocol,'wrong frozen batch')
    costs=e.load(folder/'episode_costs.json');registry=e.load(folder/'seeds.json')
    require(len(costs)==704 and len(set(registry.values()))==len(registry),'cost count or seed collision')
    native={};stages={};frames=0
    for rel,cost in costs.items():
        r=e.load(folder/rel);sig=r['signature'];key=sig['key'];template=sig['template']
        require(r['complete'] and not r.get('event_parse_errors') and cost['complete'],'invalid episode')
        require(r['episode_id']==digest(key)[:24] and r['train_seed']==s['seed'],'wrong identity')
        close(r['env_steps'],cost['env_steps'],'frame ledger');require(0<r['env_steps']<=2500,'frame limit');frames+=r['env_steps']
        require(registry[json.dumps([*template,'environment'])]==sig['environment_seed'],'environment seed')
        require([registry[json.dumps([*template,'policy',i])] for i in [0,1]]==sig['policy_seeds'],'policy seed')
        stages.setdefault(key[0],set()).add((sig['environment_seed'],*sig['policy_seeds']))
        if sig['sampling_mode']=='sampled_training':
            for i,role in enumerate(['focal','response']):
                u=float(np.random.default_rng(registry[json.dumps([*template,'mixture',i])]).random())
                close(u,sig['mixture_uniforms'][i],'mixture draw')
                require(r['sampled_specialists'][i]==int(u<sigmoid(sig[role+'_logit'])),'skill draw')
        native[r['episode_id']]=r
    close(frames,s['actual_native_frames'],'total frames')
    for a in stages:
        for b in stages:
            if a!=b:require(not stages[a]&stages[b],'stage random streams overlap')
    require({k:sum(v['stage']==k for v in costs.values()) for k in stages}=={'response_training':288,'proxy':32,'selection':256,'audit':128},'stage episode counts')
    seal=e.load(folder/'training_seal.json');require(seal['all_frozen_before_proxy'],'training not frozen')
    require(len(seal['history_hashes'])==9,'response target count')
    for rel,h in seal['history_hashes'].items():
        t=e.load(folder/rel,h);theta=0.;reward_sum=0.
        require(t['complete'] and len(t['history'])==32 and t['role']=='response','incomplete learning')
        for i,step in enumerate(t['history']):
            r=native[step['episode_id']];pr=sigmoid(theta);z=r['sampled_specialists'][1]
            scaled=float(np.clip(r['response_return']/100.,-2,2));baseline=reward_sum/i if i else 0.;grad=(scaled-baseline)*(z-pr)
            close(step['logit_before'],theta,'logit before');close(step['baseline_before'],baseline,'baseline');close(step['gradient'],grad,'gradient')
            theta=float(np.clip(theta+.5*grad,-8,8));close(step['logit_after'],theta,'learned logit');reward_sum+=scaled
        target=native[t['history'][0]['episode_id']]['signature']['key'][1]
        for horizon in [4,32]:close(seal['response_probabilities'][target][str(horizon)],sigmoid(t['history'][horizon-1]['logit_after']),'checkpoint')
    proxy=verify_block(e,folder/'proxy/common.json',native,8,c)
    selections={str(i):verify_block(e,folder/f'selection/{i}.json',native,8,c) for i in range(8)}
    audits=[verify_block(e,folder/f'audit/{b}.json',native,16,c) for b in ['A','B']]
    selection_seal=e.load(folder/'selection_seal.json')
    for field,name in [('decisions_sha256','decisions_frozen.json'),('candidate_sha256','candidates.json'),('features_sha256','features.json'),('posterior_sha256','posterior.json'),('training_seal_sha256','training_seal.json')]:
        require(sha(folder/name)==selection_seal[field],'selection input changed')
    decisions=e.load(folder/'decisions_frozen.json');scored=e.load(folder/'scored_decisions.json',s['scored_decisions_sha256'])
    grid=e.load(folder/'candidates.json')['probabilities'];q=seal['response_probabilities'];post=e.load(folder/'posterior.json')
    require(len(decisions)==len(scored),'decision row count')
    gains={}
    for h in [4,32]:
        gains[h]={'incumbent':0.}
        for i,prob in enumerate(grid):gains[h][str(i)]=float(np.mean([weighted(b['conditional_return_means'],prob,q[str(i)][str(h)],'focal_return')-weighted(b['conditional_return_means'],.5,q['incumbent'][str(h)],'focal_return') for b in audits]))
    for d,r in zip(decisions,scored):
        require(all(r[k]==v for k,v in d.items()),'audit changed selected decision')
        require('audit_gain' not in d and 'noisy_audit_isr' not in d,'audit in selection')
        h=d['adaptation'];queries=d['queried_ids'];require(len(queries)==len(set(queries))==d['hf_queries'],'duplicate query')
        expected=(h if queries else 0)+len(queries)*(h+32)
        close(d['hf_episode_cost'],expected,'HF cost')
        if d['budget_cap'] is not None:require(expected<=d['budget_cap'],'HF overspend')
        close(r['audit_gain'],gains[h][d['selected_id']],'audit return')
        close(r['noisy_audit_isr'],max(gains[h].values())-r['audit_gain'],'regret')
        observed={}
        for j in queries:
            b=selections[j]['conditional_return_means'];observed[j]=weighted(b,grid[int(j)],q[j][str(h)],'focal_return')-weighted(b,.5,q['incumbent'][str(h)],'focal_return')
        if 'observed' in d:
            for j,y in d['observed'].items():close(y,0. if j=='incumbent' else observed[j],'observed HF')
        for step in d.get('steps',[]):
            if 'query_id' in step:close(step['observed_delta'],observed[step['query_id']],'sequential HF observation')
    mechanism=s['mechanism_audit'];require(len(mechanism)==32,'mechanism row count')
    for r in mechanism:
        h=str(r['adaptation']);j=r['candidate'];prob=grid[int(j)];b=audits[['A','B'].index(r['block'])]['conditional_return_means']
        new=weighted(b,prob,q[j][h],'focal_return');old=weighted(b,.5,q['incumbent'][h],'focal_return')
        proxy=weighted(b,prob,.5,'focal_return')-weighted(b,.5,.5,'focal_return');crossed=weighted(b,prob,q['incumbent'][h],'focal_return')
        close(r['deployment_delta_audit'],new-old,'mechanism deployment');close(r['proxy_delta_audit'],proxy,'mechanism proxy');close(r['gap'],new-old-proxy,'mechanism gap')
        close(r['candidate_specific_response_effect'],new-crossed,'candidate response component')
        close(r['common_opponent_adaptation_effect'],crossed-old-proxy,'common response component')
        close(r['responder_own_gain'],weighted(b,prob,q[j][h],'response_return')-weighted(b,prob,.5,'response_return'),'response quality')
    coverage=[]
    for h in [4,32]:
        theta=np.array(post[str(h)]['mean']);C=np.array(post[str(h)]['covariance']);noise=post[str(h)]['noise_variance']
        for r in e.load(folder/'features.json'):
            x=np.array(r['features']);prediction=r['proxy_delta']+x@theta;var=float(x@C@x);error=gains[h][r['id']]-prediction
            coverage.append({'seed':s['seed'],'adaptation':h,'candidate':r['id'],'predicted_delta':float(prediction),'audit_delta':gains[h][r['id']],
                'absolute_model_error':abs(float(error)),'absolute_proxy_error':abs(gains[h][r['id']]-r['proxy_delta']),
                'latent_95_contains_noisy_audit':abs(error)<=1.96*math.sqrt(max(var,0)),
                'observation_predictive_95_contains_noisy_audit':abs(error)<=1.96*math.sqrt(max(var+noise,0)),
                'scope':'Coverage diagnostic against noisy independent audit; within-root correlated candidates, not 240 independent tests.'})
    return {'seed':s['seed'],'scored':scored,'mechanism':mechanism,'coverage':coverage,'frames':frames,'episodes':704,'verified_input_hashes':e.hashes,'contexts':set.union(*stages.values())}
