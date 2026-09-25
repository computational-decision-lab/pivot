import argparse,json,time,sys
from native_adapter import NativeEvaluator
p=argparse.ArgumentParser();p.add_argument('--force-destroy',action='store_true');p.add_argument('--reuse-names',action='store_true');p.add_argument('--repeats',type=int,default=3);args=p.parse_args()
base={'speed':20.,'headway':1.5,'distance':5.,'lane_change_distance':15.}
eval=NativeEvaluator({'start_seed':60000,'num_scenarios':1,'horizon':300,'num_agents':8})
eval.env.config['store_map']=False
eval.env.config['force_destroy']=args.force_destroy
eval.env.config['force_reuse_object_name']=args.reuse_names
try:
 for n in range(args.repeats):
  r=eval.run_episode(60000,base,base)
  print(json.dumps({'repeat':n,'force_destroy':args.force_destroy,'reuse_names':args.reuse_names,'return':r['per_agent_returns'],'flags':r['per_agent_flags'],'sha':r['trajectory_sha256'],'positions':r['initial_positions'],'seconds':r['seconds']}),flush=True)
finally:eval.close()
