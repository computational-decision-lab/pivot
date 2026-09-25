"""Compare newly generated numeric summaries against the repository artifacts."""
import argparse,json,pathlib,math
p=argparse.ArgumentParser();p.add_argument('--new',type=pathlib.Path,required=True);p.add_argument('--repo',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--remaining',type=pathlib.Path,required=True);a=p.parse_args()
reports=[]
for experiment,filename in [('e3c','closed_loop_summary.json'),('e5c','efficiency_summary.json'),('e7c','strategic_summary.json'),('e2c','operator_shift_summary.json'),('e4c','ood_summary.json')]:
    current=(a.remaining/(experiment+'_confirmatory')/filename) if experiment in ['e2c','e4c'] else a.new/experiment/filename;reference=a.repo/'results/v9'/(experiment+'-confirmatory')/filename
    if not current.exists():continue
    left=json.loads(reference.read_text());right=json.loads(current.read_text());differences=[];n=[0];maxdiff=[0.]
    def compare(x,y,path):
        if type(x)!=type(y):differences.append({'path':path,'type_difference':True});return
        if isinstance(x,dict):
            if set(x)!=set(y):differences.append({'path':path,'key_difference':True})
            for k in sorted(set(x)&set(y)):compare(x[k],y[k],path+'/'+k)
        elif isinstance(x,list):
            if len(x)!=len(y):differences.append({'path':path,'lengths':[len(x),len(y)]})
            for i,(u,v) in enumerate(zip(x,y)):compare(u,v,path+'/'+str(i))
        elif isinstance(x,(int,float)) and not isinstance(x,bool):
            n[0]+=1;d=abs(x-y);maxdiff[0]=max(maxdiff[0],d)
            if not math.isclose(x,y,rel_tol=1e-8,abs_tol=1e-10):differences.append({'path':path,'reference':x,'new':y,'absolute_difference':d})
        elif x!=y:differences.append({'path':path,'reference':x,'new':y})
    compare(left,right,'')
    decision=json.loads((current.parent/'scientific_decision.json').read_text())
    reports.append({'experiment':experiment,'numeric_fields':n[0],'max_absolute_numeric_difference':maxdiff[0],'difference_count':len(differences),'numeric_difference_count':sum('absolute_difference' in x for x in differences),'common_numeric_fields_match':not any('absolute_difference' in x for x in differences),'matches_with_tolerance':not differences,'differences_first_20':differences[:20],'new_scientific_status':decision['status'],'allowed_claim':decision.get('allowed_claim'),'reference_source_commit':json.loads((reference.parent/'provenance.json').read_text()).get('source_commit'),'new_source_commit':json.loads((current.parent/'provenance.json').read_text()).get('source_commit'),'scope':'Comparison with repository V9 artifacts; source revisions may differ. Not a full paper reproduction.'})
a.output.write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports),flush=True)
