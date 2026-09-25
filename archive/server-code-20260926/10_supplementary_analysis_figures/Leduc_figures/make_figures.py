"""Reproduce standalone Leduc figures from archived results. No experiment reruns.
Run: python code/make_figures.py (numpy, matplotlib, pypdf).
All new intervals: root-level percentile bootstrap, 10000 draws, seed 20260917.
"""
from pathlib import Path
import csv, json, hashlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter

B=Path(__file__).resolve().parents[1]; S=B/'sources'; F=B/'figures'; D=B/'plot_data'
F.mkdir(exist_ok=True);D.mkdir(exist_ok=True)
summary=json.loads((S/'reanalysis_v12/leduc_v2b_confirm/summary.json').read_text())
protocol=json.loads((S/'leduc_v2b_confirm/protocol.json').read_text())
rows=json.loads((S/'reanalysis_v12/leduc_v2b_confirm/scored_decisions.json').read_text())
roots=[json.loads(p.read_text()) for p in sorted((S/'leduc_v2b_confirm').glob('seed_*/summary.json'))]
assert [r['seed'] for r in roots]==protocol['seeds'] and len(roots)==30
assert all(r['status']=='complete' for r in roots)
seal=json.loads((S/'reanalysis_v12/leduc_v2b_confirm/selection_seal.json').read_text())
assert hashlib.sha256((S/'reanalysis_v12/leduc_v2b_confirm/decisions_sealed.json').read_bytes()).hexdigest()==seal['decisions_sha256']
assert not any(r['stop_reason'].startswith('author_error') for r in rows)
CAPS=protocol['hf_caps'];CAP=protocol['primary_hf_cap'];COST=protocol['query_cost']
BOOT_SEED=20260917;NBOOT=10000

def boot(x):
 x=np.array(x,float);assert len(x)>0 and np.isfinite(x).all()
 means=x[np.random.default_rng(BOOT_SEED).integers(0,len(x),(NBOOT,len(x)))].mean(1)
 return {'mean':float(x.mean()),'lo':float(np.percentile(means,2.5)),'hi':float(np.percentile(means,97.5)),'n':len(x)}

def vals(method,h,cap=CAP,field='isr'):
 rr=sorted([r for r in rows if r['method']==method and r['adaptation']==h and (r['cap']==cap or r['cap'] is None)],key=lambda r:r['root'])
 assert len(rr)==30 and len({r['root'] for r in rr})==30,(method,h,cap,len(rr))
 return np.array([r[field] for r in rr])

def writecsv(name,records):
 with (D/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#7A7A7A','axes.linewidth':.6,'axes.titlelocation':'left','grid.color':'#E4E7EB','grid.linewidth':.5,'savefig.dpi':300,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','lines.linewidth':1.4,'axes.unicode_minus':True})
C={'kg':'#0072B2','uniform':'#D55E00','ivr':'#009E73','lucb':'#CC79A7','stop':'#6C69A8','proxy':'#7A7A7A','nohf':'#9A6900','equilibrator':'#0072B2','exploiter':'#D55E00'}
figs=[];captions={}

def axis(ax):ax.grid(axis='y',zorder=0);ax.set_axisbelow(True)
def err(ax,x,z,color,fmt='o',**kw):
 ax.errorbar(x,z['mean'],yerr=[[z['mean']-z['lo']],[z['hi']-z['mean']]],fmt=fmt,color=color,ms=4,capsize=2,elinewidth=1,**kw)
def save(fig,name,caption):
 fig.savefig(F/(name+'.pdf'),metadata={'Title':name,'Author':'','Subject':'Leduc existing-data analysis','Keywords':'Leduc;paired;root-bootstrap'})
 fig.savefig(F/(name+'.png'),dpi=300)
 fig.savefig(F/(name+'.svg'))
 figs.append((name,fig));captions[name]=caption

# Exact root/candidate values; A and B are duplicate exact labels, not two samples.
trans=[];rootstats=[]
for r in roots:
 for h in [1,8]:
  p=np.array([r['proxy_deltas'][str(i)] for i in range(9)])
  y=np.array([r['audit_gains'][str(h)][str(i)] for i in range(9)])
  for i in range(9):trans.append({'root':r['seed'],'responder_type':r['responder_type'],'adaptation':h,'response_weight':r['response_weights']['short' if h==1 else 'long'],'candidate':i,'alpha':protocol['candidate_alphas'][i],'proxy_delta':p[i],'deployment_delta':y[i]})
  pos=p>0
  rootstats.append({'root':r['seed'],'responder_type':r['responder_type'],'adaptation':h,'IDE':float(np.abs(p-y).mean()),'squared_gap':float(((p-y)**2).mean()),'IRR':float((y[pos]<0).mean()),'positive_proxy_candidates':int(pos.sum())})
writecsv('transition_values.csv',trans);writecsv('root_mechanism_metrics.csv',rootstats)
for h in [1,8]:
 z=boot([r['IDE'] for r in rootstats if r['adaptation']==h])
 assert np.allclose([z[x] for x in ['mean','lo','hi']],[summary['mechanism'][str(h)]['IDE_mean_abs_delta_error'][x] for x in ['mean','lo','hi']])

fig,aa=plt.subplots(2,2,figsize=(5.5,4.8))
axes=aa.ravel();fig.subplots_adjust(left=.12,right=.98,bottom=.16,top=.88,wspace=.35,hspace=.66)
fig.suptitle('Leduc: proxy improvement and opponent response',fontsize=10,y=.98,x=.08,ha='left')
for ax,h,label in zip(axes[:2],[1,8],['A  Short response','B  Long response']):
 ts=[r for r in trans if r['adaptation']==h]
 for typ,m in [('equilibrator','o'),('exploiter','^')]:
  rr=[r for r in ts if r['responder_type']==typ]
  ax.scatter([r['proxy_delta'] for r in rr],[r['deployment_delta'] for r in rr],s=9,marker=m,color=C[typ],alpha=.55,linewidths=0,label=typ.capitalize())
 ax.set_xlim(-.08,2.90);ax.set_ylim(-2.3,2.90)
 ax.fill_between([0,2.90],-2.3,0,color='#F7EBE5',zorder=-2)
 ax.plot([-.08,2.90],[-.08,2.90],color='#555',ls='--',lw=.8)
 ax.axhline(0,color='#555',lw=.7);ax.axvline(0,color='#555',lw=.7)
 ax.set_title(label);ax.set_xlabel(r'Proxy improvement $\Delta_V$');ax.set_ylabel(r'Deployment improvement $\Delta_*$')
 irr=summary['mechanism'][str(h)]['IRR_reversal_rate_given_proxy_positive']
 ax.text(.05,.89,f"IRR = {irr['mean']:.0%} [{irr['lo']:.0%}, {irr['hi']:.0%}]",transform=ax.transAxes,fontsize=8)
ax=axes[2];axis(ax)
for x,h in enumerate([1,8]):
 z=boot([r['IDE'] for r in rootstats if r['adaptation']==h]);err(ax,x,z,C['kg'])
 ax.text(x,z['hi']+.06,f"{z['mean']:.3f}",ha='center',fontsize=8)
ax.set_xlim(-.5,1.5);ax.set_ylim(0,1.45);ax.set_xticks([0,1],['Short','Long']);ax.set_ylabel('Mean absolute error (IDE)');ax.set_title('C  Improvement error')
ax=axes[3];axis(ax)
for x,h in enumerate([1,8]):err(ax,x,summary['mechanism'][str(h)]['IRR_reversal_rate_given_proxy_positive'],C['kg'])
ax.set_title('D  Improvement reversals');ax.set_ylabel('Conditional reversal rate');ax.set_xticks([0,1],['Short','Long']);ax.set_xlim(-.5,1.5);ax.set_ylim(-.025,1);ax.yaxis.set_major_formatter(PercentFormatter(1))
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.03),ncol=2,frameon=False)
fig.text(.53,.013,'30 roots; 9 candidate entries/root; 95% root-bootstrap CI',fontsize=7,ha='center')
save(fig,'fig01_mechanism','Leduc confirmation, eta=0.15, 30 independent root seeds (15 per latent response type). A-B: all 270 root-candidate values per condition; points are not independent replicates. Short/long refer to prescribed weights w=1-(1-eta)^T at T=1/8 (w=0.15/0.727509), not T learning iterations. Dashed diagonal denotes agreement; shaded quadrant indicates proxy-positive deployment-negative transitions. IRR is conditional on positive proxy improvement and is averaged per root. C: mean absolute improvement error over nine candidate entries, including the zero-update copy. D: conditional reversal rates. Bars and IRR brackets are 95% root-level percentile bootstrap intervals (10,000 draws; seed 20260917). Rewards use native Leduc units; audit values are exact for the specified policies.')

# All method-budget point estimates and uncertainty are postprocessing the sealed choices.
methods=['pivot_kg','pivot_kg_stop','uniform_v2','ivr_v2','lucb_v2','no_hf_v2','proxy_only','all_hf_reference','author_random_hf','author_paired_lucb','author_global_voi','author_pivot_voi']
agg=[]
for h in [1,8]:
 for m in methods:
  for cap in ([None] if m in ['no_hf_v2','proxy_only','all_hf_reference'] else CAPS):
   for metric in ['isr','gain']:
    z=boot(vals(m,h,CAP if cap is None else cap,metric))
    agg.append({'adaptation':h,'method':m,'cap':cap,'metric':metric,'mean_physical_hands':float(vals(m,h,CAP if cap is None else cap,'hf_episode_cost').mean()),**z})
writecsv('all_method_budget_statistics.csv',agg)
for a in summary['method_table_cap192']:
 for metric,col in [('gain','mean_gain'),('isr','mean_isr')]:
  assert np.isclose(vals(a['method'],a['adaptation'],CAP,metric).mean(),a[col])

fig=plt.figure(figsize=(5.5,5.6));axes=[fig.add_axes([.12,.64,.36,.24]),fig.add_axes([.62,.64,.36,.24]),fig.add_axes([.34,.27,.61,.21])]
fig.suptitle('Leduc: decision quality at a matched HF budget',fontsize=10,x=.08,ha='left',y=.98)
styles=[('uniform_v2','Uniform v2',C['uniform'],'s','--'),('ivr_v2','IVR v2',C['ivr'],'^',':'),('lucb_v2','LUCB v2',C['lucb'],'v','-.'),('pivot_kg_stop','PIVOT-KG stop',C['stop'],'D',':'),('pivot_kg','PIVOT-KG v2',C['kg'],'o','-')]
for ax,h,title in zip(axes[:2],[1,8],['A  Short response','B  Long response']):
 axis(ax)
 for m,lab,c,mk,ls in styles:
  zs=[boot(vals(m,h,cap)) for cap in CAPS]
  ax.errorbar([1,2,4],[z['mean'] for z in zs],yerr=[[z['mean']-z['lo'] for z in zs],[z['hi']-z['mean'] for z in zs]],label=lab,color=c,marker=mk,ms=4,mfc='white' if m!='pivot_kg' else c,ls=ls,capsize=2,lw=1.15)
 for m,label,color,mk in [('no_hf_v2','Calibration only',C['nohf'],'d'),('proxy_only','Proxy only',C['proxy'],'x')]:
  err(ax,0,boot(vals(m,h)),color,mk,label=label)
 err(ax,9,boot(vals('all_hf_reference',h)),'#202020','*',label='All-HF (noisy)')
 ax.axhline(0,color='#333',lw=.6)
 ax.set_xticks([0,1,2,4,9]);ax.set_xlabel('HF query cap\n(8,192 hands/query)');ax.set_ylabel('Selection regret (ISR)');ax.set_title(title);ax.set_ylim(-.04,.93)
 ax.text(.98,.94,'v2 methods overlap' if h==1 else 'KG/stop/IVR/LUCB overlap',transform=ax.transAxes,ha='right',va='top',fontsize=7)
# primary paired forest
contrasts=[]
def contrast(name,h,other,primary=False):
 z=boot(vals('pivot_kg',h,field='gain')-vals(other,h,field='gain'));contrasts.append({'contrast':name,'kind':'prespecified primary' if primary else 'secondary','adaptation':h,'cap':CAP,**z});return z
zshort=contrast('KG - Uniform: short',1,'uniform_v2')
zlong=contrast('H2: KG - Uniform: long',8,'uniform_v2',True)
zi=boot((vals('pivot_kg',8,field='gain')-vals('uniform_v2',8,field='gain'))-(vals('pivot_kg',1,field='gain')-vals('uniform_v2',1,field='gain')))
contrasts.append({'contrast':'H3: long - short contrast','kind':'prespecified primary','adaptation':'8-1','cap':CAP,**zi})
zivr=contrast('KG - IVR: long',8,'ivr_v2')
for z,key in [(zlong,'H2_long_primary_gain_minus_comparator_cap192'),(zi,'H3_interaction_long_minus_short')]:
 assert np.allclose([z[k] for k in ['mean','lo','hi']],[summary['hypotheses'][key][k] for k in ['mean','lo','hi']],atol=1e-12)
writecsv('paired_contrasts.csv',contrasts)
ax=axes[2];ax.axvline(0,color='#555',lw=.8);ax.set_title('C  Paired gain differences')
for y,z,label in zip([3,2,1,0],[zshort,zlong,zi,zivr],['Short: KG - Uniform','H2 Long: KG - Uniform','H3 Long minus short','Long: KG - IVR']):
 ax.errorbar(z['mean'],y,xerr=[[z['mean']-z['lo']],[z['hi']-z['mean']]],fmt='o',ms=4,color=C['kg'] if y in [2,1] else '#666',capsize=3)
 ax.text(.079,y,f"{z['mean']:+.4f}",va='center',ha='right',fontsize=7)
ax.set_yticks([3,2,1,0],['Short: KG - Uniform','H2: long','H3: interaction','Long: KG - IVR'],fontsize=8);ax.set_ylim(-.6,3.6);ax.set_xlim(-.01,.083);ax.set_xlabel('Gain difference (favors KG > 0)')
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.08,-.004),ncol=3,frameon=False,columnspacing=1.2)
fig.text(.08,.15,'30 confirmation roots; 95% root-bootstrap CI.\nPrimary cap: 16,384 hands (2 queries).',fontsize=7,linespacing=1.5)
save(fig,'fig02_budget_value','Same confirmation cohort as Figure 1. A-B: v2 selectors share the calibrated correlated posterior; all plotted means and intervals are computed across 30 roots from the archived sealed decisions. Allocated query caps 1/2/4 correspond to at most 8,192/16,384/32,768 physical hands; the stopping variant can spend less (actual mean cost is included in the CSV). A query averages 4,096 paired hand differences. Zero-cost proxy/calibration-only points and the nine-query noisy all-HF reference (73,728 hands) are plotted at their actual costs. Vertical scale is shared; short-condition v2 regrets are all zero. PIVOT-KG, stopping variant, IVR and LUCB have identical regret vectors here; all methods are listed separately in the data and Figure S2. C: paired root-level gain contrasts at the prespecified two-query cap; positive favors KG. H2 and H3 share the same values because every short-condition KG-Uniform difference is zero; they are not independent replications. 95% percentile root bootstrap, 10,000 draws, seed 20260917; intervals are not multiplicity adjusted. Costs exclude offline calibration, exact post-decision audit, and simulator wall time.')

# Latent-family curves: pointwise CIs over 15 roots of each type.
fig,axes=plt.subplots(1,2,figsize=(5.5,3.5),sharex=True,sharey=True);fig.subplots_adjust(left=.13,right=.98,bottom=.28,top=.82,wspace=.26)
fig.suptitle('Leduc: candidate values by response family',fontsize=10,x=.10,ha='left',y=.98)
typeout=[]
for ax,h,title in zip(axes,[1,8],['A  Short response','B  Long response']):
 axis(ax);ax.axhline(0,color='#555',lw=.8)
 for typ,mk in [('equilibrator','o'),('exploiter','^')]:
  for field,ls in [('proxy_delta','--'),('deployment_delta','-')]:
   zs=[]
   for i in range(9):
    z=boot([r[field] for r in trans if r['adaptation']==h and r['responder_type']==typ and r['candidate']==i]);zs.append(z);typeout.append({'adaptation':h,'responder_type':typ,'alpha':protocol['candidate_alphas'][i],'metric':field,**z})
   x=protocol['candidate_alphas'];ax.plot(x,[z['mean'] for z in zs],color=C[typ],ls=ls,marker=mk if field=='deployment_delta' else None,ms=3,label=f"{typ.capitalize()} / {'proxy' if field=='proxy_delta' else 'deployment'}")
   ax.fill_between(x,[z['lo'] for z in zs],[z['hi'] for z in zs],color=C[typ],alpha=.09)
 ax.set_title(title);ax.set_xlabel(r'Candidate mixture $\alpha$');ax.set_ylabel('Expected improvement');ax.set_ylim(-1.65,2.2);ax.set_xticks([0,.25,.5,.75,1])
h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower left',bbox_to_anchor=(.10,.005),ncol=2,frameon=False)
writecsv('response_type_curves.csv',typeout)
save(fig,'figS01_response_types','Confirmation roots, eta=0.15. Candidate policies interpolate from the focal incumbent (alpha=0) toward its exact best response to the fixed proxy opponent (alpha=1). Deployment opponents interpolate toward either an exact best response to the candidate (exploiter) or a CFR+ average policy (equilibrator; 2,000 iterations, not a certified exact equilibrium). Each type has 15 roots. Solid curves are exact deployment improvements and dashed curves proxy improvements; shaded bands are pointwise 95% bootstrap intervals across roots within type. Type labels are used only for auditing, not passed to selectors. The panel illustrates the constructed response-family mechanism, not learning dynamics or population frequencies.')

# Full method inventory: do not suppress author adapters, do not treat them as fair controlled acquisition comparisons.
labels={'pivot_kg':'PIVOT-KG v2','pivot_kg_stop':'PIVOT-KG stop','uniform_v2':'Uniform v2','ivr_v2':'IVR v2','lucb_v2':'LUCB v2','no_hf_v2':'Calibration only','proxy_only':'Proxy only','all_hf_reference':'All-HF (9 queries)','author_random_hf':'Author Random-HF *','author_paired_lucb':'Author Paired-LUCB *','author_global_voi':'Author Global-VOI *','author_pivot_voi':'Author PIVOT-VOI *'}
fig,axes=plt.subplots(1,2,figsize=(5.5,5.2),sharey=True);fig.subplots_adjust(left=.34,right=.98,bottom=.20,top=.86,wspace=.15)
fig.suptitle('Leduc: full method inventory',fontsize=10,x=.12,ha='left',y=.98)
for ax,h,title in zip(axes,[1,8],['A  Short response','B  Long response']):
 ax.set_title(title);ax.grid(axis='x');ax.axvline(0,color='#555',lw=.6)
 for y,m in enumerate(methods):
  z=boot(vals(m,h));color=C['kg'] if m=='pivot_kg' else C['uniform'] if m=='uniform_v2' else '#777'
  ax.errorbar(z['mean'],y,xerr=[[z['mean']-z['lo']],[z['hi']-z['mean']]],fmt='o',ms=4,color=color,capsize=2)
  # Exact values retained in source CSV, not overprinted on intervals.
 ax.set_xlim(-.025,.92);ax.set_xlabel('ISR (lower is better)');ax.set_yticks(range(len(methods)),[labels[m] for m in methods],fontsize=7);ax.axhline(7.5,color='#888',ls=':',lw=.8)
axes[0].invert_yaxis()
fig.text(.10,.03,'* Author adapters: different posterior and batch rule; feature-centering issue.\n   Diagnostic comparison; not evidence of original-method inferiority.\n   30 roots; 95% root-bootstrap CI. Costs differ for zero-HF and all-HF.',fontsize=7,linespacing=1.6)
save(fig,'figS02_all_methods','Complete recorded method inventory on the same 30 confirmation roots. Dots are mean selection regret; horizontal bars are 95% root-bootstrap intervals. Query methods use the primary cap of 16,384 hands; calibration-only and proxy-only use zero online HF, and all-HF uses 73,728 hands. The author-code adapters use their own linear posterior and batch outcome rule; their current feature map x=(alpha-0.5)/0.25 vanishes at alpha=0.5 rather than at the incumbent alpha=0. They therefore do not isolate the acquisition rule under matched modeling assumptions. Results are retained transparently, but poor adapter performance is not evidence that the original PIVOT algorithm is intrinsically inferior. No method has been silently removed or relabeled.')

# Separate negative cohort; do not connect eta points into a confirmation response curve.
neg=json.loads((S/'leduc_v3_suite/eta_0.40_loro/summary.json').read_text())
nrows=json.loads((S/'leduc_v3_suite/eta_0.40_loro/scored_decisions.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(5.5,3.4));fig.subplots_adjust(left=.12,right=.98,bottom=.25,top=.79,wspace=.42)
fig.suptitle('Boundary: a large gap need not create query value',fontsize=10,x=.10,ha='left',y=.98)
boundary_data=[]
ax=axes[0];axis(ax)
for x,h in enumerate([1,8]):
 z=neg['mechanism'][str(h)]['noise_corrected_squared_gap_all_roots'];boundary_data.append({'cohort':'eta_0.40_calibration_LORO','metric':'squared_gap','adaptation':h,**z});err(ax,x,z,C['kg']);ax.text(x,z['hi']+.12,f"{z['mean']:.3f}",ha='center',fontsize=8)
ax.set_xticks([0,1],['Short','Long']);ax.set_ylim(0,5.5);ax.set_xlim(-.5,1.5);ax.set_title('A  Squared gap');ax.set_ylabel('Squared improvement error')
ax=axes[1];ax.axvline(0,color='#555',lw=.8);ax.set_title('B  KG - Uniform gain')
for y,h in enumerate([1,8]):
 aa={r['root']:r['gain'] for r in nrows if r['method']=='pivot_kg' and r['adaptation']==h and r['cap']==CAP};bb={r['root']:r['gain'] for r in nrows if r['method']=='uniform_v2' and r['adaptation']==h and r['cap']==CAP}
 z=boot([aa[r]-bb[r] for r in sorted(aa)]);boundary_data.append({'cohort':'eta_0.40_calibration_LORO','metric':'KG_minus_Uniform_gain','adaptation':h,**z});ax.errorbar(z['mean'],y,xerr=[[z['mean']-z['lo']],[z['hi']-z['mean']]],fmt='o',color=C['kg'],capsize=3)
ax.set_yticks([0,1],['Short','Long']);ax.set_ylim(-.5,1.5);ax.set_xlim(-.007,.025);ax.set_xlabel('Gain difference (favors KG > 0)')
fig.text(.10,.065,'eta = 0.40; 12 calibration roots (LORO).\nLong-response information ceiling = 0.',fontsize=7)
writecsv('negative_boundary_statistics.csv',boundary_data)
save(fig,'figS03_negative_boundary','A separate eta=0.40 calibration/LORO cohort (12 roots), not the eta=0.15 30-root confirmation. A: recorded squared gap means and 95% root-bootstrap intervals for short/long response. B: KG-Uniform paired gain differences at the two-query cap. Under long response the best action is to retain the incumbent, all v2 methods have zero regret, and the recorded world-specific information ceiling is zero despite a large proxy-deployment gap. This is a boundary to a universal gap-implies-query-value claim. Thirty eta=0.40 confirmation roots were generated, but no completed confirmation analysis was available in the inspected server directory; they are not used here. No comparisons across these two cohorts are presented as a prespecified causal eta effect.')

# Gallery embeds original vector PDF pages with captions on the same page.
import io, textwrap
from pypdf import PdfReader, PdfWriter, Transformation
writer=PdfWriter()
for name,fig in figs:
 page=writer.add_blank_page(width=595.276,height=841.89)
 src=PdfReader(F/(name+'.pdf')).pages[0]
 w,h=float(src.mediabox.width),float(src.mediabox.height)
 scale=min(523/w,480/h)
 page.merge_transformed_page(src,Transformation().scale(scale).translate(36,815-h*scale))
 cf=plt.figure(figsize=(595.276/72,841.89/72));cf.text(.06,(790-h*scale)/841.89,name,fontsize=11,weight='bold')
 lines=textwrap.wrap(captions[name],width=96)
 cf.text(.06,(765-h*scale)/841.89,'\n'.join(lines),fontsize=9,va='top',linespacing=1.6)
 cf.text(.06,.03,'Existing-data figures. See archived sources, plot_data, and code/make_figures.py.',fontsize=8)
 buf=io.BytesIO();cf.savefig(buf,format='pdf',transparent=True);buf.seek(0);page.merge_page(PdfReader(buf).pages[0]);plt.close(cf)
writer.add_metadata({'/Title':'Leduc figures and captions','/Author':''})
with (B/'Leduc_图表与图注.pdf').open('wb') as f:writer.write(f)
(B/'docs/figure_captions_en.md').write_text('\n\n'.join('## '+k+'\n\n'+v for k,v in captions.items())+'\n')
(B/'verification/figure_checks.json').write_text(json.dumps({'confirmation_roots':30,'calibration_roots':12,'sealed_decisions_sha256_matches':True,'all_primary_method_means_reproduced':True,'H2_H3_intervals_reproduced':True,'IDE_intervals_reproduced':True,'no_author_error_rows':True,'bootstrap':{'unit':'root','draws':NBOOT,'seed':BOOT_SEED,'method':'percentile','level':.95,'multiple_testing_adjusted':False},'not_run':'No experiment, selector, calibration or confirm analysis was rerun; only plot aggregates recomputed.','software':{'python':sys.version,'numpy':np.__version__,'matplotlib':matplotlib.__version__},'new_figures':list(captions)},indent=2))
print('FIGURES_COMPLETE',len(figs),'H2',zlong)
