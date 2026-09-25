import json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=pathlib.Path(__file__).resolve().parent/'autodl_results_20260915'
load=lambda p:json.loads((B/p).read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.facecolor':'white'})
fig,ax=plt.subplots(2,2,figsize=(11.8,8.2),layout='constrained')
blue='#23659b';orange='#be6e28';green='#307961';gray='#5f6670'
d=load('job_confirmatory/confirmatory/e3c/closed_loop_summary.json')['effects'];keys=['pivot_voi_minus_proxy_only','pivot_voi_minus_global_voi','pivot_voi_minus_paired_lucb'];labels=['vs Proxy Only','vs Global-VOI','vs Paired-LUCB']
for i,(k,label) in enumerate(zip(keys,labels)):
 r=d[k];ax[0,0].errorbar(r['mean'],2-i,xerr=[[r['mean']-r['ci_low']],[r['ci_high']-r['mean']]],fmt='o',color=blue,capsize=4)
ax[0,0].axvline(0,color=gray,lw=1,ls='--');ax[0,0].set(yticks=[2,1,0],yticklabels=labels,xlabel='Cumulative selection-loss reduction (positive favors PIVOT)',title='A  Original E3C reproduced');ax[0,0].grid(axis='x',alpha=.18);ax[0,0].text(.02,.02,'Two controlled worlds; 30 seeds each\nRepository numbers matched',transform=ax[0,0].transAxes,fontsize=9,color=gray);ax[0,0].set_ylim(-.55,2.5)
pair=load('mpe_pairwise_diagnostic/summary.json')['coverage_vs_noisy_audit_pair_differences'];vals=[pair['covered_coefficient_only'],pair['covered_with_measurement'],pair['covered_with_pair_discrepancy']]
ax[0,1].bar(range(3),np.array(vals)*100,color=[gray,blue,green],width=.6)
for i,v in enumerate(vals):ax[0,1].text(i,v*100+2,f'{v:.1%}',ha='center')
ax[0,1].axhline(95,color=gray,ls='--',lw=1);ax[0,1].set(xticks=range(3),xticklabels=['Coefficient\nuncertainty','+ Audit\nmeasurement noise','+ Calibration\ndiscrepancy'],ylabel='Coverage of noisy pair differences (%)',ylim=(0,112),title='B  MPE ranking uncertainty is understated')
for name,file,color,marker in [('Shared linear prior','replay_mpe30/summary.json',blue,'o'),('Discrepancy prior','mpe_uncertainty_ablation/summary.json',orange,'s'),('Random batch','replay_mpe30/summary.json',gray,'^')]:
 method='random_batch' if name=='Random batch' else 'author_pivot_voi';rs=sorted([r for r in load(file)['results'] if r['method']==method],key=lambda r:r['budget_packages']);xs=[r['mean_queries'] for r in rs];ys=[r['audit_gain']['mean'] for r in rs];lo=[r['audit_gain']['ci95'][0] for r in rs];hi=[r['audit_gain']['ci95'][1] for r in rs]
 ax[1,0].plot(xs,ys,label=name,color=color,marker=marker);ax[1,0].fill_between(xs,lo,hi,color=color,alpha=.075)
ax[1,0].axhline(0,color=gray,lw=1,ls='--');ax[1,0].set(xlabel='Mean query packages actually used (4 response replicates each)',ylabel='Audited deployment improvement',title='C  More uncertainty does not ensure better selection');ax[1,0].legend(frameon=False,fontsize=9);ax[1,0].grid(alpha=.18)
st=load('mpe_audit_resolution.json');rs=st['rows'];values=sorted([r['half_split_winner_agreement'] for r in rs]);ax[1,1].scatter(range(1,31),np.array(values)*100,s=24,color=blue);ax[1,1].axhline(st['mean_half_split_winner_agreement']*100,color=orange,lw=1.5,label=f"Mean {st['mean_half_split_winner_agreement']:.1%}");ax[1,1].set(xlabel='Trained policy panel (sorted for display)',ylabel='Agreement between audit halves (%)',ylim=(-3,103),title='D  Audit winners are unstable');ax[1,1].legend(frameon=False);ax[1,1].grid(alpha=.18)
fig.suptitle('PIVOT: successful controlled replication, unresolved external selection advantage',fontsize=14,fontweight='bold')
fig.supxlabel('B–D: post-hoc diagnostics on 30 existing MPE panels. Noisy audit means are not known truth.\nBands are exploratory seed-bootstrap intervals; no new confirmatory claim.',fontsize=9,color=gray)
fig.savefig(B/'research_progress.png',dpi=170);fig.savefig(B/'research_progress.pdf');plt.close(fig)
print('saved',B/'research_progress.png')
