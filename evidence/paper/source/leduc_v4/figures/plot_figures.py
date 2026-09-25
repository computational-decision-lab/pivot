"""Recreate standalone figures using only the adjacent saved plot_data.csv.
Requires Python 3, matplotlib and numpy. Run: python3 plot_figures.py
No experiment execution, fitting, resampling, or new significance tests.
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
import numpy as np

ROOT=Path(__file__).resolve().parent
DATA=list(csv.DictReader((ROOT/'plot_data.csv').open()))
BLUE='#276A9E'; ORANGE='#C46B2A'; GRAY='#75818E'; INK='#25323D'; TEAL='#297D70'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#BCC4CC','axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':220})
def rows(f): return [r for r in DATA if r['figure']==f]
def arr(rs,key): return np.array([float(r[key]) for r in rs])
def clean(ax,axis='x'):
    ax.grid(axis=axis,color='#E5E9ED',linewidth=.7);ax.set_axisbelow(True)
    ax.tick_params(length=0,pad=7)
def save(fig,name):
    for ext in ['pdf','png']: fig.savefig(ROOT/(name+'.'+ext),bbox_inches='tight',facecolor='white')
    plt.close(fig)
def errs(ax,rs,y,color=BLUE,marker='o',label=None):
    m,lo,hi=[arr(rs,k) for k in ('mean','lo','hi')]
    ax.errorbar(m,y,xerr=np.array([m-lo,hi-m]),fmt=marker,color=color,ecolor=color,capsize=3,markersize=5.5,lw=1.3,label=label,zorder=4)

r=rows('01_reversal')
fig,ax=plt.subplots(figsize=(5.8,4.1));fig.subplots_adjust(bottom=.23,top=.79,left=.17,right=.95)
x=np.arange(2);m,lo,hi=[arr(r,k) for k in ('mean','lo','hi')]
ax.bar(x,m,color=[BLUE,ORANGE],width=.46,zorder=3)
ax.errorbar(x,m,yerr=np.array([m-lo,hi-m]),fmt='none',capsize=5,color=INK,lw=1.3,zorder=4)
for i in x:ax.text(i,hi[i]+.025,f'{100*m[i]:.1f}%',ha='center',fontsize=13,fontweight='bold')
ax.set_xticks(x,[v['label'] for v in r]);ax.set_ylim(0,.81);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylabel('Proxy-positive updates that reverse (%)')
clean(ax,'y');fig.suptitle('Longer adaptation increases reversals',x=.17,ha='left',y=.98,fontweight='bold')
fig.text(.17,.865,'Leduc v4 · 30 confirmation roots · mean and saved 95% CI',fontsize=9,color=GRAY)
fig.text(.17,.02,'Conditional on positive proxy gain; computed within each root,\nthen averaged. A reversal means deployment gain < 0.',fontsize=8.5,color=GRAY)
save(fig,'01_proxy_deployment_reversal')

r=rows('02_method_regret');fig,ax=plt.subplots(figsize=(9.5,7.1));fig.subplots_adjust(left=.35,right=.86,top=.85,bottom=.16)
y=np.arange(len(r));colors=[BLUE if v['key'].startswith('pivot') else GRAY for v in r]
for j,(v,c) in enumerate(zip(r,colors)):errs(ax,[v],[j],c)
ax.set_yticks(y,[v['label'] for v in r]);ax.invert_yaxis();ax.set_xscale('symlog',linthresh=.001,linscale=1)
ax.set_xlim(-.00018,1.3);ax.set_xticks([0,.001,.01,.1,1],['0','0.001','0.01','0.1','1']);ax.set_xlabel('Selection regret (lower is better)')
for j,v in enumerate(r):ax.text(1.035,j,f"{float(v['mean']):.5f}",transform=ax.get_yaxis_transform(),va='center',fontsize=9)
ax.text(1.035,1.035,'Mean',transform=ax.transAxes,fontsize=9,color=GRAY)
clean(ax);fig.suptitle('Selection quality across all reported methods',x=.035,ha='left',y=.985,fontweight='bold')
fig.text(.035,.932,'Leduc v4 · long adaptation (8) · exact calibration · 30 roots',fontsize=9,color=GRAY)
fig.text(.035,.899,'All HF methods use 16,384 hands, except stopping (12,083.2 mean) and All-HF (65,536).',fontsize=8.5,color=GRAY)
fig.text(.035,.035,'Points and intervals are saved root means and 95% bootstrap CIs.\nSymmetric-log x axis: linear near zero (threshold 0.001), logarithmic beyond it.\nUniform fixed is the exact query-subset expectation; Uniform small averages 64 Monte Carlo draws.',fontsize=8.5,color=GRAY)
save(fig,'02_selection_regret_all_methods')

r=rows('03_stopping');fig,ax=plt.subplots(figsize=(6.1,4.2));fig.subplots_adjust(left=.17,right=.96,top=.80,bottom=.26)
m=arr(r,'mean');x=np.arange(2)
ax.bar(x,m,width=.48,color=[GRAY,TEAL],zorder=3)
for i,v in enumerate(m):ax.text(i,v+450,f'{v:,.1f}' if v%1 else f'{v:,.0f}',ha='center',fontsize=12,fontweight='bold')
ax.set_xticks(x,[v['label'] for v in r]);ax.set_ylim(0,19500);ax.set_ylabel('Mean HF hands used');ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:,.0f}'));clean(ax,'y')
fig.suptitle('Stopping saves 26.25% of HF hands',x=.17,ha='left',y=.985,fontweight='bold')
fig.text(.17,.865,'Leduc v4 · PIVOT-KG menu · long adaptation · 30 roots',fontsize=9,color=GRAY)
fig.text(.17,.075,'Observed selected-gain difference: 0 in all 30 roots.\nSaved mean hands difference: −4,300.8; 95% CI [−6,007.5, −2,662.4].\nBars show means; the interval above is for the paired difference.',fontsize=8.5,color=GRAY)
save(fig,'03_stopping_budget')

r=rows('04_e1');frozen=[v for v in r if v['version']=='Frozen'];corrected=[v for v in r if v['version']=='Corrected'];fig,ax=plt.subplots(figsize=(8.2,5.3));fig.subplots_adjust(left=.25,right=.96,top=.77,bottom=.22)
y=np.arange(len(corrected));errs(ax,frozen,y-.12,GRAY,'s','Frozen analysis');errs(ax,corrected,y+.12,BLUE,'o','Corrected analysis')
ax.set_yticks(y,[v['label'] for v in corrected]);ax.invert_yaxis();ax.axvline(0,color=INK,lw=.8,zorder=1);ax.set_xlim(-.045,.17);ax.set_xlabel('PIVOT-KG − Uniform selected gain (positive favors KG)');clean(ax)
ax.legend(frameon=False,loc='lower right',bbox_to_anchor=(.99,1.015),ncol=2,fontsize=9)
fig.suptitle('All seven E1 settings, including null and negative results',x=.04,ha='left',y=.98,fontweight='bold')
fig.text(.04,.875,'Leduc v3 suite · long adaptation · primary budget 16,384 hands · 30 roots per setting',fontsize=9,color=GRAY)
fig.text(.04,.05,'Frozen and corrected analyses use the same saved worlds; they are not independent replications.\nPointwise saved 95% CIs; no adjustment for seven settings. The effect is not uniformly positive.',fontsize=8.5,color=GRAY)
save(fig,'04_e1_all_seven_settings')

r=rows('05_contrasts');fig,ax=plt.subplots(figsize=(10.1,6.3));fig.subplots_adjust(left=.40,right=.82,top=.85,bottom=.18)
y=np.arange(len(r));errs(ax,r,y,BLUE);ax.set_yticks(y,[v['label'] for v in r]);ax.invert_yaxis();ax.axvline(0,color=INK,lw=.8);ax.set_xlim(-.04,.08);clean(ax)
ax.set_xlabel('Selected-gain difference (positive favors first term)')
for j,v in enumerate(r):ax.text(1.035,j,f"{float(v['mean']):+.5f}",transform=ax.get_yaxis_transform(),va='center',fontsize=9)
ax.text(1.035,1.035,'Mean',transform=ax.transAxes,fontsize=9,color=GRAY)
fig.suptitle('Component contrasts retain the mixed findings',x=.035,ha='left',y=.98,fontweight='bold')
fig.text(.035,.911,'Leduc v4 · long adaptation · primary budget 16,384 · 30 roots',fontsize=9,color=GRAY)
fig.text(.035,.05,'All top-level gain contrasts with saved paired 95% CIs; H_G is the long-minus-short interaction.\nH_A expected Uniform is the prespecified primary comparison. Other intervals are unadjusted.\nA zero-width interval is the recorded result; it is not a claim of universal equivalence.',fontsize=8.5,color=GRAY)
save(fig,'05_v4_component_contrasts')
print('Created five PDF + PNG pairs.')
