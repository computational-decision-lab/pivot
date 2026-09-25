"""Submission-style figures from measured benchmark artifacts only."""
import argparse,json,os,tempfile
from pathlib import Path
cache=Path(tempfile.gettempdir())/'colin-pivot-plot-cache';cache.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(cache))
os.environ.setdefault('XDG_CACHE_HOME',str(cache/'font-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm

COLORS={'author_batch':'#167d9a','author_batch_no_stop':'#204c81','random_batch':'#7d8792',
        'proxy_only':'#c47a38','author_pivot_voi':'#167d9a','author_pivot_no_stop':'#204c81'}
LABELS={'author_batch':'PIVOT (author)','author_batch_no_stop':'PIVOT, fixed budget',
        'random_batch':'Random paired validation','proxy_only':'Proxy only',
        'author_pivot_voi':'PIVOT (author)','author_pivot_no_stop':'PIVOT, fixed budget'}
TASK_LABELS={'push':'MPE2 · Simple Push','adversary':'MPE2 · Simple Adversary',
             'melting_pd':'Melting Pot · Repeated Prisoner’s Dilemma','melting_stag':'Melting Pot · Stag Hunt'}

def read(p):return json.loads(Path(p).read_text())
def ci(x):
    x=np.array(x,float);rng=np.random.default_rng(20260915);d=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    return float(x.mean()),np.quantile(d,[.025,.975])
def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
        'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#aab2bb',
        'axes.labelcolor':'#293744','text.color':'#293744','xtick.color':'#566473','ytick.color':'#566473',
        'savefig.facecolor':'white','figure.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42})
def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=190);fig.savefig(out/(name+'.pdf'));plt.close(fig)
def interval(ax,mean,bounds,y,color):
    ax.errorbar(mean,y,xerr=[[mean-bounds[0]],[bounds[1]-mean]],fmt='o',color=color,capsize=3,ms=6,lw=1.5)
def mechanisms(ax,rows):
    keys=['response_effect','candidate_specific_effect','response_own_reward_gain']
    labels=['Change in update value','Candidate-specific response','Responder’s own reward']
    for y,key,color in zip([2,1,0],keys,['#167d9a','#715a9c','#c47a38']):
        values=[r[key] for r in rows];mean,bounds=ci(values)
        ax.scatter(values,np.full(len(values),y),s=14,color=color,alpha=.3)
        interval(ax,mean,bounds,y,color)
    ax.axvline(0,color='#a0a7af',ls='--',lw=1);ax.set(yticks=[2,1,0],yticklabels=labels,ylim=(-.6,2.6),xlabel='Mean return difference')
    ax.grid(axis='x',alpha=.12)
def gain_cost(ax,rows,mapping,query_key):
    for method in mapping:
        points=sorted([r for r in rows if r['method']==method],key=lambda r:r.get('budget',r.get('budget_packages',0)))
        if not points:continue
        xs=[r[query_key] for r in points];ys=[r['audit_gain']['mean'] for r in points]
        lo=[r['audit_gain']['ci95'][0] for r in points];hi=[r['audit_gain']['ci95'][1] for r in points]
        author=method in ['author_batch','author_pivot_voi'];proxy=method=='proxy_only'
        ax.plot(xs,ys,marker='D' if author else ('x' if proxy else 'o'),linestyle='-',
                color=COLORS[method],label=LABELS[method],ms=6 if author or proxy else 4,
                markerfacecolor='none' if author else COLORS[method],lw=1.7,zorder=5 if author else (6 if proxy else 3))
        ax.fill_between(xs,lo,hi,color=COLORS[method],alpha=.07)
        if len(set(xs))==1:
            ax.errorbar([xs[0]],[ys[0]],yerr=[[ys[0]-lo[0]],[hi[0]-ys[0]]],fmt='none',color=COLORS[method],capsize=3,alpha=.8,zorder=4)
    ax.axhline(0,color='#a0a7af',ls='--',lw=1);ax.grid(alpha=.12)
    ax.set(xlabel='Mean paired validation queries actually used',ylabel='Independently evaluated update gain')

def mpe_figure(root,out):
    data=read(root/'fresh_mpe_general/summary.json');mech=read(root/'fresh_mpe_general/mechanism_by_seed.json')
    fig,axes=plt.subplots(2,3,figsize=(15.5,8.3),layout='constrained')
    for i,task in enumerate(['push','adversary']):
        candidates=read(root/f'fresh_mpe_general/{task}/candidate_audit.json');x=[];y=[];se=[]
        for r in candidates:
            a=np.array(r['audit_replicates']);x.extend(r['proxy']);y.extend(a.mean(1));se.extend(a.std(1,ddof=1)/np.sqrt(a.shape[1]))
        ax=axes[i,0];x=np.array(x);y=np.array(y);se=np.array(se)
        ax.errorbar(x,y,yerr=se,fmt='none',ecolor='#167d9a',alpha=.23,lw=.8)
        ax.scatter(x,y,c='#167d9a',s=26,edgecolors='white',linewidths=.4,alpha=.8)
        lo=min(x.min(),(y-se).min());hi=max(x.max(),(y+se).max());pad=max((hi-lo)*.08,.01)
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],color='#9fa9b5',ls='--',lw=1)
        ax.axhline(0,color='#c8ccd1',lw=.8);ax.axvline(0,color='#c8ccd1',lw=.8)
        ax.set(xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),xlabel='Cheap backtest update gain',ylabel='Gain after candidate-specific adaptation',title=f'{chr(65+i*3)}  Backtest → deployment')
        ax.text(.03,.97,'32 candidates · 8 fresh training seeds\nBars: ±1 response-replicate SE',transform=ax.transAxes,va='top',fontsize=8,color='#607080')
        mechanisms(axes[i,1],[r for r in mech if r['task']==task]);axes[i,1].set_title(f'{chr(66+i*3)}  What changed after deployment?')
        gain_cost(axes[i,2],data['tables'][task]['methods'],['author_batch','author_batch_no_stop','random_batch','proxy_only'],'mean_queries')
        axes[i,2].set_title(f'{chr(67+i*3)}  Validation benefit and cost');axes[i,2].legend(frameon=False,fontsize=8,loc='best')
        axes[i,0].text(0,1.16,TASK_LABELS[task],transform=axes[i,0].transAxes,fontsize=12,fontweight='bold')
    fig.suptitle('Does a backtest upgrade survive an adaptive world?',fontsize=17,fontweight='bold')
    fig.supxlabel('MPE2 adaptive extension · 8 new training seeds per task · All methods share the same candidate panels.\nIntervals/bands: exploratory seed-cluster bootstrap. Query costs only; training, calibration, features and independent audit excluded.',fontsize=9,color='#637181')
    save(fig,out,'MPE2_general_benchmark')

def melting_figure(root,out):
    b=root/'melting_general_long';data=read(b/'summary.json');scored=read(b/'scored_decisions.json');mech=read(b/'mechanism_by_seed.json')
    fig,axes=plt.subplots(2,3,figsize=(15.5,8.3),layout='constrained')
    for i,task in enumerate(['melting_pd','melting_stag']):
        ax=axes[i,0]
        for method in ['author_pivot_voi','random_batch','proxy_only']:
            rows=[r for r in scored if r['task']==task and r['method']==method and r['budget_packages']==(0 if method=='proxy_only' else 2)]
            short=[r for r in rows if r['response_steps']==32768];long=[r for r in rows if r['response_steps']==131072]
            values=[[r['frozen_gain'] for r in short],[r['audit_gain'] for r in short],[r['audit_gain'] for r in long]]
            summaries=[ci(v) for v in values];xs=[0,32.768,131.072];ys=[v[0] for v in summaries]
            ax.plot(xs,ys,'o-',color=COLORS[method],label=LABELS[method],ms=4)
            ax.fill_between(xs,[v[1][0] for v in summaries],[v[1][1] for v in summaries],color=COLORS[method],alpha=.08)
        ax.axhline(0,color='#a0a7af',ls='--',lw=1);ax.grid(alpha=.12)
        ax.set(xticks=[0,32.768,131.072],xticklabels=['0','32.8k','131.1k'],xlabel='Opponent adaptation steps per branch',ylabel='Gain of the same selected update',title=f'{chr(65+i*3)}  Longer opponent adaptation')
        ax.legend(frameon=False,fontsize=8,loc='upper left');ax.text(.03,.03,'Choices fixed using short-response queries\nPIVOT/random cap: 2 packages (4 paired queries)',transform=ax.transAxes,fontsize=8,color='#607080')
        mechanisms(axes[i,1],[r for r in mech if r['task']==task and r['response_steps']==131072]);axes[i,1].set_title(f'{chr(66+i*3)}  Long-response mechanism')
        rows=[r for r in data['tables'][task]['methods'] if r['response_steps']==32768]
        gain_cost(axes[i,2],rows,['author_pivot_voi','author_pivot_no_stop','random_batch','proxy_only'],'mean_query_packages');axes[i,2].set_xlabel('Mean query packages used (2 paired queries each)')
        axes[i,2].set_title(f'{chr(67+i*3)}  Short-response benefit and cost');axes[i,2].legend(frameon=False,fontsize=8)
        axes[i,0].text(0,1.16,TASK_LABELS[task],transform=axes[i,0].transAxes,fontsize=12,fontweight='bold')
    fig.suptitle('One selected upgrade, different deployment responses',fontsize=17,fontweight='bold')
    fig.supxlabel('Melting Pot adaptive extension · 6 pre-existing training seeds per task · Author choices frozen before replay scoring.\nShort/long audit: 4/2 replicates per candidate; historical-study completion and development evaluation. Descriptive seed-bootstrap bands; not the full official suite.',fontsize=9,color='#637181')
    save(fig,out,'MeltingPot_general_benchmark')

def ranks(v):
    # Average ranks retain genuine ties instead of inventing arbitrary order.
    from scipy.stats import rankdata
    return rankdata(-np.array(v,float),method='average')
def ranking_figure(root,out):
    melting=read(root/'melting_general_long/candidate_audit.json');panels=[]
    for task in ['push','adversary']:
        rows=read(root/f'fresh_mpe_general/{task}/candidate_audit.json')
        panels.append((task,rows))
    for task in ['melting_pd','melting_stag']:
        panels.append((task,[r for r in melting if r['task']==task and r['response_steps']==131072]))
    fig,axes=plt.subplots(2,2,figsize=(12,8.4),layout='constrained');norm=TwoSlopeNorm(vcenter=0,vmin=-3,vmax=3)
    for ax,(task,rows),letter in zip(axes.ravel(),panels,'ABCD'):
        rows=sorted(rows,key=lambda r:r['seed']);matrix=np.array([ranks(r['proxy'])-ranks(np.array(r['audit_replicates']).mean(1)) for r in rows]).T
        im=ax.imshow(matrix,cmap='PuOr_r',norm=norm,aspect='auto')
        for i in range(4):
            for j in range(len(rows)):
                v=matrix[i,j];ax.text(j,i,f'{v:+g}',ha='center',va='center',fontsize=10,color='white' if abs(v)>=2 else '#263849')
        ax.set(yticks=range(4),yticklabels=[f'Update {j+1}' for j in range(4)],xticks=range(len(rows)),xticklabels=[str(r['seed']) for r in rows],xlabel='Training seed',title=f'{letter}  {TASK_LABELS[task]}')
        ax.tick_params(axis='x',labelsize=8)
    fig.colorbar(im,ax=axes,label='Backtest rank − deployment rank (positive: moves up)',shrink=.68,pad=.025)
    fig.suptitle('Which upgrades change rank after deployment?',fontsize=17,fontweight='bold')
    fig.supxlabel('Ranks use sample means, so these are observed ranking changes—not individually confirmed latent reversals.\nMPE2: short response on fresh panels. Melting Pot: long-response stress test on existing panels. Four candidates per seed; ties use average ranks.',fontsize=9,color='#637181')
    save(fig,out,'Candidate_rank_changes')

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    assert read(a.results/'fresh_mpe_general/summary.json')['status']=='complete'
    assert read(a.results/'melting_general_long/summary.json')['status']=='complete'
    style();mpe_figure(a.results,a.output);melting_figure(a.results,a.output);ranking_figure(a.results,a.output)
    print('Three figure sets generated from completed experiment artifacts')
if __name__=='__main__':main()
