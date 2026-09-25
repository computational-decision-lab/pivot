"""Plot only a completely verified, frozen mechanism cohort."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);a=p.parse_args()
    summary=json.loads((a.analysis/'summary.json').read_text())
    assert summary['accounting']['native_episodes']==2688 and summary['seed_count']==12
    with (a.analysis/'seed_results.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==12
    stats=summary['statistics'];colors=['#3178a8','#cf6a32']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(13,4.1))
    for ax,key,title,ylabel in zip(axs[:2],['responder_gain','cross_block_gap_squared'],
            ['A. Did the responder learn?','B. Did improvement fidelity deteriorate?'],
            ['Responder own-return gain','Cross-block squared gap (return squared)']):
        values=np.array([[float(r[f'{key}_{h}']) for h in [4,32]] for r in rows])
        for pair in values:ax.plot([0,1],pair,color='#a6adb4',alpha=.55,lw=.8,marker='o',ms=3)
        for i,h in enumerate([4,32]):
            s=stats[f'{key}_{h}'];mean=s['mean'];lo,hi=s['ci95']
            ax.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color=colors[i],markersize=7,capsize=5,zorder=5)
        ax.set_xticks([0,1],['Short: 4 episodes','Long: 32 episodes']);ax.set_ylabel(ylabel)
        ax.set_title(title,loc='left',fontsize=11);ax.axhline(0,color='black',lw=.7,alpha=.5)
        ax.set_xlim(-.35,1.35)
    ax=axs[2]
    for i,(prob,color) in enumerate(zip([.25,.75],['#7760a9','#328974'])):
        means=[];low=[];high=[]
        for h in [4,32]:
            s=stats[f'signed_gap_{prob}_{h}'];means.append(s['mean']);low.append(s['mean']-s['ci95'][0]);high.append(s['ci95'][1]-s['mean'])
        x=np.array([0,1])+(i-.5)*.07
        ax.errorbar(x,means,yerr=[low,high],color=color,marker='o',capsize=5,label=f'Probe p={prob}')
    ax.axhline(0,color='black',lw=.7,alpha=.5);ax.set_xticks([0,1],['Short: 4 episodes','Long: 32 episodes']);ax.set_xlim(-.35,1.35)
    ax.set_ylabel('Deployment improvement - proxy improvement');ax.set_title('C. Which updates change?',loc='left',fontsize=11);ax.legend(frameon=False)
    fig.suptitle('Melting Pot adaptive extension: mechanism screen, not a PIVOT comparison',fontsize=12,y=1.01)
    fig.text(.5,-.01,'12 independent response-training seeds; 95% paired seed bootstrap intervals. Fixed official skill network.',ha='center',fontsize=9)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(a.analysis/f'mechanism_short_long.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)
    manifest={n:hashlib.sha256((a.analysis/n).read_bytes()).hexdigest() for n in ['summary.json','seed_results.csv']}
    (a.analysis/'figure_inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
