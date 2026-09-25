"""Figures from verified 30-seed native method evidence only."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);a=p.parse_args()
    s=json.loads((a.analysis/'summary.json').read_text());assert s['seed_count']==30
    with (a.analysis/'frontier.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    methods={'pivot_sequential_adaptive_stop':('PIVOT paper-loop extension','#bd3a3a'),
        'uniform_random_matched':('Uniform/Random HF (matched)','#3178a8'),
        'author_paired_lucb':('LUCB (author E5C variant)','#9467bd'),
        'author_global_voi':('Global-VOI (author E5C variant)','#ce8735'),
        'calibrated_no_hf':('Calibration only','#328974'),'proxy_only':('Proxy only','#666666'),
        'all_hf_reference':('All-HF reference','#111111')}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,h in zip(axes,[4,32]):
        for method,(label,color) in methods.items():
            group=sorted([r for r in rows if int(r['adaptation'])==h and r['method']==method],key=lambda r:float(r['mean_hf_episode_cost']))
            x=[float(r['mean_hf_episode_cost']) for r in group];y=[float(r['mean_audit_isr']) for r in group]
            lo=[v-float(r['isr_ci_low']) for r,v in zip(group,y)];hi=[float(r['isr_ci_high'])-v for r,v in zip(group,y)]
            ax.errorbar(x,y,yerr=[lo,hi],label=label,color=color,marker='o',capsize=3,lw=1.4)
        ax.set_title(f'{"Short" if h==4 else "Long"} response: {h} learning episodes');ax.set_xlabel('HF cost: charged native episodes');ax.set_ylabel('Held-out audit ISR (lower is better)')
        ax.set_ylim(bottom=0)
    axes[1].legend(loc='center left',bbox_to_anchor=(1,0.5),frameon=False,fontsize=9)
    fig.suptitle('Melting Pot adaptive extension: 30 fresh response-world seeds')
    fig.text(.5,-.01,'95% root-seed bootstrap intervals. Fixed skill network; one update round. Calibration/proxy/audit costs reported separately.',ha='center',fontsize=9)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(a.analysis/f'hf_cost_vs_regret.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.2));effects=s['primary_effects']
    for y,key in enumerate(['short_regret_reduction','long_regret_reduction','interaction']):
        z=effects[key];m=z['mean'];lo,hi=z['ci95'];ax.errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt='o',color='#bd3a3a',capsize=5)
    ax.axvline(0,color='black',lw=.8);ax.set_yticks([0,1,2],['Short: Uniform - PIVOT ISR','Long: Uniform - PIVOT ISR','Long minus short interaction']);ax.invert_yaxis()
    ax.set_xlabel('Positive favors PIVOT / increasing method value');ax.set_title('Predeclared contrasts at HF cap 192 episodes')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(a.analysis/f'primary_contrasts.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)
if __name__=='__main__':main()
