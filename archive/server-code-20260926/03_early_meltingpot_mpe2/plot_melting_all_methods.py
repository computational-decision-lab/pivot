"""Show every prespecified method at the primary budget, in fixed protocol order."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);a=p.parse_args()
    with (a.analysis/'frontier.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    labels={'proxy_only':'Proxy only','calibrated_no_hf':'Calibration only (no HF)',
            'pivot_sequential':'PIVOT fixed-budget extension','pivot_sequential_adaptive_stop':'PIVOT with stopping extension',
            'uniform_random_matched':'Uniform/Random HF (matched)','global_ivr_matched':'Global-IVR (matched)',
            'posterior_lucb_matched':'Posterior LUCB heuristic (matched)','author_random_hf':'Random HF (author E5C)',
            'author_paired_lucb':'Paired LUCB (author E5C variant)','author_global_voi':'Global-VOI (author E5C variant)',
            'author_pivot_voi':'PIVOT-VOI batch (author E5C)','all_hf_reference':'All-HF reference'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,7.8),sharey=True)
    for ax,h in zip(axes,[4,32]):
        for i,(method,label) in enumerate(labels.items()):
            data=[r for r in rows if int(r['adaptation'])==h and r['method']==method and (r['budget_cap']=='192' or method in ['proxy_only','calibrated_no_hf','all_hf_reference'])];assert len(data)==1
            r=data[0];mean=float(r['mean_audit_isr']);lo=float(r['isr_ci_low']);hi=float(r['isr_ci_high']);cost=float(r['mean_hf_episode_cost'])
            color='#b44448' if method.startswith('pivot_sequential') else '#207d69' if method=='global_ivr_matched' else '#397aa4' if method=='uniform_random_matched' else '#6b6b6b'
            ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=3,color=color)
            ax.text(1.015,i,f'{cost:.0f}',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=9,color='#555555')
        ax.set_title(f'{"Short" if h==4 else "Long"} response ({h} learning episodes)',loc='left')
        ax.set_xlabel('Independent audit ISR (lower is better)');ax.set_xlim(left=0)
        ax.axhline(10.5,color='#aaaaaa',lw=.8,ls=':')
        ax.text(1.01,1.005,'HF cost',transform=ax.transAxes,fontsize=9,ha='left',va='bottom')
    axes[0].set_yticks(range(len(labels)),list(labels.values()));axes[0].invert_yaxis()
    fig.suptitle('All prespecified methods at HF cap 192: 30 fresh seeds',fontsize=14)
    fig.tight_layout(rect=[0,.07,.96,.96],w_pad=4)
    fig.text(.5,.04,'95% root-seed bootstrap intervals. Right-side numbers: charged HF episodes. All-HF exceeds the primary cap.',ha='center',fontsize=9)
    fig.text(.5,.018,'Descriptive secondary comparisons; noisy audit maximum. Fixed skill network, one Melting Pot substrate, one update round.',ha='center',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(a.analysis/f'all_methods_primary_budget.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
