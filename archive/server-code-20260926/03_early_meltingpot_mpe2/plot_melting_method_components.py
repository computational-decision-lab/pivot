"""Present predeclared mechanism and component contrasts; no new hypothesis tests."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);a=p.parse_args()
    s=json.loads((a.analysis/'summary.json').read_text());assert s['seed_count']==30
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),gridspec_kw={'width_ratios':[1,1.35]})
    mechanism=s['confirmation_mechanism']
    for ax,key,title,ylabel in [(axes[0,0],'own_gain','A  Response learning quality','Responder return gain from adaptation'),
                               (axes[1,0],'gap_squared','B  Proxy/deployment separation','Cross-block squared gap (return squared)')]:
        for i,h in enumerate([4,32]):
            z=mechanism[f'{key}_{h}'];m=z['mean'];lo,hi=z['ci95']
            ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=5,color=['#7b91a7','#a84245'][i])
        ax.set_xticks([0,1],['Short (4 episodes)','Long (32 episodes)']);ax.set_xlim(-.5,1.5);ax.set_ylabel(ylabel);ax.set_title(title,loc='left')
        ax.axhline(0,color='#999999',lw=.6)
    ax=axes[0,1]
    for i,(key,label) in enumerate([('short_regret_reduction','Short response'),('long_regret_reduction','Long response'),('interaction','Long minus short')]):
        z=s['primary_effects'][key];m=z['mean'];lo,hi=z['ci95'];ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color='#a84245')
    ax.set_yticks([0,1,2],['Short response','Long response','Long minus short']);ax.invert_yaxis();ax.axvline(0,color='black',lw=.8)
    ax.set_title('C  Does PIVOT beat matched Uniform?',loc='left');ax.set_xlabel('Uniform ISR minus PIVOT ISR; positive favors PIVOT')
    ax=axes[1,1]
    contrasts=['calibration_value','beyond_calibration','fixed_budget_acquisition','stopping_value']
    labels=['Calibration vs proxy','PIVOT vs calibration only','Fixed-budget PIVOT vs Uniform','Early stop vs fixed-budget PIVOT']
    for i,key in enumerate(contrasts):
        for offset,h,color in [(-.13,4,'#7b91a7'),(.13,32,'#a84245')]:
            z=next(x for x in s['registered_diagnostics'] if x['contrast']==key and x['adaptation']==h)
            m=z['mean'];lo,hi=z['ci95'];ax.errorbar(m,i+offset,xerr=[[m-lo],[hi-m]],fmt='o',capsize=3,color=color,label=('Short' if h==4 else 'Long') if i==0 else None)
    ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.axvline(0,color='black',lw=.8);ax.legend(frameon=False,fontsize=9,loc='best')
    ax.set_title('D  Which component contributes?',loc='left');ax.set_xlabel('Paired audited gain difference; positive favors first method')
    fig.suptitle('Melting Pot: mechanism and method, 30 fresh seeds',fontsize=14)
    fig.tight_layout(rect=[0,.04,1,.96],w_pad=3,h_pad=2)
    fig.text(.5,.015,'Fixed skill network; single-round adaptive extension. HF cap 192 for C/D. 95% root-seed bootstrap intervals; component contrasts are diagnostic.',ha='center',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(a.analysis/f'mechanism_and_method.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)
    (a.analysis/'component_figure_inputs.json').write_text(json.dumps({k:s[k] for k in ['protocol_sha256','seed_count','confirmation_mechanism','primary_effects','registered_diagnostics']},indent=2)+'\n')


if __name__=='__main__':main()
