"""Export neutral, paper-ready figure blocks and measured comparison tables."""
import argparse,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();b=a.results
    mpe=json.loads((b/'fresh_mpe_general/summary.json').read_text());mel=json.loads((b/'melting_general_long/summary.json').read_text())
    assert mpe['status']==mel['status']=='complete'
    text=r'''% Requires graphicx. Copy this file together with the figures/ directory.
% Selected-task adaptive extensions, not complete official benchmark suites.
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/MPE2_general_benchmark.pdf}
\caption{MPE2 adaptive extension on eight newly trained candidate panels per task.
Each panel contains four PPO-generated policy updates. We measure cheap backtest
gains, paired deployment gains after candidate-specific opponent adaptation,
response mechanisms, and the original PIVOT query policy's benefit--cost curve.
Audit uses eight independent response-training replicates per candidate, each
evaluated over 128 episodes. Candidate points share trained panels; intervals
resample training-seed clusters. PIVOT and all comparison choices are frozen
before independent audit generation.}
\label{fig:general-mpe-adaptive}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/MeltingPot_general_benchmark.pdf}
\caption{Melting Pot adaptive extension: the same selected updates under different
opponent-training budgets. We complete the prespecified short- and long-response
audits for six existing panels per substrate. The author PIVOT choices use short
response queries; long-response evaluation tests these fixed choices rather than
reselecting with audit labels. Short/long response budgets are 32,768/131,072
steps per branch, with four/two response replicates and 16 evaluation episodes
per replicate. This is historical-study completion and development evaluation;
the intervals are descriptive seed-cluster bootstrap intervals. Longer training
is not assumed to produce a stronger opponent. Policies use the documented
finite-horizon pooled-RGB MLP PPO adapter.}
\label{fig:general-melting-adaptive}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/Candidate_rank_changes.pdf}
\caption{Observed changes in candidate ranking from backtest to deployment.
Each column is a training-seed panel. Colors show backtest rank minus deployment
rank, using average ranks for ties. Deployment ranks use independent audit sample
means, so individual changes are not claims of statistically confirmed latent
rank reversals. MPE2 shows the new short-response panels; Melting Pot shows the
completed long-response stress test.}
\label{fig:general-candidate-ranks}
\end{figure*}

\begin{table}[t]
\centering
\small
\begin{tabular}{lrr}
\hline
Task & PIVOT allocation $-$ random & Holm $p$ \\
\hline
'''
    for task in ['push','adversary']:
        t=mpe['tables'][task];r=t['contrasts'][0]['gain_difference'];low,high=r['ci95']
        text+=f"{task} & ${r['mean']:.4f}\\;[{low:.4f}, {high:.4f}]$ & ${t['primary_holm_p']:.4f}$ \\\\\n"
    text+=r'''\hline
\end{tabular}
\caption{Prespecified MPE2 allocation comparison after exactly two paid paired
queries in each arm. We use the author's EVSI query rule without early stopping
to isolate allocation; the original stopping policy is retained in the full
benefit--cost curves. Intervals bootstrap paired training-seed differences;
the approximate $t$ tests are Holm-adjusted across two tasks. These are
eight-seed pilot results, not a claim of universal dominance.}
\label{tab:general-mpe-allocation}
\end{table}
'''
    (b/'benchmark_figures_and_table.tex').write_text(text)
    print('LaTeX figure captions and measured table exported')
if __name__=='__main__':main()
