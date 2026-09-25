"""Create the benchmark delivery report only from completed measured results."""
import argparse,datetime,json
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def n(x):return f'{x:.4f}'
def fmt(x):return f"{n(x['mean'])} [{n(x['ci95'][0])}, {n(x['ci95'][1])}]"
def ci(x):
    x=np.array(x,float);rng=np.random.default_rng(20260915);s=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    return {'mean':float(x.mean()),'ci95':np.quantile(s,[.025,.975]).tolist()}
def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();b=a.results.resolve()
    mpe=read(b/'fresh_mpe_general/summary.json');mel=read(b/'melting_general_long/summary.json');metrics=read(b/'general_paper_metrics.json');verify=read(b/'general_benchmark_verification.json')
    assert all(d['status']=='complete' for d in [mpe,mel,metrics]);assert verify['status']=='verified'
    assert all((b/'figures'/name).exists() for name in ['MPE2_general_benchmark.png','MeltingPot_general_benchmark.png','Candidate_rank_changes.png'])
    lines=['# 两套 general benchmark 补实验结果','',
      '本轮按会议目标完成 MPE2 自适应扩展和 Melting Pot 自适应扩展两套平台的选定任务。主交付是原论文方法在外部环境中的完整实验、响应机制和收益—成本图，不以 PIVOT 必须获胜作为完成条件。',
      '', '| 平台 | 本轮实际完成 |', '|---|---|',
      f"| MPE2 | 两任务 × 8 个新训练种子，64 个新候选；256 组 selection 配对响应、512 组独立 audit；{mpe['n_decisions']} 次方法×预算选择；新增 {mpe['total_physical_environment_steps']:,} 环境步 |",
      f"| Melting Pot | 两任务 × 6 个已有训练种子，48 个候选；补齐 64 组长期配对响应；合计 192 组短期与 96 组长期审计；168 个已冻结作者选择在两种响应预算下评分 {mel['n_scored_author_decisions_across_two_horizons']} 次；本轮新增 {mel['new_physical_environment_steps']:,} 环境步 |",
      '', 'MPE2 的候选和响应者本轮重新进行 PPO 训练。Melting Pot 复用已有策略、完成更长期的 PPO 对手响应。Melting Pot 的部分历史审计已被查看，本轮属于研究补齐与开发性评估；新 MPE2 独立审计则在所有选择冻结后才生成。两者都是选定任务上的适应扩展，不是完整官方评测套件；环境步也不能跨平台直接当作等价算力。',
      '', '## 作者 PIVOT 的结果与成本', '',
      'MPE2 主分配对照固定为作者 VOI 查询规则与随机查询各实际支付 2 次配对验证；不停止版本用来隔离查询分配。原作者带停止规则的完整移植结果另见全部预算曲线。正差值表示作者查询分配得到的部署增益更高。', '',
      '| MPE2 任务 | 作者查询 − 随机查询的部署增益差，95% 区间 | 两任务 Holm 校正 p |', '|---|---|---|']
    for task in ['push','adversary']:
        t=mpe['tables'][task];c=t['contrasts'][0];assert c['contrast']=='author_batch_no_stop minus random_batch' and c['budget']==2
        lines.append(f"| {task} | {fmt(c['gain_difference'])} | {t['primary_holm_p']:.4f} |")
    lines+=['','Push 的零差值来自这 8 个面板上两种规则选中了相同候选，不能推出算法普遍等效。Adversary 的比较只有 2 个面板改变选择，不能据非负的样本均值宣称稳定优势。这些是每任务 8 个新种子的 pilot 结果。区间和检验用于表明结果精度；不据单次显著性宣称普遍领先。', '',
      'Melting Pot 的原作者选择在短期响应下已冻结。下面展示同一选择在不同响应训练预算下的表现，PIVOT 和随机验证的预算上限均为 2 个包（每包 2 次配对响应）；由于停止规则，实际成本可能不同，不能将其称为严格等成本对照。', '',
      '| Melting Pot 任务 | 短响应 PIVOT 部署增益 | 长响应 PIVOT 部署增益 | 短响应平均实际查询包数 |', '|---|---|---|---|']
    for task in ['melting_pd','melting_stag']:
        rows=mel['tables'][task]['methods'];short=next(r for r in rows if r['method']=='author_pivot_voi' and r['budget_packages']==2 and r['response_steps']==32768);long=next(r for r in rows if r['method']=='author_pivot_voi' and r['budget_packages']==2 and r['response_steps']==131072)
        lines.append(f"| {task} | {fmt(short['audit_gain'])} | {fmt(long['audit_gain'])} | {short['mean_query_packages']:.3f} |")
    lines+=['','## 训练质量检查','',
      '这里比较已训练的 incumbent 与未训练策略，二者面对同一个初始对手、使用配对评估种子。正数表示训练后的收益更高；不同平台奖励尺度不同，不能横向比较大小。', '',
      '| 任务 | 训练后 − 未训练的收益差，95% 区间 |', '|---|---|']
    for task in ['push','adversary']:
        lines.append(f"| MPE2 {task} | {fmt(mpe['tables'][task]['mechanism']['focal_quality_gain'])} |")
    for task in ['melting_pd','melting_stag']:
        lines.append(f"| {task} | {fmt(mel['tables'][task]['focal_quality_gain'])} |")
    lines+=['', 'MPE2 的这项检查显示训练有效。Melting Pot 这批策略没有显示稳定超过未训练策略，因此其结果适合作为方法迁移和响应预算的开发性实验；若要写成强策略之间的交互证据，还需改善观察/记忆结构并重新检验训练质量。延长对手训练不能替代这项检查。', '',
      '## 部署以后发生了什么','',
      '三项机制分开报告：更新价值的响应效应；只改变响应训练对象所产生的候选特异效应；响应者自身的收益变化。MPE2 使用 8,192 步响应，Melting Pot 下表使用 131,072 步长期响应。', '',
      '| 任务 | 更新价值的响应效应 | 候选特异响应效应 | 响应者自身收益变化 |', '|---|---|---|---|']
    for task in ['push','adversary']:
        t=mpe['tables'][task]['mechanism'];lines.append(f"| MPE2 {task} | {fmt(t['response_effect'])} | {fmt(t['candidate_specific_effect'])} | {fmt(t['response_own_reward_gain'])} |")
    for task in ['melting_pd','melting_stag']:
        t=mel['tables'][task]['mechanism_by_response_budget']['131072'];lines.append(f"| {task} | {fmt(t['response_effect'])} | {fmt(t['candidate_specific_effect'])} | {fmt(t['response_own_reward_gain'])} |")
    lines+=['','机制区间是描述性的，未按所有机制比较校正。对手进行了学习更新，不自动意味着其能力提高，更不意味着焦点升级必然失效。候选散点和排名热图中的变化也包含评估噪声。', '',
      '## 与论文定义对齐的指标','',
      '下面使用廉价回测预测。IDE 是升级增益的平均绝对预测误差，ISC 是增益符号一致率，IRR 是预测为正的升级中部署估计为负的比例。部署值来自独立 audit 均值，故这些是含噪估计；IRR 不是经逐候选统计确认的真实有害升级率。', '',
      '| 任务与响应步数 | IDE | ISC | IRR（样本计数） |', '|---|---|---|---|']
    for key,t in metrics['tables'].items():
        q=t['proxy'];r=q['observed_IRR'];s=q['observed_ISC'];rate='未定义' if r['rate'] is None else f"{r['rate']:.1%} ({r['numerator']}/{r['denominator']})";isc='未定义' if s['rate'] is None else f"{s['rate']:.1%}"
        lines.append(f"| {key} | {q['estimated_IDE']['mean']:.4f} | {isc} | {rate} |")
    lines+=['','完整区间及共享后验修正后的同组指标保存在 general_paper_metrics.json。所有统计以训练种子为聚类单位；没有把数万条轨迹当作数万个独立样本，也没有把单次升级面板的结果叫作多轮累计 CISR。', '',
      '## 图表','',f"![MPE2 benchmark]({b/'figures/MPE2_general_benchmark.png'})",'',f"![Melting Pot benchmark]({b/'figures/MeltingPot_general_benchmark.png'})",'',f"![候选排名变化]({b/'figures/Candidate_rank_changes.png'})",'',
      '三组图提供 PNG 预览与 PDF 矢量版本，含全部预定任务和预算。排名图用观察到的样本均值排序，不能据颜色将单个候选宣布为真实反转。', '',
      '## 这部分可以作为怎样的贡献','',
      '这是一组把论文的升级验证框架应用到两套通用交互平台的补实验：公开候选更新、配对响应、独立评估与实际查询成本，展示回测判断、部署响应和验证决策之间的关系。更强的实验叙述应来自上述跨任务机制与成本曲线；方法胜出、无差别或更差都按原样报告。', '',
      '顺序后验更新、相关误差校准和固定成本消融作为附加实现保存，不替代原作者基线，也不把已有 knowledge-gradient 原理重新命名为新理论。[Frazier、Powell 与 Dayanik（2009）](https://people.orie.cornell.edu/pfrazier/pub/CorrelatedKG.pdf)', '',
      '样本量仍是 MPE2 每任务 8 个、Melting Pot 每任务 6 个训练种子。本批决策空间为四个更新候选；不升级选项与多轮策略继承尚未评测。Melting Pot 策略是局部 5×5 pooled RGB 的无记忆 MLP PPO，训练质量和有限时域限制了外推；两套实验也不能直接推广为所有 LLM self-improving agents 的结论。', '',
      '## 小型证据文件','',
      f"- [MPE2 每种子结果 CSV]({b/'fresh_mpe_general/method_seed_results.csv'})",
      f"- [Melting Pot 每种子结果 CSV]({b/'melting_general_long/method_seed_results.csv'})",
      f"- [论文指标与区间]({b/'general_paper_metrics.json'})",
      f"- [完整性核验]({b/'general_benchmark_verification.json'})",'',
      '模型权重和大体积轨迹留在服务器。本地只保存精简证据、图表与报告。']
    out=b.parent/'两套General_Benchmark结果_20260915.md';out.write_text('\n'.join(lines)+'\n');print(out)
if __name__=='__main__':main()
