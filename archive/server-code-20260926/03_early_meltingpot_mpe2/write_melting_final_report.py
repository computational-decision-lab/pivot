"""Human-readable report of completed frozen evidence; includes nulls and controls."""
import argparse
import csv
import json
from pathlib import Path


def stat(s):
    return f"{s['mean']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    s=json.loads((a.analysis/'summary.json').read_text());assert s['seed_count']==30
    with (a.analysis/'frontier.csv').open(encoding='utf-8-sig') as f:frontier=list(csv.DictReader(f))
    m=s['confirmation_mechanism'];effects=s['primary_effects'];cost=s['accounting']
    mech=m['own_gain_long_minus_short']['ci95'][0]>0 and m['gap_squared_long_minus_short']['ci95'][0]>1
    method=s['method_contrasts_supported']
    conclusion=('本轮在限定实验范围内通过了预先规定的完整机制与方法检验。' if s['status']=='CHAIN_SUPPORTED_IN_THIS_SCOPE' else
                '本轮没有通过预先规定的完整机制与方法检验。')
    names={'proxy_only':'只看 proxy','calibrated_no_hf':'校准后直接选择，不查询 HF','pivot_sequential':'PIVOT，固定查询预算',
           'pivot_sequential_adaptive_stop':'PIVOT，逐次更新并允许停止','uniform_random_matched':'Uniform/Random HF，共享估计器',
           'global_ivr_matched':'Global IVR，共享估计器','posterior_lucb_matched':'后验 best/challenger 启发式，共享估计器',
           'author_random_hf':'作者 Random HF','author_paired_lucb':'作者 Paired LUCB 变体','author_global_voi':'作者 Global-VOI 变体',
           'author_pivot_voi':'作者 PIVOT-VOI batch','all_hf_reference':'All-HF 参考'}
    lines=['# Melting Pot 正式结果与交付说明','',conclusion,'',
        f"机制确认：{'通过' if mech else '未通过'}；PIVOT 相对共享估计器 Uniform 的主要方法与交互检验：{'通过' if method else '未通过'}。这两个判断分开报告，不用其中一个替代另一个。",'',
        '## 我做了什么','',
        '我先停止 MPE2 的追加实验并保留 null/negative，然后修复 Melting Pot 的种子重放问题。用固定的 4 局/32 局响应学习比较机制，12 个新种子的机制检查通过后，冻结 8 个候选更新、HF 预算和分析规则，再用 30 个完全新的根种子比较方法。',
        '',f"正式实验完成 {cost['native_episodes']:,} 局、{cost['native_frames']:,} 个原生环境帧；前一阶段校准/机制实验另有 {cost['calibration_native_episodes']:,} 局，不能合并成 42 个方法确认种子。",'',
        '我用的是官方固定 stag/hare 技能网络，上层的 responder 根据真实游戏回报学习技能混合概率。候选是提前固定的技能混合概率编辑，包含不更新选项；没有端到端重训 PPO，也没有用 LLM 生成这批候选。底层执行顺序补丁已明确记录，因此材料称为 Melting Pot adaptive extension。',
        '', '## 机制是否再次成立','',
        '| 新 30 个种子的配对比较 | 均值及 95% 区间 |','|---|---:|',
        f"| 对手自身回报：long−short | {stat(m['own_gain_long_minus_short'])} |",
        f"| 噪声修正平方 gap：long−short | {stat(m['gap_squared_long_minus_short'])} |",'',
        '平方 gap 使用独立 A/B 评估块的乘积估计，单位为回报平方。区间的独立单位是根种子，候选和对局没有被当成额外独立训练样本。',
        '', '## PIVOT 是否更会选','',
        '主要 HF 上限为 192 局。下表正数表示 PIVOT 相对共享估计器的 Uniform/Random HF 选得更好；交互项正数表示这种优势在 long 下更大。',
        '', '| 预先规定的比较 | 配对 regret reduction 及 95% 区间 |','|---|---:|',
        f"| Short：Uniform − PIVOT ISR | {stat(effects['short_regret_reduction'])} |",
        f"| Long：Uniform − PIVOT ISR | {stat(effects['long_regret_reduction'])} |",
        f"| Long−short 的方法优势变化 | {stat(effects['interaction'])} |",'',
        '这些方法差异等于独立审计下已选更新的收益差，共同的带噪声最大值会相消。因此主要结论不依赖把某个测量最高的候选当成精确 oracle。',
        '', '## 收益来自哪一部分','',
        '| 诊断对比：前者减后者 | Short | Long |','|---|---:|---:|']
    labels={'calibration_value':'校准、不查询 − proxy','beyond_calibration':'PIVOT（允许停止）− 校准、不查询',
            'fixed_budget_acquisition':'固定预算 PIVOT − 共享估计器 Uniform','stopping_value':'允许停止的 PIVOT − 固定预算 PIVOT',
            'author_batch_mixed_estimator':'作者 batch PIVOT − 作者 Random'}
    for key,label in labels.items():
        pair=[next(x for x in s['registered_diagnostics'] if x['contrast']==key and x['adaptation']==h) for h in [4,32]]
        lines.append(f"| {label} | {stat(pair[0])} | {stat(pair[1])} |")
    lines += ['', '这些是预先列出的诊断对比，不是多重检验校正后的独立成功声明。如果 PIVOT 仅胜过 proxy，却不胜过“校准、不查询”，主要证据是校准价值；不能据此说主动查询有贡献。作者 batch 对比还包含未查询候选估计器不同的问题。',
              '', '## 所有方法的主要预算结果','',
              '原始 ISR 使用带噪声的独立审计最大值，仅用于描述。All-HF 是参考项，可以超过 192 局上限，不是无噪声真值。以下保留所有方法，不按胜负筛选。','',
              '| 方法 | Short ISR / HF 局数 | Long ISR / HF 局数 |','|---|---:|---:|']
    for key,label in names.items():
        values=[]
        for h in [4,32]:
            rows=[r for r in frontier if int(r['adaptation'])==h and r['method']==key and (r['budget_cap']=='192' or key in ['proxy_only','calibrated_no_hf','all_hf_reference'])]
            assert len(rows)==1;row=rows[0];values.append(f"{float(row['mean_audit_isr']):.3f} / {float(row['mean_hf_episode_cost']):.1f}")
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    lines += ['', 'HF 局数是冻结的逻辑成本：有查询时，incumbent 响应学习 H 局计一次，每个候选查询计 H+32 局。真实模拟通过方法之间共享缓存、short/long 共享训练前缀减少重复，物理总量另行记录。proxy、校准与独立审计成本单列，并非免费。长响应下每个查询更贵，同一 192 局上限最多查询 2 个候选，short 下最多 5 个。因此交互项描述固定模拟预算下两个响应条件的方法差异，不单独识别 gap 的因果中介作用。',
              '', '## 图和复核','',
              '- `mechanism_and_method.png`：左侧分别看响应学习质量、gap；右侧看主要方法对比和组件诊断。',
              '- `hf_cost_vs_regret.png`：横轴按 protocol 计入方法预算的 HF 局数，纵轴带噪声的独立审计 ISR；越低越好。',
              '- `primary_contrasts.png`：主要方法检验；区间跨 0 表示本轮没有把该差异与 0 区分开。',
              '- `method_seed_results.csv`、`primary_seed_contrasts.csv`：逐 seed 的选择、成本和配对差异。',
              '- `summary.json`：完整结果、成本、预先规定的判断与置信度诊断。',
              '', '## 论文中能怎样表述','',
              '本实验检验固定官方技能网络、单个 substrate、单轮候选选择，以及条件于各候选冻结响应训练结果的 deployment improvement。审计采用新的环境/网络随机数，但不为每次审计重新训练 responder。30 个根种子覆盖了响应训练的变化。',
              '', '不将本实验扩大为完整 Melting Pot 套件、端到端 PPO、多轮 CISR 或普遍方法优势。LUCB/Global-VOI 的作者实现与本轮额外启发式都有准确名称，不能混称标准算法的所有版本。',
              '', '机制成立但方法未通过时，保留结果并检查方法的估计器一致性、校准和停止置信度，不继续追加 test seeds 或扫设置。代码核查另见 `implementation_audit/方法实现核查.md`；合成数值检查不计入上述原生对局。',
              '',f"冻结 protocol SHA256：`{s['protocol_sha256']}`。",'']
    a.output.write_text('\n'.join(lines))
    print(str(a.output))


if __name__=='__main__':main()
