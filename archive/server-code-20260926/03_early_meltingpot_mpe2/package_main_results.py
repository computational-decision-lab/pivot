"""Curate existing measured results into a small, self-contained local delivery."""
from pathlib import Path
import csv, datetime, hashlib, json, re, shutil

BASE = Path(__file__).resolve().parent
NEW = BASE / 'general_results_20260915'
OLD = BASE / 'autodl_results_20260915'
OUT = BASE.parent / '论文主要结果_20260915'

def read(p):
    return json.loads(p.read_text())

def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def interval(x):
    return f"{x['mean']:.4f} [{x['ci95'][0]:.4f}, {x['ci95'][1]:.4f}]"

def main():
    mpe = read(NEW / 'fresh_mpe_general/summary.json')
    melt = read(NEW / 'melting_general_long/summary.json')
    metrics = read(NEW / 'general_paper_metrics.json')
    comparison = read(OLD / 'all_v9_comparison.json')
    assert mpe['status'] == melt['status'] == metrics['status'] == 'complete'
    assert read(NEW / 'general_benchmark_verification.json')['status'] == 'verified'
    OUT.mkdir(exist_ok=False)
    for d in ['图表', '数据', '数据/原文复现源汇总']:
        (OUT / d).mkdir(exist_ok=True)
    sources = {}

    def copy(src, relative):
        dest = OUT / relative
        shutil.copy2(src, dest)
        assert hashlib.sha256(src.read_bytes()).digest() == hashlib.sha256(dest.read_bytes()).digest()
        sources[relative] = str(src)

    for name in ['MPE2_general_benchmark', 'MeltingPot_general_benchmark', 'Candidate_rank_changes']:
        for ext in ['png', 'pdf']:
            copy(NEW / 'figures' / f'{name}.{ext}', f'图表/{name}.{ext}')
    for source, dest in {
        'fresh_mpe_general/summary.json': 'MPE2汇总.json',
        'melting_general_long/summary.json': 'MeltingPot汇总.json',
        'fresh_mpe_general/method_seed_results.csv': 'MPE2逐种子.csv',
        'melting_general_long/method_seed_results.csv': 'MeltingPot逐种子.csv',
        'general_paper_metrics.json': '论文指标及区间.json',
        'general_benchmark_verification.json': '完整性核验.json',
        'fresh_mpe_general/protocol.json': 'MPE2实验协议.json',
        'fresh_mpe_general/global_selection_seal.json': 'MPE2选择冻结记录.json',
        'fresh_mpe_general/source_hashes.json': 'MPE2源码哈希.json',
        'fresh_mpe_general/environment.json': 'MPE2环境版本.json',
        'melting_general_long/protocol.json': 'MeltingPot实验协议.json',
    }.items():
        copy(NEW / source, '数据/' + dest)
    copy(OLD / 'all_v9_comparison.json', '数据/原文数值比对.json')
    copy(OLD / 'e2_hash_order_diagnostic.json', '数据/E2区间顺序敏感性诊断.json')
    author_paths = {
        'E2': 'remaining_v9/e2c_confirmatory/operator_shift_summary.json',
        'E3': 'job_confirmatory/confirmatory/e3c/closed_loop_summary.json',
        'E4': 'remaining_v9/e4c_confirmatory/ood_summary.json',
        'E5': 'job_confirmatory/confirmatory/e5c/efficiency_summary.json',
        'E7': 'job_confirmatory/confirmatory/e7c/strategic_summary.json',
    }
    author = {k: read(OLD / p) for k, p in author_paths.items()}
    for k, p in author_paths.items():
        copy(OLD / p, f'数据/原文复现源汇总/{k}.json')

    rows = []
    for task in ['push', 'adversary']:
        t = mpe['tables'][task]
        for method in ['author_batch_no_stop', 'random_batch']:
            selected = next(r for r in t['methods'] if r['method'] == method and r['budget'] == 2)
            assert selected['mean_queries'] == 2
        primary = t['contrasts'][0]
        assert primary['budget'] == 2 and primary['contrast'] == 'author_batch_no_stop minus random_batch'
        rows.append({'平台': 'MPE2', '任务': task, '训练种子数': 8, '候选数': 32,
                     '每分支响应训练步数': 8192,
                     '响应效应均值': t['mechanism']['response_effect']['mean'],
                     '候选特异效应均值': t['mechanism']['candidate_specific_effect']['mean'],
                     '响应者自身收益变化': t['mechanism']['response_own_reward_gain']['mean'],
                     '作者固定2查询减随机查询': primary['gain_difference']['mean'],
                     '主比较区间下限': primary['gain_difference']['ci95'][0],
                     '主比较区间上限': primary['gain_difference']['ci95'][1],
                     '两任务Holm校正p': t['primary_holm_p'],
                     '回测正而部署估计负的个数': metrics['tables']['MPE2/' + task]['proxy']['observed_IRR']['numerator'],
                     '回测正的候选个数': metrics['tables']['MPE2/' + task]['proxy']['observed_IRR']['denominator'],
                     '状态': '全新训练；选择冻结后生成独立审计'})
    for task in ['melting_pd', 'melting_stag']:
        for steps in [32768, 131072]:
            t = melt['tables'][task]['mechanism_by_response_budget'][str(steps)]
            q = metrics['tables'][f'MeltingPot/{task}/{steps}']['proxy']['observed_IRR']
            rows.append({'平台': 'Melting Pot', '任务': task, '训练种子数': 6, '候选数': 24,
                         '每分支响应训练步数': steps,
                         '响应效应均值': t['response_effect']['mean'],
                         '候选特异效应均值': t['candidate_specific_effect']['mean'],
                         '响应者自身收益变化': t['response_own_reward_gain']['mean'],
                         '作者固定2查询减随机查询': '', '主比较区间下限': '', '主比较区间上限': '', '两任务Holm校正p': '',
                         '回测正而部署估计负的个数': q['numerator'], '回测正的候选个数': q['denominator'],
                         '状态': '历史实验补齐；同一冻结选择跨响应预算评估；策略质量有限'})
    write_csv(OUT / '外部Benchmark主结果.csv', rows)
    rows = []

    def add_author(experiment, label, mean, low, high, scope):
        rows.append({'实验': experiment, '指标': label, '均值': mean, '区间下限': low,
                     '区间上限': high, '结论范围': scope})

    a = author['E2']
    add_author('E2C', '算子偏移增加的绝对升级预测误差', a['shift_effect_mean'], a['shift_effect_ci_low'], a['shift_effect_ci_high'],
               '受控效应数值复现；两个区间端点有遍历顺序敏感性')
    for name, label in [('pivot_voi_minus_proxy_only', '对Proxy Only的累计选择损失减少'),
                        ('pivot_voi_minus_global_voi', '对Global-VOI的累计选择损失减少')]:
        a = author['E3']['effects'][name]
        add_author('E3C', label, a['mean'], a['ci_low'], a['ci_high'], '作者40轮受控实验；正数有利于PIVOT；沿用作者汇总区间')
    a = author['E4']
    add_author('E4C', '差分评估器减全局评估器的ISC', a['transition_minus_global_isc_mean'], a['transition_minus_global_isc_ci_low'], a['transition_minus_global_isc_ci_high'],
               '复现原文负结果；不支持差分评估器优势')
    a = author['E5']
    add_author('E5C', '注册预算下相对Proxy Only的选择损失减少', a['pivot_voi_minus_proxy_cisr_reduction'], a['pivot_voi_minus_proxy_ci_low'], a['pivot_voi_minus_proxy_ci_high'],
               '平均效应很小；不代表所有任务或所有查询规则')
    a = author['E7']
    add_author('E7C', '公式适应机制的战略响应效应', a['adaptive_effect_mean'], a['adaptive_effect_ci_low'], a['adaptive_effect_ci_high'],
               '有限公式响应机制；不是神经网络对手训练或均衡证据')
    write_csv(OUT / '原文复现主结果.csv', rows)

    # Preserve the measured report while making the delivery relocatable.
    detailed = (BASE / '两套General_Benchmark结果_20260915.md').read_text()
    replacements = {str(NEW / 'figures'): '图表',
                    str(NEW / 'fresh_mpe_general/method_seed_results.csv'): '数据/MPE2逐种子.csv',
                    str(NEW / 'melting_general_long/method_seed_results.csv'): '数据/MeltingPot逐种子.csv',
                    str(NEW / 'general_paper_metrics.json'): '数据/论文指标及区间.json',
                    str(NEW / 'general_benchmark_verification.json'): '数据/完整性核验.json'}
    for old, new in replacements.items():
        detailed = detailed.replace(old, new)
    detailed = detailed.replace('还需改善观察/记忆结构并重新检验训练质量', '还需先诊断并改善训练，再检验策略质量；观察与记忆结构是待检查因素')
    (OUT / '完整Benchmark结果.md').write_text(detailed)
    copy(BASE / 'General_Benchmark_实验说明.md', '实验设置.md')
    latex = (NEW / 'benchmark_figures_and_table.tex').read_text().replace('figures/', '图表/')
    (OUT / '论文图表片段.tex').write_text(latex)
    counts = sum(x['numeric_fields'] for x in comparison)
    unmatched = sum(x['numeric_difference_count'] for x in comparison)
    mech = mpe['tables']['adversary']['mechanism']['candidate_specific_effect']
    readme = f'''# 主要结果与论文结论对照

更新：2026-09-15。这里整理的是已经完成的本地实验产物，没有新启动云端训练，也没有复制模型权重或大体积轨迹。

## 先给结论

**这些结果支持论文限定范围内的部分结论，并补充了外部环境证据；尚不能说 PIVOT 在两个 general benchmark 上稳定优于简单验证。**

证据分两层：第一层是用作者代码重跑受控实验，检查原有数值能否复现；第二层是我们增加的 MPE2 PPO 自适应扩展及 Melting Pot 选定任务扩展。数值复现、方法能运行、机制信号和方法优势是不同层面的结果。

原文第 1、2、7 页明确限制普遍优势、均衡与市场因果结论。因此“未证明全面领先”不与原文发生矛盾；它也不能被反过来当作已证明方法有效的证据。

## 主要文件

- [外部 benchmark 主表](外部Benchmark主结果.csv)：两个平台的任务、响应效应、固定成本比较和观察到的符号变化。
- [原文复现主表](原文复现主结果.csv)：五项实验的核心数值，包含负结果。
- [完整 benchmark 结果及三组图](完整Benchmark结果.md)：全部方法、预算和区间说明。
- [实验设置](实验设置.md)：候选、PPO、响应分支、样本划分和成本口径。
- [论文图表片段](论文图表片段.tex)：可随图表目录一起放入论文项目的 LaTeX 片段。

CSV 使用带 BOM 的 UTF-8 编码，可用 Excel 打开。表中空白表示未进行该项相同成本比较，不表示数值为零。原始奖励量纲不同，不把跨平台的增益直接平均。

## 到底支撑哪条结论

| 论文原主张与位置 | 本次证据 | 支持程度 |
|---|---|---|
| 更新应在部署响应后评价；回测增益不自动等于部署增益（第 1、2、7 页） | 新 MPE2 Adversary 的候选特异响应效应 {interval(mech)}；14 个回测为正的候选中 3 个部署审计均值为负 | **初步机制支持。** 机制区间未按所有探索比较校正；3/14 是含噪均值的符号计数，不能逐个宣布真实反转。 |
| 算子偏移可增加局部更新误差（第 5、6 页） | E2C 点估计 0.2627 复现 | **支持所测受控结果的可复现性。** 两个 bootstrap 端点存在顺序敏感性；新 benchmark 没有另做系统的算子偏移实验。 |
| PIVOT 在指定受控比较中可减少闭环选择损失（第 6 页） | E3C 相对 Proxy Only 减少 1.4883，区间 [1.0618, 1.9105]；相对 Global-VOI 为 -0.2530 | **复现了有限场景有效及方法排序边界。** 新外部实验是单次四候选面板，不能替代作者 40 轮闭环结果。 |
| 差分评估器未必优于全局评估器（第 6、7 页） | E4C 的 ISC 差 -0.2590，区间 [-0.3588, -0.1707]，与原文负结果一致 | **负结果得到复现。** 不支持注册的差分学习器优势，也不等于两者等效。 |
| 对手响应可增加负面战略影响，但只对所测机制负责（第 6、7 页） | E7C 的公式机制效应 -0.0240 得到复现；新 MPE2 Adversary 有负向候选特异信号，Push 则接近零 | **受控结果复现，外部有限补强。** 不能推出任何对手学习都会让升级失效，也不能把有限 PPO 响应叫作最优回应或均衡。 |
| 配对后验及预算查询规则可以作为可执行的验证流程（第 2、5 页） | 原作者选择入口已接入两套平台；新 MPE2 冻结 464 次方法×预算选择，再生成 512 组独立审计；成本核验通过 | **支持工程可行性和可检查性。** 校准依赖任务特征与噪声估计；能执行不自动说明收益更高。 |
| PIVOT 在新增 benchmark 中稳定领先 | MPE2 固定两次查询时，作者无停止查询规则减随机查询：Push 0；Adversary +0.0270，Holm p=0.5165 | **当前未获支持。** 此项是我们检验的外部优势问题，原文没有作普遍领先承诺。 |

## 两套 benchmark 的成熟度

**MPE2：目前可作为主要的外部 pilot 结果。** 两个任务各 8 个新训练种子，共 64 个新候选。初始训练质量检查为正；查询数据和审计数据隔离，所有选择在生成新审计前冻结。Adversary 回测与部署符号一致率为 56.25%，Push 为 87.5%，显示所测任务之间的差异。全局策略排序是否仍优秀没有在这批新实验中同步验证，因此不能把这些数据称为“全局排序优秀但局部判断失效”的新完整反例。

**Melting Pot：目前是开发性与探索性补实验。** 两个任务各 6 个已有训练种子，共 48 个候选，现有 192 组短期和 96 组长期审计；本轮补齐了其中 64 组长期响应。168 个先前固定的方法×预算选择在两种响应预算下评分，共 336 行，不能称为 336 次新选择。部分历史审计曾被查看，不称为全新盲测。策略未显示稳定超过未训练基线，结论精度和策略质量都需要改善。

MPE2 已在原论文 Figure 5 的冻结外部参考中出现。本次增加的是 PPO 自适应响应、独立审计、机制对照与预算分析；不能把 MPE2 称作原文从未用过的新平台。两套平台都只选了两个任务，未完成全部官方任务套件。

## 没有检验的范围

- 六条理论命题及其证明未由这些 benchmark 重新证明，有限实验也不能证明普遍定理。
- 新增外部实验没有测多轮策略继承、长期累计 CISR 或拒绝升级；决策空间仅含四个更新候选。
- 未完成以 LLM 生成候选为核心的新 benchmark，也没有新增真实市场因果、环境均衡或普适最优回应证据。
- 本包不把图表表现力当作结论强度。符号和排名使用独立评估的样本均值，个体反转可能包含噪声。

## 原文复现的可靠性边界

五项原实验共有 {counts:,} 个数值字段，其中 {counts-unmatched:,} 个在容差内一致。E2C 的两个区间端点不同，已保存无序遍历影响 bootstrap 的诊断；原始输出没有被诊断值覆盖。参考产物提交为 c5c9496…，本次固定代码为 9e3be72…，存在代码版本及元数据差异。

这是对作者实现和保存产物的数值复现，不是独立证明所有统计假设成立。E2/E4 等汇总区间沿用作者计算单位，不能统一称为独立训练种子 bootstrap。本包保存原汇总供复核，不把它们改写成新的独立统计分析。

## 可以对组里这样概括

我们复现了论文五项受控实验的主要数值，并完成两套平台的选定任务自适应扩展。新增实验能够测量更新在对手响应后的价值、排名和验证成本，给论文的核心问题补充外部证据。当前没有证据说明 PIVOT 普遍优于简单验证；MPE2 可作为主要 pilot 结果，Melting Pot 暂作探索性补充，进一步提升策略质量后再增强其论证权重。

## 复核与来源

小型数据、逐种子 CSV、冻结记录和源摘要在“数据”目录。三组图各提供 PNG 预览和 PDF 矢量版，图中没有删去不利任务或预算。[原文数值比对](数据/原文数值比对.json)、[新增实验完整性核验](数据/完整性核验.json)、[E2 区间诊断](数据/E2区间顺序敏感性诊断.json)可直接检查。

论文依据：用户提供的 Colin_ICLR.pdf，第 1、2、5、6、7 页及 Figure 5；原文件位于 {BASE.parent.parent / 'Colin_ICLR.pdf'}。文件 SHA-256 为 8a81842a67ac1d82d61a2c18e9bda69820cbdcc22602d54b56d08da63a89925c。本地已有原文件，本包不重复复制。
'''
    (OUT / '先看这里.md').write_text(readme)
    # Relative links remain valid if the folder is moved or shared.
    for path in OUT.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if target.startswith(('http:', 'https:', '#')):
                continue
            assert (path.parent / target).exists(), (path, target)
    files = sorted(p for p in OUT.rglob('*') if p.is_file())
    manifest = {
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'purpose': 'Small local evidence package; completed outputs only; no new training',
        'model_weights_included': False,
        'paper_sha256': '8a81842a67ac1d82d61a2c18e9bda69820cbdcc22602d54b56d08da63a89925c',
        'files': {str(p.relative_to(OUT)): {'bytes': p.stat().st_size,
                  'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                  'copied_from': sources.get(str(p.relative_to(OUT)))} for p in files},
    }
    (OUT / '文件校验.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    files = [p for p in OUT.rglob('*') if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    assert size < 5_000_000
    print(json.dumps({'folder': str(OUT), 'files': len(files), 'bytes': size,
                      'all_links_valid': True, 'copied_file_hashes_match': True}, ensure_ascii=False))

if __name__ == '__main__':
    main()
