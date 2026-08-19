# PIVOT 项目文档

本目录记录已经冻结的 ICLR 2027 研究方向、实施路线和当前实现状态。P0-P9
研究 harness 已落盘并通过本地验证；注册 fixture gate 和七个公开市场 session
的观察性执行审计已单独记录，但不等于论文级结论。完整主规格保存在
`master-goal.md`，它优先于较早的讨论稿。

核心冻结项：

- 研究问题：自我改进 agent 的更新 `pi -> pi'` 在环境因自身部署而改变后，是否仍然是进步。
- 核心现象：Improvement Reversal，即 `Delta_V > 0` 但 `Delta_* < 0`。
- 方法：PIVOT（Paired Interventional Verification of Optimization Transitions）。
- 环境阶梯：World 0 Observer -> World 1 Actor -> World 2 Strategic；金融内部为 F0 -> F1 -> F2 -> F3/F4。
- 实施顺序：P0 -> P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P8 -> P9 -> P10；先证明 estimand，再做预算方法，最后才接 LLM/EvoQuant 与 F3 world model。
- 时间约束：按本轮记录的 ICLR 2027 目标，摘要截止 2026-09-18 AOE，全文截止 2026-09-25 AOE，主文最多 9 页。

- Design/specification: [2026-08-19-pivot-design.md](superpowers/specs/2026-08-19-pivot-design.md)
- Master implementation plan: [2026-08-19-pivot-master-implementation.md](superpowers/plans/2026-08-19-pivot-master-implementation.md)
- Superseded planning draft: [2026-08-19-pivot-implementation.md](superpowers/plans/2026-08-19-pivot-implementation.md)
- Gate ledger: [gates.md](experiments/gates.md)
- Public finance evidence and F2 amendment: [public-finance-evidence-2026-08-19.md](experiments/public-finance-evidence-2026-08-19.md)
- Clean-room reproduction evidence: [clean-room-evidence-2026-08-19.md](experiments/clean-room-evidence-2026-08-19.md)
- Public expansion protocol/evidence: [public-expansion-protocol.md](experiments/public-expansion-protocol.md), [public-expansion-evidence-2026-08-19.md](experiments/public-expansion-evidence-2026-08-19.md)
- Requirement coverage: [master-goal-coverage.md](master-goal-coverage.md)
- Research question: [research_question.md](research_question.md)
- Estimands and metrics: [estimands.md](estimands.md)
- Claim boundary: [claim_boundary.md](claim_boundary.md)
- Experiment protocol: [experiment_protocol.md](experiment_protocol.md)
- Theory notes: [theory_notes.md](theory_notes.md)
- Reproducibility contract: [reproducibility.md](reproducibility.md)
- ICLR execution schedule: [iclr2027-execution-schedule.md](iclr2027-execution-schedule.md)

项目范围锁定为一个核心现象、一个统计对象、一个方法和一条环境阶梯；M3 与 LLM/EvoQuant 只作为后置扩展。任何实验结果都必须先进入 transition-level artifact 和 gate ledger，才能进入论文叙事。当前公开 percentage-depth 结果只支持 execution-proxy 审计，不支持 causal endogenous-response claim。
