# PIVOT 项目文档

本目录记录已经冻结的 ICLR 2027 研究方向和实施路线。当前只做文档与规划落盘，尚未声称任何实验 gate 已通过。

核心冻结项：

- 研究问题：自我改进 agent 的更新 `pi -> pi'` 在环境因自身部署而改变后，是否仍然是进步。
- 核心现象：Improvement Reversal，即 `Delta_V > 0` 但 `Delta_* < 0`。
- 方法：PIVOT（Paired Interventional Validation of Optimization Transitions）。
- 环境阶梯：Observer -> Actor -> Strategic。
- 实施顺序：先 controlled performative environment，再 paired evaluator、差分模型和预算分配，最后才接金融、竞争对手、EvoQuant/LLM 与 M3。
- 时间约束：按本轮记录的 ICLR 2027 目标，摘要截止 2026-09-18 AOE，全文截止 2026-09-25 AOE，主文最多 9 页。

- Design/specification: [2026-08-19-pivot-design.md](superpowers/specs/2026-08-19-pivot-design.md)
- Implementation plan: [2026-08-19-pivot-implementation.md](superpowers/plans/2026-08-19-pivot-implementation.md)
- Gate ledger: [gates.md](experiments/gates.md)

项目范围锁定为一个核心现象、一个统计对象、一个方法和一条环境阶梯；M3 与 LLM/EvoQuant 只作为后置扩展。
