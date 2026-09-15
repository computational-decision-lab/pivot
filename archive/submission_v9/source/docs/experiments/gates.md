# PIVOT Gate Ledger

这是实施过程中的证据登记表。没有 run ID、配置快照、seed 列表、查询预算和置信区间的结果，不得标记为通过。

| Gate | 判定内容 | 状态 | 证据 |
| --- | --- | --- | --- |
| A | Controlled Improvement Reversal 在非病态 response/footprint 区域稳定出现 | Fixture Pass; paper promotion pending | 注册 P2 `p2-r01..r03`；high-response IRR CI `[0.9722, 0.9722]` |
| B | IRR 与 update footprint 和/或 environment response 有结构关系 | Fixture Pass; paper promotion pending | 注册 P2 response contrast CI `[0.9722, 0.9722]` |
| C | Global policy fidelity 不能完全替代 local improvement fidelity | Fixture Pass; paper promotion pending | 注册 E4 ISC(local-global) CI `[0.1204, 0.1204]`; ISR difference tied |
| D | PIVOT 在固定 HF budget 下优于 Random/Top Proxy | Fixture Pass; paper promotion pending | 注册 E5 budget=1 paired ISR gains: Random `4.4401..4.8694`, Top Proxy `3.1282..3.1322` |
| E | Finance F0/F1/F2 在合理 participation 下有结构差异 | Fixture Pass; public audit null for reversal; paper promotion pending | 修正后注册 E6 zero-equivalence exact; expanded BTC/ETH/BNB 12-session depth proxy pooled effect `-4.1554e-7` (95% CI `[-5.3969e-7,-2.9638e-7]`); reversal `0/7` at 1%, holdout `0/5` |
| F | Strategic response 增加系统性 effect | Fixture Pass; paper promotion pending | 修正后注册 E7/E8 SIRR 均为 `1.0`; E8 actor delta `0.00170454`; effect `-0.04052`; sensitivity contrast `-0.10822` |

## Evidence record

每次更新 gate 时，必须追加一行并填写：

```text
date | gate | status | run_ids | config_hashes | seeds | hf_budget | metric_and_ci | reviewer_note
```

`Fixture Pass` 只表示注册配置下的预注册判定通过，不等于跨环境或论文最终结论。任何 gate 失败都保留失败证据，并在实施计划中记录是缩小 claim、修改环境，还是停止后续阶段。

2026-08-19 的 F2 fill-only semantics amendment、公开数据边界、配置哈希和
重跑结果记录在 `public-finance-evidence-2026-08-19.md`。旧 E6/F 数值不得用于
当前代码的论文 claim。

多资产扩展的冻结网格、子配置契约、12-session 结果和 holdout 分析记录在
`public-expansion-protocol.md` 与 `public-expansion-evidence-2026-08-19.md`。
