# PIVOT Gate Ledger

这是实施过程中的证据登记表。没有 run ID、配置快照、seed 列表、查询预算和置信区间的结果，不得标记为通过。

| Gate | 判定内容 | 状态 | 证据 |
| --- | --- | --- | --- |
| A | Controlled Improvement Reversal 在非病态 response/footprint 区域稳定出现 | Not run | P2 smoke 已生成；需独立冻结 runs/CI |
| B | IRR 与 update footprint 和/或 environment response 有结构关系 | Not run | P2 phase table 已生成；需独立聚合/结构检验 |
| C | Global policy fidelity 不能完全替代 local improvement fidelity | Not run | E4 matched-budget smoke 已生成；当前差异需独立聚合 |
| D | PIVOT 在固定 HF budget 下优于 Random/Top Proxy | Not run | E5 calibrated smoke 在小预算改善；需注册 runs/paired CI |
| E | Finance F0/F1/F2 在合理 participation 下有结构差异 | Not run | F2 fixture smoke 已生成；需公开数据校准 |
| F | Strategic response 增加系统性 effect | Not run | E7/E8 paired sweep smoke 已生成；需正式 seed/旋钮聚合 |

## Evidence record

每次更新 gate 时，必须追加一行并填写：

```text
date | gate | status | run_ids | config_hashes | seeds | hf_budget | metric_and_ci | reviewer_note
```

`Not run` 是当前真实状态，不是实验结论。任何 gate 失败都保留失败证据，并在实施计划中记录是缩小 claim、修改环境，还是停止后续阶段。
