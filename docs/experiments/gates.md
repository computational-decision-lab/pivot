# PIVOT Gate Ledger

这是实施过程中的证据登记表。没有 run ID、配置快照、seed 列表、查询预算和置信区间的结果，不得标记为通过。

| Gate | 判定内容 | 状态 | 证据 |
| --- | --- | --- | --- |
| A | Controlled Improvement Reversal 在非病态 response/footprint 区域稳定出现 | Not run | 等待 P2/E1 |
| B | IRR 与 update footprint 和/或 environment response 有结构关系 | Not run | 等待 P2/E2 |
| C | Global Fidelity 不能完全替代 Improvement Fidelity | Not run | 等待 P3/E4 |
| D | PIVOT 在匹配 HF budget 下优于 Random HF 与 Top Proxy HF | Not run | 等待 P4-P5/E5 |
| E | Finance F0/F1 与 F2 在可辩护 participation 范围有结构差异 | Not run | 等待 P6-P7/E6 |
| F | Strategic response 超出 mechanical response，产生系统性效应 | Not run | 等待 P9/E7-E8 |

## Evidence record

每次更新 gate 时，必须追加一行并填写：

```text
date | gate | status | run_ids | config_hashes | seeds | hf_budget | metric_and_ci | reviewer_note
```

`Not run` 是当前真实状态，不是实验结论。任何 gate 失败都保留失败证据，并在实施计划中记录是缩小 claim、修改环境，还是停止后续阶段。
