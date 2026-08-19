# PIVOT Gate Ledger

这是实施过程中的证据登记表。没有 run ID、配置快照、seed 列表、查询预算和置信区间的结果，不得标记为通过。

| Gate | 判定内容 | 状态 | 证据 |
| --- | --- | --- | --- |
| A | Controlled Improvement Reversal 在非极端 response/footprint 区域稳定出现 | Not run | 等待 Tasks 1-6 |
| B | Global Fidelity 不能完全替代 Improvement Fidelity | Not run | 等待 Task 7 |
| C | PIVOT 在匹配 HF budget 下优于 Random HF 与 Top Proxy HF | Not run | 等待 Task 8 |
| D | Finance participation sweep 的 actor 差异具备可辩护范围和跨 session 稳定性 | Not run | 等待 Task 10 |
| E | Strategic reversal 具有系统性均值效应，而非只有方差增加 | Not run | 等待 Task 11 |

## Evidence record

每次更新 gate 时，必须追加一行并填写：

```text
date | gate | status | run_ids | config_hashes | seeds | hf_budget | metric_and_ci | reviewer_note
```

`Not run` 是当前真实状态，不是实验结论。任何 gate 失败都保留失败证据，并在实施计划中记录是缩小 claim、修改环境，还是停止后续阶段。
