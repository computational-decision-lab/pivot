# V10 Number and Provenance Audit

Status: **PASS**

The LaTeX macros are recomputed from the frozen confirmatory JSON artifacts; no experiment is rerun.

| Macro | Actual | Expected | Pass |
| --- | ---: | ---: | --- |
| `OperatorShiftSeeds` | `30` | `30` | True |
| `OperatorShiftRows` | `36000` | `36000` | True |
| `OperatorShiftEffect` | `0.2627` | `0.262693` | True |
| `OperatorShiftEffectCI` | `(0.1684, 0.3763)` | `(0.16837888967779296, 0.3763466806661414)` | True |
| `ClosedLoopSeeds` | `30` | `30` | True |
| `ClosedLoopRounds` | `40` | `40` | True |
| `ClosedLoopEffect` | `1.4883` | `1.48832` | True |
| `ClosedLoopEffectCI` | `(1.0618, 1.9105)` | `(1.0618401589208712, 1.9105004088834245)` | True |
| `OODSeeds` | `30` | `30` | True |
| `OODRows` | `36000` | `36000` | True |
| `OODGain` | `-0.259` | `-0.25904` | True |
| `OODGainCI` | `(-0.3588, -0.1707)` | `(-0.3587730650355345, -0.17067604313768528)` | True |
| `EfficiencySeeds` | `30` | `30` | True |
| `EfficiencyPairedN` | `180` | `180` | True |
| `EfficiencyEffect` | `0.000668` | `0.000667931` | True |
| `EfficiencyEffectCI` | `(1.1e-05, 0.0013)` | `(1.100865090954107e-05, 0.0013294208175714253)` | True |
| `StrategicSeeds` | `30` | `30` | True |
| `StrategicClusters` | `30` | `30` | True |
| `StrategicEffect` | `-0.024` | `-0.0239726` | True |
| `StrategicEffectCI` | `(-0.0249, -0.0231)` | `(-0.024900347450970375, -0.023106334486912433)` | True |
| `StrategicSIRR` | `0.9495` | `0.949493` | True |

## Row counts

- `e2_transition_rows`: 36000
- `e3_transition_rows`: 192000
- `e7_strategic_rows`: 3600

## Warnings

- result prose contains only macro-backed estimates plus figure/table constants:
