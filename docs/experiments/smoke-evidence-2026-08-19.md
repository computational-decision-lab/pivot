# Smoke Verification Record

This record documents implementation verification only. It is not a scientific
gate pass and must not be quoted as an independent paper result.

## Run

- Commit: `99b4525fdaae73904384775b989aed32ec0dfdd7`
- Clean output root: `/tmp/pivot-final-committed.tNBX0W`
- Python: 3.10.12
- P2 seeds: `1,2,3,4,5`
- E7 seeds: `1,2,3`
- E8 seeds: `1,2,3`
- E9 HF budget: `1` per round, `8` rounds, `8` queries
- P2 manifest SHA-256: `6eaa589bf4a3469679c06ae387699b0c7bfb8bee16b03eddce5015877122a153`

Configuration SHA-256 values:

```text
configs/sweeps/p2.yaml                 9432e4214bdd672cbb5f293b9017949b9af92138315d801ee7564667e6475d7f
configs/sweeps/e4.yaml                 e2a7011bf13edc8332d651913b8f9e0e641c6ae3a1641c3ac2ea5af17deec122
configs/sweeps/e5.yaml                 c2f61c70066b6bb057ccef6a4ee1695bc29021fb35fa068116591024d3d5f398
configs/finance/f2_participation.yaml  38ac68ba3ae2070ec82fc4022cf76df0743f522466028595d59283991a7bf7ec
configs/sweeps/e7.yaml                 65e78519a3c524e9eb891436e90e48356db9b53b2c9efec5533d3c57357914e0
configs/sweeps/e8.yaml                 892cddcaa7ee92c856d29c072245f21c23ecf32dbe1189cef447e90202650490
configs/sweeps/e9.yaml                 2fbebccf7b9833c32e1277fe3f32f74ff33ed98bb38040e96cdda5b7db2977b0
```

## Artifact checks

| Stage | Result |
| --- | --- |
| P2 controlled sweep | 240 transition rows; JSONL + Parquet; IRR `0.7291667` |
| E3 overoptimization | 12 rounds persisted |
| E4 matched-budget transfer | 24 HF transitions; disjoint train/test IDs; global/local/boosted ISR emitted |
| E5 budget frontier | 27 records, 48 held-out groups, all seven baseline labels present |
| E6 F0/F1/F2 | 15 paired rows; F2 equals F1 at participation `0.0` and crosses below zero by `0.05` in the fixture |
| E7 strategic reversal | 3/3 paired rows satisfy `delta_proxy > 0`, `delta_actor > 0`, `delta_strategic < 0` |
| E8 competition sweep | 90 rows covering fixed/reactive/adaptive modes, opponent counts, steps, rates, and sensitivities |
| E9 closed loop | 8 rounds, 24 transition rows, 8 HF query rows |
| Figure bundle | 7/7 PNGs and source CSVs validated |

At E5 budget `1`, the smoke means were:

```text
Random HF:    ISR 4.67646298
Top Proxy HF: ISR 3.37218099
PIVOT:        ISR 0.24290896
```

These numbers are fixture smoke diagnostics only. Formal Gates A-F remain
`Not run` until independent registered jobs, paired intervals, and the
finance/strategic calibration protocol are completed.

## Reproduction command

From the repository root, run the commands in the experiment order listed in
`docs/experiment_protocol.md`; use fresh output directories because transition
stores refuse to overwrite an existing run.
