# Frozen Poker v2b sensitivity inputs

This directory is a normalized, read-only copy of the inputs used to make the
Poker prior-sensitivity replay reproducible without the original server paths.

The 12 calibration roots and 30 test roots come from
`实验结果/Leduc_研究成果_20260923/04_早期v2b修正版/Leduc_v2b_修正版图与结果_20260920.zip`.
The frozen long/short posterior comes byte-for-byte from
`paper/pivot_iclr2027_supplementary.zip` at
`evidence/latest/leduc_v2b/posterior_v2_spec.json`. The selector implementation
is the frozen `pivot_v2.py` from the same v2b result package.

The protocol JSON contains the internal field
`protocol_sha256=e86c5c4810e6af7c9c953382cdae746ae4546e7e0552d1f53bbdd1552cd6feef`.
The byte hash of the public protocol file itself is
`b92b5c627f511af517db81ebe10efb9cfe7460c6628bb96f4af3f0a123d51b74`. The prior
replay summary recorded the internal protocol hash as its `input_hashes.protocol`;
the distinction is retained here instead of silently relabeling the file.

Run from the repository root:

```bash
python3 docs/rebuttal/audit/run_poker_sensitivity.py \
  --code docs/rebuttal/audit_inputs/poker_v2b/code \
  --calibration docs/rebuttal/audit_inputs/poker_v2b/calibration \
  --test docs/rebuttal/audit_inputs/poker_v2b/test \
  --output results/rebuttal/20261109-p0-poker-sensitivity-input-replay-v1 \
  --bootstrap-draws 1000
```

The normalized replay should reproduce the sealed decision hash
`281df6083c2f26eb336535f1a371a9b3595f276411c4b67222eae9b9ddedec9f`.
The analysis is an offline replay over saved decision banks; it does not run
OpenSpiel or create new rollouts.
