# Public Expansion Evidence

This record reports the frozen multi-asset public audit from
`configs/data/public_expansion_grid.yaml`. It is an observational execution
audit and a negative/limiting result for reversal, not a causal actor-world
validation.

- Verified implementation commit:
  `301422c29f96db11ba200b9b1729bcd75eb44342`.
- Protocol freeze and date-key amendment are documented in
  `public-expansion-protocol.md`.

## Frozen inputs

- Grid: `public-um-3asset-quarter-starts-v1`.
- Assets: `BTCUSDT`, `ETHUSDT`, `BNBUSDT`.
- Calendar blocks: `2023-01-01`, `2023-04-01`, `2023-07-01`, `2023-10-01`.
- Holdout blocks: `2023-07-01` and `2023-10-01`, fixed before persistent
  acquisition, parsing, or outcome inspection. A prior network probe checked
  only URL availability and byte size.
- Update: `position_size 0.2 -> 0.6`, intensity `1.0`, bias `0.0`.
- Primary participation: `0.01`; impact multiplier: `1.0`.
- All participation rates: `0`, `0.001`, `0.005`, `0.01`, `0.05`.
- All impact multipliers: `0.5`, `1.0`, `2.0`.
- 12 asset/date pairs, 24 checksum-bound archives, 6,262,942 bytes.
- All 24 manifest hashes match the corresponding official `.CHECKSUM` files.

The acquisition cache is ignored by Git. The tracked manifests bind every
archive to the official [Binance public-data repository](https://github.com/binance/binance-public-data)
and [archive](https://data.binance.vision/).

## Execution result

Run root: `/tmp/pivot-public-expansion-301422c`.

- Complete grid: `true`.
- Parsed rows: `180`.
- Primary sessions: `12`.
- Failed assets: `0`; failure ledger: `[]`.
- F1-positive primary sessions: `7`.
- Primary pooled depth mechanical effect: `-4.1554292821e-7`, bootstrap 95% CI
  `[-5.3969299082e-7, -2.9637992191e-7]`.
- Primary depth-proxy reversal: `0/7` (`0.0`) among F1-positive sessions.
- Frozen holdout: `6` sessions, `5` F1-positive, reversal `0/5`.
- At every predeclared participation rate under multiplier `1.0`, reversal was
  `0/7` among F1-positive sessions.

The negative mechanical effect is consistent across BTCUSDT, ETHUSDT, and
BNBUSDT, but the depth proxy does not produce an Improvement Reversal in this
frozen public grid. This is evidence against promoting the finance reversal
claim from the observational audit, not evidence that reversal is impossible
in a causal interactive market.

## Provenance hashes

```text
grid config              83baa98440c78d3df5c1c1e0a92303e00fe0b435d5a312720ca0d85ef7348a36
BTC manifest             9bbb15e73262b91f0b3b48d0dbd29692d64f91865db6579f9316232b67b8dfe8
ETH manifest             2283de00571d3ede2ed89e61e4c2ea4239b19df007ee6e34804014d87101f1b2
BNB manifest             abc9ce36ce97d1d030ba967e40cd587843c5bbed086fb5904275d40a0e42823b
expansion config         e5dbd458df3f4286517e925c1d662571c3e3ce74ee724d8199829450f19f16e5
BTC subconfig            2f69e088866813d5bfdc27a15268e516e6045dd12e6460a04f5fce8641f732d7
ETH subconfig            87b9b441b5511fa0ce671ed0492b42a647f34ad2d03cbe84d4923ad58046f3b7
BNB subconfig            0d1082629dbec7452e71459cd2acf00cfa1422d34878345b718b6022975265c8
summary                  66e2539b36ffa48674b070ddc18683a3af474ad6140ba8737ed0efb6a7afaaae
rows                     bd131c38f1027d851f4fb2bc64d809c3939dda499194397d70771a04496de8c9
provenance               aafd589de8b8eac24e930ac90bf7006fb0002de6785f62d85cb715e006783d79
failure ledger           4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945
```

## Interpretation boundary

Every row carries `causal_impact_identified=false`; the run also records
`live_orders=false`. Percentage depth is used as a contemporaneous liquidity
curve and cannot identify replenishment, hidden liquidity, post-trade response,
impact recovery, or strategic adaptation. The observed paths therefore do not
support a causal F2 claim or a universal finance conclusion.

The holdout is reported separately and was not used to tune the update,
participation grid, or impact multiplier. The primary update remains an
exploratory typed edit until a future confirmatory update-generation rule is
preregistered.

## Reproduction

```bash
.venv/bin/python experiments/e6_public_expansion.py \
  --config configs/finance/e6_public_expansion.yaml \
  --output results/raw/e6-public-expansion
```

The command validates all three subconfigs against the frozen grid, revalidates
all archive checksums, preserves failures, and refuses to declare a complete
grid unless every asset/date pair is present.
