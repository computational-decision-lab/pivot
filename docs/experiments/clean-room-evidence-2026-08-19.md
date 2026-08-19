# Clean-room Reproduction Evidence

This record covers a fresh-process reproduction from commit
`0ba4117ecfd4362ab46aa112ccb3a9b3cf4490c0`. It is an artifact-integrity
record, not a promotion of any scientific gate.

## Run roots

- Clean-room root: `/tmp/pivot-cleanroom-0ba4117.YRtIvj`
- Registered outputs: `p2`, `e4`, `e5`, `e6`, `e7`, `e8`, `e9`
- Independent combined gate output: `f-summary.json`
- Fresh public archive cache: `public-data/binance-um-btcusdt-2023-01-01-07-v1`
- Figure output: `figures`

The registered runner created 21 isolated runs: three each for P2, E4, E5,
E6, E7, E8, and E9. Every `run_manifest.json` reports `status=ok` and
`exit_code=0`. E3 also completed its 12-round run independently.

## Aggregate hashes

```text
p2-summary.json       70c068bf804ccac9daa1e9fe2c88420a8a43da7fb008d5a61bc8005e53028e6f
e4-summary.json       cf3b3468e5328482664325c9ed647393a72c6cc6098262c130063a6142ef3dc0
e5-summary.json       fe170c6b4967838bf6041c97db004ddb1a6f83adca5ac10d88ffb8d2e9a43280
e6-summary.json       cffc95db92cbd1a2ea085a983b636a3d5162745a108ca4199a2167320b4a4c25
f-summary.json        a30f893f19ed291cce20e681db3b2123ed70af4d324bb991e92041f7ebb9c3f3
e9-summary.json       53242c101635bfcd7cc1cdb50f40fb271b3785e25e7631f6e055326b8f31700e
e3/overoptimization   289ea0c34c81abfcc632938e51a743fc352833ab9e60dad243a29d59588a13b9
figures/validation    2cb0689339dfb936d75b6d51c9b7b681f65ab0322b6852318e86e53a40b5e2e2
```

The public calibration was run twice: once against the existing verified
cache and once against a fresh cache downloaded into the clean-room root. The
fresh run produced the same rows and summary as the first run.

```text
e6-public-fresh/summary.json       ee50047c89461f569c5647891106a40364e99389f0eb85760ad6b880e6a87aa2
e6-public-fresh/public_finance_rows.json
                                   be812e2aef8ec08fd4b31ff7110a5d1a03fa924c48dc919cfbf99ed0bd0c17a8
e6-public-fresh/provenance.json    7c32b5a699760e35c31b2dae4beac8173b9a71df39afce714758290a92e8f767
```

## Fresh public acquisition

The acquisition command downloaded 14 archives, reused 0, and wrote
`live_orders=false`. All 14 actual SHA-256 values equal their manifest
values. The manifest hash is
`0bac42a27be747e208fedc01e42fb72cfa5743ad74370354ad233f647ea78046`; the
fresh receipt hash is
`501ae067983fe08dd4ff0ada2c3393777b58380eaa89e5fb794d43436d1535e7`.

The public archive remains an observational execution source. Re-running from
a fresh cache verifies acquisition, parsing, pairing, and deterministic output;
it does not turn percentage depth into a causal endogenous-response oracle.

## Figure validation

`make_paper_figures.py` consumed only the clean-room source tree and reported
`validated figures=7`. Each canonical figure has a PNG and adjacent CSV source
table. No figure was copied from the repository's `results/` directory.

## Reproduction commands

```bash
for pair in \
  "p2 configs/registered/p2.yaml" \
  "e4 configs/registered/e4.yaml" \
  "e5 configs/registered/e5.yaml" \
  "e6 configs/registered/e6.yaml" \
  "e7 configs/registered/e7.yaml" \
  "e8 configs/registered/e8.yaml" \
  "e9 configs/registered/e9.yaml"; do
  set -- $pair
  .venv/bin/python scripts/run_registered.py --registry "$2" \
    --output "/tmp/pivot-cleanroom-0ba4117.YRtIvj/$1"
done

.venv/bin/python scripts/aggregate_registered.py --experiment f --mode adaptive \
  --inputs /tmp/pivot-cleanroom-0ba4117.YRtIvj/e7/e7-r01 \
           /tmp/pivot-cleanroom-0ba4117.YRtIvj/e7/e7-r02 \
           /tmp/pivot-cleanroom-0ba4117.YRtIvj/e7/e7-r03 \
  --e8-inputs /tmp/pivot-cleanroom-0ba4117.YRtIvj/e8/e8-r01 \
              /tmp/pivot-cleanroom-0ba4117.YRtIvj/e8/e8-r02 \
              /tmp/pivot-cleanroom-0ba4117.YRtIvj/e8/e8-r03 \
  --output /tmp/pivot-cleanroom-0ba4117.YRtIvj/f-summary.json

.venv/bin/python scripts/make_paper_figures.py \
  --input /tmp/pivot-cleanroom-0ba4117.YRtIvj \
  --output /tmp/pivot-cleanroom-0ba4117.YRtIvj/figures
```

The public acquisition and calibration commands are recorded in
`public-finance-evidence-2026-08-19.md`. Use a new output root for every run;
the registered runner intentionally refuses to overwrite non-empty evidence.
