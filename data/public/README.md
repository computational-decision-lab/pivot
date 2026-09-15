# Public Finance Cache

This directory is a local, reproducible cache for checksum-pinned public
market-data archives. Binary archives and acquisition receipts are ignored by
Git. Fetch them with:

```bash
python scripts/fetch_public_finance.py \
  --manifest configs/data/binance_btcusdt_um_2023-01-01_07.yaml \
  --output-root data/public
```

The tracked YAML manifest records the provider URLs and expected SHA-256
digests. These observations are replay inputs only. They are not live feeds,
real fills, or ground truth for endogenous market response.

The frozen multi-asset expansion uses three manifests and a separate cache
root:

```bash
for manifest in \
  configs/data/binance_btcusdt_quarter_2023.yaml \
  configs/data/binance_ethusdt_quarter_2023.yaml \
  configs/data/binance_bnbusdt_quarter_2023.yaml; do
  .venv/bin/python scripts/fetch_public_finance.py \
    --manifest "$manifest" \
    --output-root data/public/public-expansion-v1
done
```

Each manifest has four predeclared calendar sessions. The acquisition command
revalidates existing files and refuses to replace a corrupt cache entry.
