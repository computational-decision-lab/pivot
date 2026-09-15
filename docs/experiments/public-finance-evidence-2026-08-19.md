# Public Finance Calibration and F2 Semantics Amendment

This record supersedes the E6 and combined-F numerical outputs in
`registered-evidence-2026-08-19.md`. P2, E4, and E5 are unaffected. The
supersession is necessary because the original F2 fixture charged impact on an
unchanged holding at every bar. Commit `4c1677c` corrects F2 to charge half the
terminal linear impact only when a virtual fill occurs. Transient impact stock
is retained as diagnostic state, not repeatedly deducted as execution cost.

## Frozen code and outputs

- Verified code commit: `4c1677cf71ed88825e902eaff5631da3427ab83c`
- Evidence root: `/tmp/pivot-evidence-4c1677c.qDSzix`
- Public dataset manifest SHA-256:
  `0bac42a27be747e208fedc01e42fb72cfa5743ad74370354ad233f647ea78046`
- Public experiment config SHA-256:
  `69c303d471b694e018ea25431c02e246de4af81dbcc806a6554deabcc3e15848`

Output SHA-256:

```text
public calibration  e035e20030ecb4fed827598af2e82b34c3b91c1740b837997b41bc91a3bb99a8
public rows         be812e2aef8ec08fd4b31ff7110a5d1a03fa924c48dc919cfbf99ed0bd0c17a8
public summary      ee50047c89461f569c5647891106a40364e99389f0eb85760ad6b880e6a87aa2
fixture E6 summary  e103a8f4c01bd1d10a83094cd5a97dfb38712e0d022067f0341da51f694c5a4d
fixture F summary   20dad33cc8ef122f407470601847b1febe9bb74e53c8babae7dc8b8acb36c7be
fixture E9 summary  4cb3a8ab32a6e3a65c827fac5c495d36e905c329e7b8c1cedb94864ffdeea4d7
```

All 12 affected registered runs (`e6-r01..r03`, `e7-r01..r03`,
`e8-r01..r03`, `e9-r01..r03`) completed with status `ok` and exit code `0`.

## Public data boundary

The frozen manifest contains seven BTCUSDT USD-M futures sessions from
2023-01-01 through 2023-01-07. Each session binds a one-minute kline archive and
percentage book-depth archive from the [official Binance public-data
repository](https://github.com/binance/binance-public-data) and
[archive](https://data.binance.vision/) to its published SHA-256. The 14 binary
archives total about 3.6 MB locally and remain ignored by Git.

Kline rows supply the observed price and quote-volume path. Book-depth rows
supply cumulative notional at 1--5 percent on each side. Bars are trimmed to
the already-observed depth interval; 2--13 leading/trailing bars are removed
per session rather than filling missing early depth with future snapshots.

The depth-aware world computes virtual order notional as participation times
observed quote volume times filled policy change, interpolates terminal impact
on the contemporaneous depth curve, and charges half terminal impact under a
linear-depth assumption. This is an auditable execution proxy. It is not a
counterfactual market response and always emits
`ground_truth_for_endogenous_response: false`.

## Public calibration result

The exploratory typed update changes only `position_size`, from `0.2` to
`0.6`, while holding a constant direction signal fixed. Execution assumptions
are recorded, not estimated from the percentage-depth files: 0.5 bps spread,
4.0 bps fee, and 0.5 bps slippage. Five of seven sessions have positive F1
improvement, so F1-to-F2 reversal has a non-empty denominator.

At the primary 1 percent participation:

- zero-participation F2-depth minus F1 is exactly `0`;
- mean F2-depth minus F1 is `-2.6690e-7`, bootstrap 95% CI
  `[-3.9424e-7, -1.6332e-7]`;
- F1-to-F2-depth reversal is `0/5` among F1-positive sessions.

Across the frozen participation sweep, the depth mechanical effect becomes
monotonically more negative. At 5 percent participation, one of five
F1-positive sessions crosses zero, for a conditional depth-proxy reversal rate
of `0.2`. The linearized median-depth proxy produces no sign reversal in the
same sweep.

This is structured observational evidence, not causal validation. Gate E
therefore remains `Fixture Pass; external audit partial; paper promotion
pending`.

## Corrected registered fixture result

After the fill-only impact correction:

- Gate E remains Fixture Pass: zero-participation F2-F1 is `0`; the registered
  5 percent effect is `-0.0002303999863`.
- Gate F remains Fixture Pass: E7 SIRR is `1.0`, E7 competition effect is
  `-0.0513736704`, E8 competition effect is `-0.0405236541`, and the E8
  sensitivity contrast is `-0.1082189302`.
- E8 actor improvement is now positive (`0.00170453706`) before adaptive
  opponent response. E8 SIRR is therefore directly defined and equals `1.0`;
  the prior empty-denominator caveat is superseded.

## Remaining claim blockers

- Seven consecutive sessions and one asset are insufficient for external
  validity; expansion must be frozen before inspection across assets and
  volatility/liquidity regimes.
- Percentage depth identifies an execution curve, not causal replenishment,
  post-trade response, hidden liquidity, or strategic adaptation.
- The public typed update was selected during exploratory protocol refinement;
  a confirmatory update-generation and holdout-selection rule remains to be
  preregistered.
- No live orders, broker connections, real fills, credentials, or private L2
  data were used.

## Reproduction

```bash
python scripts/fetch_public_finance.py \
  --manifest configs/data/binance_btcusdt_um_2023-01-01_07.yaml \
  --output-root data/public

python experiments/e6_public_calibration.py \
  --config configs/finance/e6_public_calibration.yaml \
  --output results/raw/e6-public-calibration
```
