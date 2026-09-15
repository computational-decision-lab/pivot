# Public Expansion Protocol

This protocol freezes the next public-data audit before inspecting any market
outcome. It expands the existing single-asset, seven-session audit to a fixed
grid of three USD-M futures symbols and four calendar blocks. The grid is
recorded in `configs/data/public_expansion_grid.yaml`.

## Protocol integrity amendment

The initial freeze is commit `7a95c62`; its grid SHA-256 is
`ad2156924e6bc37727c25a9b519e03db99b2415d6328b7d01c73b4b5ff4ef30e`.
During the first aggregate run, PyYAML parsed the four unquoted `audit_roles`
keys as date objects, so the holdout lookup returned zero sessions. Commit
`301422c` quotes those four keys and adds a regression test; the resulting grid
hash is `83baa98440c78d3df5c1c1e0a92303e00fe0b435d5a312720ca0d85ef7348a36`.

This amendment changes only YAML key typing. Assets, dates, audit roles,
update, participation rates, multipliers, execution assumptions, and acceptance
rules are unchanged. The 180 transition rows before and after the fix have the
same SHA-256,
`bd131c38f1027d851f4fb2bc64d809c3939dda499194397d70771a04496de8c9`.
Only the holdout summary/provenance was corrected. The pre-amendment summary is
not valid evidence.

## Frozen design

- Assets: `BTCUSDT`, `ETHUSDT`, `BNBUSDT`.
- Dates: `2023-01-01`, `2023-04-01`, `2023-07-01`, `2023-10-01` for every asset.
- Primary calendar holdout: `2023-07-01` and `2023-10-01`; no configuration,
  update, or threshold may be changed after inspecting those blocks.
- Update: one typed `position_size` edit, `0.2 -> 0.6`, with fixed intensity
  `1.0` and bias `0.0`.
- Participation grid: `0`, `0.1%`, `0.5%`, `1%`, `5%`.
- Impact multipliers: `0.5`, `1.0`, `2.0`; primary multiplier `1.0`.
- Execution assumptions: `0.5` bps spread, `4.0` bps fee, `0.5` bps
  slippage, full virtual fill, queue depth `1.0`.

The primary pooled estimand is the mean depth-proxy mechanical effect
`Delta_F2_depth - Delta_F1` at one percent participation and multiplier `1.0`.
The primary reversal estimand is the conditional depth-proxy reversal rate
among sessions with positive F1 improvement. Asset and calendar-block results
are secondary diagnostics. All predeclared pairs are included; no session is
removed because its outcome is inconvenient. Parser or coverage failures stay
in the failure ledger.

## Claim boundary

This is an observational execution audit, not causal validation of an actor
world. Percentage book depth is used to construct a contemporaneous liquidity
curve; it does not identify replenishment, hidden liquidity, post-trade
response, or strategic adaptation. Every result must retain
`causal_impact_identified: false` and `live_orders: false`.

The update and primary holdout were frozen before persistent acquisition,
parsing, or outcome inspection. Before the freeze, a network-only availability
probe checked the preselected URLs and byte sizes; it did not retain or parse
the archives. The audit can strengthen execution plausibility and test
heterogeneity across assets and calendar blocks. It cannot, by itself, promote
Gate E to a causal or universal claim.

## Acceptance rules

1. All 12 asset/date pairs are attempted.
2. Every acquired archive matches its manifest SHA-256.
3. The update, participation grid, execution assumptions, and primary
   multiplier are identical across all runs.
4. The primary aggregate is computed without selecting sessions by outcome.
5. Holdout results are reported separately and never used to tune the update.
6. Any missing or malformed archive is retained as a failure and blocks a
   complete-grid statement.
