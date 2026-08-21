# V6 Theory Empirical Evidence

Updated 2026-08-21. This note records the registered, reproducible checks for
the two V6 claims that are small enough to test exactly before adding any new
application domain.

## Claims and constructions

**Global Fidelity Blindness (GFB).** The true policy values are equally spaced
on `[0, 1]`; the verifier swaps one adjacent pair. The operator samples only
the affected replacement transition. As the family grows, global MAE and rank
deficit shrink, while every operator transition has the wrong improvement
sign.

The operator samples define the empirical transition law `Q_A`. For a loss
`L`, the reported operator-relative estimand is
`IF(V, A; L) = E_{Q_A}[L(Delta_V, Delta_*)]`; the artifact records the
absolute-delta IF (IDE) and sign-error IF (`1 - ISC`) explicitly.

**Response-Footprint Sensitivity (RFS).** The response map is
`M_lambda(x) = lambda x` and the value functional is
`J(x, m) = x - L_J m`. This gives an exact check of
`|Delta_actor - Delta_direct| <= L_J L_M d`.

These are transparent theorem checks. They do not identify causal market
impact, strategic response, or the fidelity of a learned simulator.

## Registered run

```bash
.venv/bin/python experiments/e10_theory_empirical.py \
  --config configs/theory/v6_empirical.yaml \
  --output results/theory/e10-theory-empirical
```

The output directory is hash-indexed by `manifest.json` and contains the
configuration, provenance, row-level JSONL data, CSV summary, and two figures.
The recorded code commit is the parent repository commit that generated the
run; no raw vendor data or credentials are involved.

## Results

The full grid has 105 GFB rows (7 policy-family sizes, 3 epsilon targets, 5
seeds) and 180 RFS rows (6 response strengths, 6 footprints, 5 seeds).

| Check | Result |
| --- | ---: |
| GFB epsilon targets covered (`0.1`, `0.01`, `0.001`) | 3/3 |
| Smallest passing family for `epsilon = 0.001` | 64 |
| Minimum Spearman correlation | 0.9970588 |
| Maximum global MAE | 0.0083333 |
| Minimum operator-conditioned IRR | 1.0 |
| Maximum operator IDE | 0.1333333 |
| RFS bound-holding rows | 180/180 |
| Maximum observed/bound ratio | 1.0000000000000033 |
| Maximum bound violation | `1.11e-16` (floating-point tolerance) |

The result supports the narrow statement that global value/rank fidelity can
be arbitrarily close while an update operator remains systematically blind to
the affected local transition. It also verifies that the response-footprint
error bound is tight for the stated scalar construction.

## Reproducibility gates

- Focused unit and integration tests: `5 passed`.
- Ruff on the new module and tests: clean.
- Mypy on `src/pivot/theory`: clean.
- Independent artifact validation: `valid: true`, no manifest errors.
- Figures were visually checked at high resolution for labels, legends, and
  axis/title overlap.
