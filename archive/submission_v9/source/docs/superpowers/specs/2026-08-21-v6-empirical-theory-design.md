# V6 Empirical Theory Checks

**Status:** approved by the V6 objective supplied for this continuation

## Scope

Add two narrow, deterministic experiments that test the two mathematical
claims in the V6 objective without introducing a new application domain:

1. **Global Fidelity Blindness (GFB).** Construct a finite policy family whose
   normalized deployment values are monotone. Swap the values of one adjacent
   pair only in the proxy. Under a uniform policy distribution, the resulting
   global MAE and Spearman deficit vanish as the family grows, while an
   improvement operator concentrated on the swapped pair sees a proxy-positive,
   deployment-negative transition on every draw. The experiment reports MAE,
   Spearman, operator-conditioned IDE/ISC/IRR, and the global-to-operator error
   ratio for a grid of policy-family sizes and target epsilons. The sampled rows
   are the empirical transition law $Q_{\mathcal A}$, so absolute-delta IF is
   IDE and sign-error IF is $1-\mathrm{ISC}$.
2. **Response-Footprint Scaling (RFS).** Use a scalar response map
   `M_lambda(x)=lambda*x` and value `J(x,m)=x-L_J*m`. For an update footprint
   `d=|x'-x|`, the known regularity constants are `L_M=lambda` and `L_J`.
   The experiment evaluates direct and actor deltas over response/footprint
   grids and checks `|delta_actor-delta_direct| <= L_J*L_M*d` row by row.

## Artifact contract

`experiments/e10_theory_empirical.py` writes a run directory containing:

- `global_fidelity_rows.jsonl` and `response_footprint_rows.jsonl`;
- `metrics.json` with aggregate rows and theorem-gate pass flags;
- `summary.csv` and two publication figures;
- `provenance.json`, `config_snapshot.json`, and a SHA-256 `manifest.json`.

The row-level values are deterministic for a fixed config and seed list;
`provenance.json` additionally records the run timestamp. It is a numerical
test of the stated constructions and regularity bound, not evidence of causal
market response. No external data or credentials are used.

## Tests and acceptance gates

- Unit tests prove the adjacent-swap construction has exactly one reversed
  local transition, decreasing global MAE, and increasing rank fidelity as the
  policy family grows.
- Unit tests prove the response-footprint bound is satisfied and reaches
  equality for the analytic construction.
- Integration tests verify deterministic JSONL/CSV/manifest output and the
  required pass flags.
- The registered run must cover at least seven policy-family sizes, three
  target epsilons, six response strengths, six footprint sizes, and five
  independent seeds. Results are reported as a theorem check, not as a new
  superiority claim.
