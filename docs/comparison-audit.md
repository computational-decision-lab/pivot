# What the PIVOT-KG versus Uniform comparisons change

The first Leduc, Kuhn and Melting Pot results in Table 1 compare **query
allocation with matched posterior updating and terminal selection**. It would
be misleading to describe these three comparisons as differing in the entire
selection rule while implying that posterior or terminal selection also differ.
The outcome measured is final selection regret; the controlled method
difference is acquisition within that shared decision procedure.

## First cohorts: source and saved decisions

`evidence/paper/source/core/analyze_v4.py:149` constructs the same posterior
specification for each method, using the same proxy vector, observation bank,
HF costs, budget and root seed. The model is fitted once per response horizon
on disjoint calibration roots (`:347`).

`evidence/paper/source/core/pivot_v2.py:165` chooses KG per cost for `pivot_kg`;
`:177` chooses a random unqueried candidate for `uniform_v2`. Both execute the
same Gaussian `post.condition(j, value)` at `:197`, then use posterior means
including no-update and the same final `argmax` at `:200-204`. Uniform receives
the calibrated posterior and its observations update all correlated candidates.
The comparison is not against an uncalibrated or proxy-only Uniform baseline.

The first cohorts retain one frozen Uniform allocation draw per root. This is
not the exact-subset expectation used in later experiments. Root-bootstrap CIs
condition on the fitted calibration model and do not include calibration-set
sampling uncertainty.

| Table 1 cohort, long response | Roots | HF queries each | PIVOT ISR | Uniform ISR | Uniform minus PIVOT ISR (saved 95% CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| Leduc | 30 | 2 | 0.00926866 | 0.04575370 | 0.03648504 [0.01446961, 0.06368512] |
| Kuhn | 30 | 2 | 0.05602424 | 0.05798809 | 0.00196385 [0, 0.00579334] |
| Melting Pot | 30 | 2 | 0.83599977 | 1.74246318 | 0.90646341 [-0.78428382, 3.40062847] |

`python reproduction/audit_comparisons.py --output OUTPUT.json` replays all
180 primary method/root decisions from saved posterior specifications and
selection banks. It checks queried candidates, final selections, posterior
estimates, query counts, selected gains, ISR, and the reported paired means.
All replayed posterior estimates matched exactly in release verification.
The CIs above are read from the archived bootstrap summaries; this audit does
not claim a new bootstrap or a new simulator experiment.

## Other cohorts must be interpreted separately

- **Second Leduc:** `source/leduc_v4/code/pivot_v3.py` shares posterior updating
  and terminal selection, but the primary PIVOT rule uses a candidate/query-size
  menu and can query a candidate repeatedly; Uniform uses fixed-size queries.
  The primary contrast therefore combines allocation and sample-size/repetition
  choices. Its component diagnostics belong to this second cohort and are not
  a decomposition of the three Table 1 rows.
- **Highway redesign:** `source/highway_redesign/source/experiments/revision/highway_b4_redesign.py:93-135`
  builds the same Gaussian posterior for every Uniform subset and conditions
  it before terminal selection. Uniform averages all feasible subsets. This
  matches the posterior used by sequential PIVOT in the redesign runner.
- **Earlier Highway original/replication:** the older runners use observed
  values plus proxy estimates for Uniform terminal selection and a calibrated
  posterior for PIVOT. Those earlier comparisons do evaluate different complete
  validation-and-selection rules; they are not the current redesign result.
- **MetaDrive first cohort:** `metadrive/code/selector_analysis.py` invokes the
  shared `pivot_v2` posterior and selector. Expected Uniform averages 100
  allocation draws per root. This averaging adds no independent test roots.

These distinctions are based on the frozen delivered source and saved
decisions, not on labels such as “PIVOT” or “Uniform” alone.
