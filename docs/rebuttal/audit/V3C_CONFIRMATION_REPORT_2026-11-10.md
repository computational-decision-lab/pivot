# V3c Confirmation Report

The confirmation amendment removes the common sentinel so that all three
selectors receive the same two sequential method-specific HF queries per
round. The contextual prior, proposer function, response world, terminal
posterior-mean selector, seed schedule, calibration cohort, and five-round
horizon are fixed across methods. The first-round candidate panel is shared;
after deployment, each method's proposer is applied to its own incumbent, so
later panels may diverge. The confirmation uses roots 960000, 970000, and
980000, 120 per response regime, after a 50-root development run on
860000--880000. See `P1_LOOP_V3C_PROTOCOL_CLARIFICATION_2026-11-10.md` for the
post hoc wording clarification; the hash-bound protocol is unchanged.

The primary method, PIVOT-KG-Loop-v3c, has mean five-round cumulative ISR
0.00888, 0.00141, and 0.08886 at low, medium, and high response. IVR has
0.05574, 0.44046, and 0.98532; Top-Proxy has 0.07095, 0.23241, and
0.33346. Lower ISR is better. The six paired baseline-minus-PIVOT contrasts
have positive 95% bootstrap lower bounds: IVR [0.0385, 0.3706, 0.6449] and
Top-Proxy [0.0491, 0.1805, 0.2019], ordered low/medium/high. No sampled sign
flip was as extreme as the observed contrast in 100,000 randomizations. The
plus-one Monte Carlo estimate is approximately 1e-5 per test, and the
Holm-adjusted estimates are approximately 6e-5 across the six prespecified
contrasts; these are resolution-limited estimates rather than exact
probabilities.

The secondary PIVOT-minus-baseline cumulative deployment-return contrasts are also
positive in all six comparisons, with lower bounds [0.0647, 0.3666, 0.9790]
against IVR and [0.0489, 0.1429, 0.1457] against Top-Proxy. The analyzer's
joint acceptance rule therefore passes in all three response regimes.

The independent analyzer recomputed the calibration model and replayed 5,400
round ledgers, 10,800 HF queries, 5,400 independent evaluation panels, and
5,400 state-carrying deployment returns. All seed, budget, noisy-conditioning,
state-transition, ISR recomputation, and aggregate-trajectory checks pass.

Calibration pressure support is [0.054, 0.696]. Among the 1,800 test states per
regime, 0, 148 (8.2%), and 423 (23.5%) are outside that range for low, medium,
and high response. By method, the out-of-support fractions are 0/600 in low
response; 0/600, 76/600, and 72/600 for PIVOT, IVR, and Top-Proxy at medium;
and 83/600, 163/600, and 177/600 at high response. This support diagnostic
indicates contextual extrapolation on some trajectories and belongs in the
interpretation of the result.

The result supports a multi-round claim only for the explicitly defined
PIVOT-KG-Loop-v3c extension in this controlled world. It does not establish
universal superiority, physical-driving performance, or superiority of the
static submitted Algorithm 1 in arbitrary self-improvement systems. Fresh
confirmation roots are conditional on the selected protocol: Holm covers the
six prespecified within-v3c contrasts, not the preceding v1--v4a variant
search. The `Online` labels for IVR and Top-Proxy refer only to sequential
within-round posterior conditioning; v3c resets the correction model each
round and does not perform persistent online calibration. The multidimensional
support audit finds that `round_fraction` is constant at zero in calibration,
so its later-round effect is not identified; the full feature-range, distance,
residual, and interval diagnostics are included in
`V3C_MULTIDIM_SUPPORT_REPORT_2026-11-10.md`.
