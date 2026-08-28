# Promotion Replay

The external replay is **DEV-only**: `42`
method rows over `2` immutable
candidate batches, with `16`
physical paired evaluations and `0`
logical HF query decisions.  Cache hits are logged per selector, so physical
reuse never silently reduces the registered logical budget.  The post-decision
truth audit is excluded from that budget.  Every method receives the same
candidate-batch hash.
The shared promotion analysis has status `DEV_ONLY`,
terminal state `UNDERPOWERED`, and fixed effect
orientation `proxy_minus_pivot; positive_favors_pivot`.
Its target budget is `4` and
the paired-effect summary is `{"ci_high": 0.0, "ci_low": 0.0, "direction": "proxy_minus_pivot; positive_favors_pivot", "estimate": 0.0, "n_clusters": 2, "n_rows": 2}`.
Confirmatory promotion results: **NOT_RUN**.
