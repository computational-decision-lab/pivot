# PIVOT Core

This component name denotes the agent-agnostic protocol layer.  The
maintained implementation is exposed as the `pivot_core` facade in
`src/pivot_core/` and delegates to the tested `src/pivot/` modules.

It contains policy transitions, paired evaluators, update footprints, the
differential posterior, and EVSI-per-cost selection.  Importing `pivot_core`
does not run an experiment or access sealed data.

```python
from pivot_core import PolicyTransition, PairedEvaluator, run_pivot_voi_round
```

`run_pivot_voi_round` is the promotion-gate entry point. It reuses the
agent-agnostic paired evaluator and posterior contracts; importing it does not
query a model or open a sealed task plane.

The V15 external study uses the same contracts through
`experiments/v15/` so historical experiment paths remain reproducible.
