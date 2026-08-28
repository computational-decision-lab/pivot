# PIVOT Core

This component name denotes the agent-agnostic protocol layer.  The
maintained implementation is exposed as the `pivot_core` facade in
`src/pivot_core/` and delegates to the tested `src/pivot/` modules.

It contains policy transitions, paired evaluators, update footprints, the
differential posterior, and EVSI-per-cost selection.  Importing `pivot_core`
does not run an experiment or access sealed data.

```python
from pivot_core import PolicyTransition, PairedEvaluator, select_pivot_voi
```

The V15 external study uses the same contracts through
`experiments/v15/` so historical experiment paths remain reproducible.
