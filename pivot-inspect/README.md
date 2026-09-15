# PIVOT Inspect Boundary

This component name denotes the Inspect-controlled external-agent boundary.
The maintained Python facade is `pivot_inspect` in `src/pivot_inspect/`.

It exposes dry-run adapter contracts and the explicit paired execution bridge
for Inspect AI, mini-SWE-agent, and Pi.  Importing the facade never invokes a
model, opens a sealed task plane, or starts a container.

```python
from pivot_inspect import InspectControlPlane, MiniSWEAdapter, PiAdapter
```

Use the commands in `experiments/v15/__main__.py` for DEV or explicitly
authorized confirmatory execution.
