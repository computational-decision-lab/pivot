# Current paper reproduction

Start with the [reviewer guide](../docs/reviewer-guide.md) or the
[installation and experiment guide](../docs/reproduction.md).

| Entry | Purpose |
| --- | --- |
| [`run.py`](run.py) | `verify`, `analyze`, `figures`, `smoke`, and `full` commands |
| [`audit_comparisons.py`](audit_comparisons.py) | Replay 180 saved Table 1 decisions with matched posterior/selection |
| [`environments/`](environments/) | Separate replay and simulator environments |
| [`figures/`](figures/) | Figure provenance, immutable plotting inputs, and generators |
| [`manuscript/`](manuscript/) | Current paper snapshot and PDF |
| [`highway_redesign/`](highway_redesign/) | Frozen redesign runner and supporting protocol |

Saved observations live in [`evidence/paper/`](../evidence/paper/). Always choose
a separate output directory. Full cohort reruns are distinct from the offline
saved-evidence checks; missing inputs are listed in the reproduction guide.
