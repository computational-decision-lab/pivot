# V15 Implementation Matrix

This matrix is the final engineering boundary for the modern-agent upgrade.
`IMPLEMENTED_LOCAL` means the protocol, code path, tests, and DEV artifact
contract exist in this repository. `CONFIRMATORY_REQUIRED` means that the
code is ready but the claim cannot be made until the frozen external run is
authorized and completed.

| Requirement | Status | Evidence |
|---|---|---|
| Pre-modern-agent fallback snapshot | `IMPLEMENTED_LOCAL` | `snapshot/v15_pre_modern_agent/` and `V15_BASELINE_SNAPSHOT.md` |
| Inspect control-plane probe | `IMPLEMENTED_LOCAL` | `experiments/v15/control_plane.py`, `pivot_inspect` |
| mini-SWE execution adapter | `IMPLEMENTED_LOCAL` | `experiments/v15/external_runtime.py`, pinned runtime |
| Pi cross-scaffold adapter | `IMPLEMENTED_LOCAL` | `experiments/v15/run_pi_replication.py`, `pivot_inspect` |
| Two independent proposal operators | `IMPLEMENTED_LOCAL` | `experiments/v15/external_operators.py` |
| Sealed proxy/gate/assessment planes | `IMPLEMENTED_LOCAL` | `experiments/v15/planes.py`, leakage tests |
| Fresh paired sandboxes and budget gate | `IMPLEMENTED_LOCAL` | `experiments/v15/external_runtime.py`, `sandbox.py` |
| Frozen candidate archive | `IMPLEMENTED_LOCAL` | `experiments/v15/evidence.py`, archive manifest |
| PIVOT-VOI promotion replay | `IMPLEMENTED_LOCAL` | `experiments/v15/external_promotion.py`, promotion artifacts |
| Closed-loop terminal assessment contract | `IMPLEMENTED_LOCAL` | `experiments/v15/external_closed_loop.py` |
| Non-LLM strategic response | `IMPLEMENTED_LOCAL` | `experiments/v15/run_strategic.py`, DEV response artifact |
| Independent-agent strategic response path | `IMPLEMENTED_LOCAL` | `experiments/v15/agent_response.py`, identity-blind schema |
| Registered ablations | `IMPLEMENTED_LOCAL` | `experiments/v15/run_ablations.py` |
| Canonical tables and provenance | `IMPLEMENTED_LOCAL` | `experiments/v15/canonical.py`, Parquet/CSV schemas |
| Figure render/view/audit lifecycle | `IMPLEMENTED_LOCAL` | `figures/v15/`, visual review manifest |
| Manuscript, supplement, release, and audits | `IMPLEMENTED_LOCAL` | `paper/iclr2027/`, `release/v15/`, audit reports |
| Confirmatory mini-SWE transition audit | `CONFIRMATORY_REQUIRED` | Explicitly unopened; terminal state is not inferred from DEV |
| Confirmatory promotion and closed loop | `CONFIRMATORY_REQUIRED` | Explicitly unopened; assessment remains untouched |
| Confirmatory Pi replication | `CONFIRMATORY_REQUIRED` | DEV replication only; no cross-scaffold claim promoted |
| Confirmatory strategic validation | `CONFIRMATORY_REQUIRED` | DEV mutation response exists; external response claim remains closed |

The release is therefore an honest, reproducible engineering handoff with a
`BLOCKED` scientific final status until the external confirmation gates are
opened under the frozen lock.  No result-dependent branch changes the
protocol, tasks, operators, or figures.
