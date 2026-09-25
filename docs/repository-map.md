# Directory guide

The paper's current reproduction path uses `reproduction/` and `evidence/paper/`.
Historical paths remain in place where experiment scripts depend on them.

| Paths | Status and use |
| --- | --- |
| `reproduction/run.py`, `reproduction/environments/` | Current reviewer entry point and separate dependency locks |
| `reproduction/manuscript/` | Compiled paper, source, and source-revision/hash record |
| `reproduction/figures/` | Ten paper figures, frozen inputs, generators, and reference assets |
| `evidence/paper/` | Immutable observations, saved decisions, protocols, and cohort-specific source |
| `docs/`, `release/iclr2027-repro-v2/` | Review guides and version-specific validation records |
| `src/`, `tests/` | Shared libraries and their tests; reported cohorts may use frozen source copies |
| `archive/server-code-20260926/`, `archive/local-code-20260926/` | Supplied server code and independent local snapshot, with source manifests |
| `docs/archive/v15/root-reports-20260926/` | Ten former root-level reports, moved byte-for-byte with a mapping manifest |
| `experiments/`, `configs/`, `results/`, `figures/`, `tables/` | Historical experiment families and their outputs; use only when directed by a frozen protocol |
| `paper/` | Legacy bidirectional Overleaf integration directory; its older README/build scripts are not the current reviewer entry point |
| `artifacts/`, `snapshot/`, `research/`, `benchmarks/` | Historical development and benchmark records |
| `pivot-core/`, `pivot-inspect/` | Short descriptions of older library facades |

For a compact checkout without historical directories, use the anonymous
supplement ZIP. It has no Git history and includes the current runnable code,
frozen evidence, cohort protocols, and reviewer instructions.
