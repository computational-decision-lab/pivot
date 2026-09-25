# Reviewer guide

This page gives a short route through the artifact without requiring familiarity
with the repository's development history.

## Recommended reading order

1. Read the repository `README.md` for scope and the quick command path.
2. Run `python reproduction/run.py verify --output outputs/verify`.
3. Inspect `docs/comparison-audit.md` for what differs between PIVOT-KG and
   Uniform in each cohort.
4. Use `docs/figure-map.md` to trace a manuscript figure to its input and code.
5. Use `docs/reproduction.md` only when a simulator rerun or a cohort-specific
   environment is needed.

## Claim-to-evidence index

| Manuscript claim or result | Fast check | Detailed evidence |
| --- | --- | --- |
| 51/90 roots have disjoint proxy/deployment optimal sets | `reproduction/run.py verify` | `reproduction/verify_saved_evidence.py`, `evidence/paper/` |
| Table 1 PIVOT-KG vs Uniform changes query allocation under matched updating/selection | `reproduction/audit_comparisons.py` | `docs/comparison-audit.md`, frozen selectors in `evidence/paper/source/core/` |
| Second-Leduc primary contrast is `+0.0298887` | `reproduction/run.py verify` | `reproduction/figures/data/leduc/summary.json` |
| Highway redesign reduces ISR at `B=4` | `reproduction/run.py verify` | `reproduction/highway_redesign/`, frozen redesign evidence |
| MetaDrive `B=2` is unresolved | `reproduction/run.py verify` | `reproduction/figures/data/metadrive/`, `replot_metadrive.py` |
| Ten figures match their declared provenance | `reproduction/run.py figures` | `docs/figure-map.md`, generated `figure-manifest.json` |

## Command scope

- `verify`, `analyze`, and `figures` are offline saved-evidence operations.
- `smoke` is a small native environment check, not a full experiment.
- `full` starts a frozen cohort launcher and can require substantial compute,
  simulator assets, and a separate environment.
- A passing saved-evidence replay does not mean every historical cohort was
  independently rerun.

## Artifact boundaries

The paper-facing artifact is `reproduction/` plus `evidence/paper/`. The
`archive/`, historical experiment-version directories, and old reports are
retained for provenance. They should not be substituted for the frozen protocol
named by the current reproduction runner.

The exact historical Melting Pot full rerun remains unavailable because the
delivered research materials lack three specialist SavedModel payloads and the
original frozen confirmation protocol. Exact expected hashes are in
`docs/protocol-chronology.md`; saved decision replay and the bounded native smoke
remain available.
