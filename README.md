# PIVOT: Improvement Fidelity in Adaptive Worlds

Official code and reproducibility materials for **When Better Gets Worse:
Improvement Fidelity for Self-Improving Agents in Adaptive Worlds**.

[![Reproducibility checks](https://github.com/computational-decision-lab/pivot/actions/workflows/reproducibility.yml/badge.svg)](https://github.com/computational-decision-lab/pivot/actions/workflows/reproducibility.yml)

PIVOT evaluates proposed policy replacements under the worlds their deployment
induces. PIVOT-KG allocates paired high-fidelity observations to uncertainty
that can change the replacement decision.

## Reviewer guide

| Goal | Start here |
| --- | --- |
| Recompute the reported statistics | [`reproduction/run.py`](reproduction/run.py) and the commands below |
| Follow a short claim-to-evidence route | [`docs/reviewer-guide.md`](docs/reviewer-guide.md) |
| Understand what each figure uses | [`docs/figure-map.md`](docs/figure-map.md) |
| Inspect the PIVOT-KG versus Uniform comparison | [`docs/comparison-audit.md`](docs/comparison-audit.md) |
| Reproduce a particular cohort | [`docs/reproduction.md`](docs/reproduction.md) |
| Check protocols and unavailable inputs | [`docs/protocol-chronology.md`](docs/protocol-chronology.md) |
| Read the paper | [PDF](reproduction/manuscript/paper.pdf) · [LaTeX source](reproduction/manuscript/) |
| Download a frozen package | [ICLR 2027 reproducibility release v2](https://github.com/computational-decision-lab/pivot/releases/tag/iclr2027-repro-v2) |
| Verify release scope and tests | [`release/iclr2027-repro-v2/validation.json`](release/iclr2027-repro-v2/validation.json) |

The paper artifact is under `reproduction/`; the manuscript snapshot records its
Overleaf revision in `snapshot.json`. Directories such as
`experiments/v9`, `experiments/v15`, `results`, and `archive` preserve research
history and are not the primary reproduction entry points.

## Quick saved-evidence check

This offline path requires Python 3.10 on Linux. It needs no simulator, cloud
account, API key, or access to the authors' machines.

```bash
git clone https://github.com/computational-decision-lab/pivot.git
cd pivot
python3.10 -m venv .venv-replay
.venv-replay/bin/python -m pip install -r reproduction/environments/replay-requirements.txt
.venv-replay/bin/python -m pip install --no-deps -e .
.venv-replay/bin/python reproduction/run.py verify --output outputs/verify
.venv-replay/bin/python reproduction/run.py analyze --output outputs/analysis
.venv-replay/bin/python reproduction/run.py figures --output outputs/figures
```

After installation, the same three checks can be run with
`make reproduce-paper REPLAY_PYTHON=.venv-replay/bin/python`.

The commands verify 1,051 packaged evidence files, recompute the saved-result
aggregates, and create or check all ten manuscript figures. Expected checkpoints
include 51/90 disjoint optimal sets, the second-Leduc primary mean contrast
`+0.0298887`, and the unresolved MetaDrive contrast
`+1.88922 [-1.29334, 5.49203]`.

The first Leduc, Kuhn, and Melting Pot comparisons replay **180 saved
method/root decisions** in total. In those rows, PIVOT-KG and Uniform share the
fitted posterior, observation update, and terminal selection rule; query
allocation is the controlled method difference.

These commands replay saved observations. They do not constitute a new simulator
experiment or an independent full-study replication.

## Reproduction levels

| Level | Command | What it establishes |
| --- | --- | --- |
| Integrity | `python reproduction/run.py verify --output DIR` | Frozen input hashes and saved-result recomputation |
| Analysis | `python reproduction/run.py analyze --experiment NAME --output DIR` | Cohort-specific saved aggregates and comparison audits |
| Figures | `python reproduction/run.py figures --output DIR` | The ten paper figures from saved inputs or hash-pinned assets |
| Native smoke | `python reproduction/run.py smoke --experiment NAME --output DIR` | Bounded simulator/environment execution where dependencies are installed |
| Full cohort | `python reproduction/run.py full --experiment NAME --output NEW_DIR` | Frozen cohort launcher; environment- and compute-dependent |

Detailed commands and separate simulator environments are in
[`docs/reproduction.md`](docs/reproduction.md). Every output path must be outside
`evidence/paper`; packaged evidence is treated as immutable input.

## What is included

| Path | Contents |
| --- | --- |
| `reproduction/` | Reviewer-facing entry points, environment locks, figure code, and manuscript snapshot |
| `evidence/paper/` | Frozen observations, decisions, protocols, source snapshots, manifests, and hashes |
| `src/` | Maintained PIVOT metrics, transition, posterior, and acquisition code |
| `tests/release/` | Release and command-line acceptance tests |
| `tests/unit/` | Core algorithm tests |
| `docs/` | Reproduction guide, figure map, comparison audit, provenance, and evidence limits |
| `release/iclr2027-repro-v2/` | Validation, citation, manuscript, and comparison reports |
| `archive/` | Historical collaborator/local code snapshots retained for provenance |

## Paper-to-code map

| Paper component | Implementation or evidence |
| --- | --- |
| Improvement Fidelity metrics and transitions | [metrics](src/pivot/metrics/improvement.py), [transitions](src/pivot/core/transition.py) |
| Table 1 Gaussian posterior, KG, and terminal selection | [frozen selector](evidence/paper/source/core/pivot_v2.py) |
| Table 1 matched comparison | [decision replay](reproduction/audit_comparisons.py), [comparison audit](docs/comparison-audit.md) |
| Decision relevance and ties | [saved-evidence verifier](reproduction/verify_saved_evidence.py) |
| HighwayEnv redesign | [frozen runner and protocol](reproduction/highway_redesign/) |
| MetaDrive stress test | [figure generator](reproduction/figures/source/replot_metadrive.py), [frozen experiment code](evidence/paper/metadrive/code/) |

See the [directory guide](docs/repository-map.md) for historical paths. The
frozen per-cohort selectors are the authority for the reported results;
development implementations under `src/` are not interchangeable with them.

## Verified scope

The released commit has passed the GitHub reproducibility workflow, including
dependency installation, evidence verification, saved-result analysis, figure
generation, and selected core/release tests. The anonymous ZIP was also installed
and checked in a separate environment. Exact commands and outcomes are recorded
in [`release/iclr2027-repro-v2/validation.json`](release/iclr2027-repro-v2/validation.json).

Known limits are explicit:

- Full experimental cohorts were not rerun during packaging.
- The exact historical Melting Pot rerun needs three absent specialist-model
  files and the original frozen confirmation protocol. Saved decisions remain
  verifiable; missing hashes are documented.
- The recorded historical OpenSpiel version differs from the version used for
  bounded native smoke validation, so byte-level historical equivalence is not
  claimed.
- MetaDrive is boundary evidence: the prespecified `B=2` contrast remains
  unresolved because its confidence interval crosses zero.

## Release packages and checksums

The [versioned release](https://github.com/computational-decision-lab/pivot/releases/tag/iclr2027-repro-v2)
contains:

- `pivot-iclr2027-repro-v2-anonymous.zip` — reviewer-facing current-paper package;
- `pivot-iclr2027-repro-v2-full.zip` — current materials plus historical source;
- compiled manuscript, validation report, package manifest, and SHA-256 file.

To rebuild the ZIPs outside the checkout:

```bash
python scripts/build_submission_release.py --output /tmp/pivot-release
```

Third-party notices are preserved. The repository does not grant a new blanket
license over collaborator or third-party code; see
[`docs/rights-and-dependencies.md`](docs/rights-and-dependencies.md).

## Citation

During review, cite the paper through its ICLR submission. A formal BibTeX entry
will be added after the proceedings metadata is available.
