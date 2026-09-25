# PIVOT: Improvement Fidelity in Adaptive Worlds

Code and reproducibility materials for **When Better Gets Worse: Improvement
Fidelity for Self-Improving Agents in Adaptive Worlds**.

PIVOT evaluates proposed policy replacements under the worlds their deployment
induces. PIVOT-KG allocates paired high-fidelity observations to uncertainty that
can change the replacement decision.

## Quick start: recompute the saved evidence

The offline path requires Python 3.10 on Linux and no simulator, cloud account,
API key, or access to the original authors' machines.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r reproduction/environments/replay-requirements.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python reproduction/run.py verify --output outputs/verify
.venv/bin/python reproduction/run.py analyze --output outputs/analysis
.venv/bin/python reproduction/run.py figures --output outputs/figures
```

Verification checks the supplied evidence and recomputes the tie-aware 51/90
optimal-set disagreements, Leduc's primary mean contrast of 0.0298887, and
MetaDrive's primary contrast of 1.88922 with interval [-1.29334, 5.49203].
MetaDrive remains unresolved. These checks replay saved observations; they do
not replace a fresh simulator run.

## Reproduce an experiment

[The reproduction guide](docs/reproduction.md) gives separate environments,
commands, frozen protocols, and limitations for controlled worlds, OpenSpiel
(Kuhn and Leduc), MeltingPot, HighwayEnv, and MetaDrive. Use `smoke` for a small
native check and `full --experiment NAME` for a complete frozen experiment.
Outputs always go to a separate directory; the distributed evidence is an input.

The [figure map](docs/figure-map.md) connects the ten manuscript figures to
their inputs and scripts. It distinguishes statistical regeneration from
replaying a supplied, hash-checked figure asset. The user-supplied Figure 4 is
retained without changing its data or layout.

## Repository layout

| Directory | Contents |
| --- | --- |
| `src/`, `experiments/`, `configs/` | Existing libraries, runners, and configurations; historical interfaces retained |
| `reproduction/` | Current paper's entry points, environments, figures, and read-only manuscript snapshot |
| `evidence/paper/` | Frozen observations, decisions, protocols, source snapshots, and hashes |
| `tests/` | Library tests and release entry-point checks |
| `docs/` | Reproduction instructions, figure map, provenance, and verification scope |
| `archive/` | Versioned research code, including the collaborator's server code |
| `release/iclr2027-repro-v1/` | Release specification and actual verification reports |
| `paper/` | Existing Overleaf synchronization destination; separate from release assembly |

Start with `reproduction/`, not the older V9/V15 development release commands.
Earlier implementations and cohorts remain available for provenance, but are
not interchangeable with the current paper's protocols or claims. See
[code provenance](docs/code-provenance.md) for the collaborator import and
local-versus-remote source differences.

## Release packages

The release builder creates two ZIPs with file manifests and SHA-256 checksums:

- **Full research package:** current reproducibility materials and historical
  research code, including versioned collaborator contributions.
- **Anonymous submission package:** the current paper's runnable materials,
  evidence, and documentation, without Git history or author repository links.

```bash
python scripts/build_submission_release.py --output /tmp/pivot-release
```

The public repository does not itself serve as an anonymous review link. Use
the anonymous ZIP for the conference's supplementary upload. The applicable
requirements are described in the [ICLR 2027 author guide](https://iclr.cc/Conferences/2027/AuthorGuidelines).

## Validation and limitations

[Release validation](release/iclr2027-repro-v1/validation.json) records the
commands actually executed, results, environment, and any unavailable checks.
Full simulator cohorts are not rerun as part of packaging. Historical source
and saved-result availability vary by cohort; no missing run is substituted
with synthetic data. Frozen results are not edited to match a rerun.
Melting Pot's exact frozen confirmation protocol and three specialist model
files were not present in the delivered archives. Saved results can be checked;
a complete historical rerun requires restoring those inputs. See the
[protocol chronology and missing-input record](docs/protocol-chronology.md).

Third-party notices are preserved. No new blanket software license is asserted
for collaborator or third-party code; see [rights and dependencies](docs/rights-and-dependencies.md).

中文说明：本次整理保留各历史研究版本，并将当前论文复现入口集中到
`reproduction/`。完整包用于研究归档；匿名包用于投稿附件。已保存结果重算、
最小模拟器验证和完整实验重跑会分别标明，不混为同一种复现结论。
